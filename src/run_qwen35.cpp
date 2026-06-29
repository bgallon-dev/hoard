// M2/M3: streaming MoE decode. Non-expert weights resident in VRAM; EXPERT weights live in
// host RAM and stream into a bounded per-layer VRAM slot pool on demand.
//   run_moe_stream <model> <ids_csv> <ngen> <mode> [K]
//     mode = naive : K=8 slots/layer, re-copy the 8 selected experts every token
//     mode = cache : K slots/layer, persistent LRU residency cache (copy only on miss)
// Prints generated tokens + metrics: peak VRAM, decode tok/s, expert copies, cache hit rate.
//
// Per layer per token: graph A (attn w/ KV cache + router) -> read top-8 -> stream experts
// into slots -> graph B (expert FFN over slots) -> host weighted-sum -> next layer.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <string>
#include <vector>
#include <map>
#include <list>
#include <set>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <algorithm>
#include <random>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <functional>
#include <iostream>
#include "llama.h"                 // vocab/tokenizer for the chat REPL (links llama.dll)

#ifndef GGML_ROPE_TYPE_NEOX
#define GGML_ROPE_TYPE_NEOX 2
#endif
// hparams: now read from the GGUF (parameterized so the engine generalizes past OLMoE)
static int n_layer,n_embd,n_head,n_head_kv,head_dim,n_ff,n_expert,n_used,n_vocab,n_ctx_orig,rope_type=2/*NEOX*/;
static int n_embd_attn,qk_norm_dim=0;     // n_embd_attn=n_head*head_dim (Qwen3: 4096 != n_embd 2048)
static float eps,freq_base;
static bool has_qk_norm=false;            // OLMoE/Qwen3 yes; Qwen2/Mixtral no
static bool qk_perhead=false;             // Qwen3: QK-norm per-head over head_dim (post-reshape); OLMoE: full pre-reshape
static bool wnorm=false;                  // norm_topk_prob: renormalize top-k router weights to sum 1 (Qwen3=true, OLMoE=false)
static int  eos_id=50279;                 // read from GGUF tokenizer.ggml.eos_token_id (OLMoE 50279, Qwen3 151645)
// --- qwen35moe / qwen3next: hybrid Gated Delta Net (linear attn) + full attention ---
static bool is_qwen35=false;     // either hybrid-GDN arch (qwen35moe OR qwen3next)
static bool is_qwen3next=false;  // qwen3next only: fused ssm_ba (vs separate beta/alpha), NEOX rope (vs IMRoPE mrope)
static int  ssm_d_conv=4,ssm_d_inner=0,ssm_d_state=0,ssm_dt_rank=0,ssm_n_group=0; // GDN linear-attn dims
static int  n_ff_shexp=0;                  // shared-expert FFN length (0=none)
static int  full_attn_interval=4;          // every Nth layer (1-indexed) is full attention
static std::vector<char> is_recr;          // per layer: 1=linear(GDN recurrent), 0=full attention
static std::atomic<bool> g_stop{false};    // SERVE: interrupt the current generation (Stop button)
static float g_temp=0.0f;                  // SERVE: sampling temperature (0 = greedy argmax)
static std::mt19937 g_rng((unsigned)time(nullptr));  // sampler RNG (advances across turns -> regenerate varies)
static int  rope_sections[4]={0,0,0,0};
static std::string ARCH;
static uint32_t gku32(gguf_context*g,const std::string&k,uint32_t d){int64_t i=gguf_find_key(g,k.c_str());return i<0?d:gguf_get_val_u32(g,i);}
static float    gkf32(gguf_context*g,const std::string&k,float    d){int64_t i=gguf_find_key(g,k.c_str());return i<0?d:gguf_get_val_f32(g,i);}

// ---- non-expert weights (VRAM) ----
struct Model {
    gguf_context* gguf=nullptr; ggml_context* ctx=nullptr;
    ggml_backend_t backend=nullptr; ggml_backend_buffer_t wbuf=nullptr;
    std::map<std::string, ggml_tensor*> ten;
    ggml_tensor* get(const std::string& n){ auto it=ten.find(n); if(it==ten.end()){fprintf(stderr,"MISSING %s\n",n.c_str());exit(1);} return it->second; }
    ggml_tensor* blk(int il,const char* s){ char b[80]; snprintf(b,sizeof b,"blk.%d.%s",il,s); return get(b); }
};
// ---- expert weights: stored ON DISK; tiers (RAM cache, VRAM slots) sit above ----
struct HostExperts {
    FILE* f=nullptr;                                   // the GGUF file (NVMe-backed)
    std::vector<size_t> goff,uoff,doff;                // file offset of each layer's gate/up/down_exps DATA
    std::vector<size_t> gstride,ustride,dstride;       // per-expert byte stride
    std::vector<ggml_type> gtype,utype,dtype;          // PER LAYER (mixed quant across layers)
    // read expert e's gate/up/down bytes from disk into dst (3 reads). Returns bytes read.
    size_t read(int il,int e,uint8_t* gbuf,uint8_t* ubuf,uint8_t* dbuf){
        _fseeki64(f,(int64_t)(goff[il]+(size_t)e*gstride[il]),SEEK_SET); fread(gbuf,1,gstride[il],f);
        _fseeki64(f,(int64_t)(uoff[il]+(size_t)e*ustride[il]),SEEK_SET); fread(ubuf,1,ustride[il],f);
        _fseeki64(f,(int64_t)(doff[il]+(size_t)e*dstride[il]),SEEK_SET); fread(dbuf,1,dstride[il],f);
        return gstride[il]+ustride[il]+dstride[il];
    }
};

// ---- device-direct (NO_BUFFERING) parallel expert reads: bypass page cache + saturate NVMe queue depth ----
// measured: drive ceiling ~3.5 GB/s, saturates at QD4; buffered fread path was stuck at ~1 GB/s.
// read the aligned superset [aoff,aoff+asize) DIRECTLY into dst (an aligned RT buffer) - no scratch, no memcpy.
// real data lands at dst[head .. head+rsize).
struct IOJob { uint8_t* dst; uint64_t aoff; uint32_t asize; uint32_t head; uint32_t rsize; };
// Persistent worker pool. The previous version spawned `nw` std::threads on EVERY run_jobs call
// (once per layer => 48x/token), each doing ~1 read then exiting. iobench shows the 970 EVO gives
// 3.52 GB/s at QD8 on 512KB random reads, but per-call spawn/join collapsed the effective queue
// depth to ~2-3 and capped realized bandwidth at ~2.08 GB/s. Spawning the workers ONCE and feeding
// them per-batch keeps QD8 in flight => recovers the full ~3.5 GB/s. Lossless.
struct DirectIO {
    int n=0; std::vector<HANDLE> h; std::vector<FILE*> bf; uint64_t fsize=0; std::string fpath;
    std::vector<std::thread> pool;
    std::mutex m; std::condition_variable cv_work, cv_done;
    std::vector<IOJob>* cur=nullptr; int batch_total=0; uint64_t gen=0; bool stop=false;
    std::atomic<int> next_idx{0}, done_cnt{0}, ready{0};
    void init(const char* p,int nn){
        n=nn; fpath=p;
        HANDLE q=CreateFileA(p,GENERIC_READ,FILE_SHARE_READ,NULL,OPEN_EXISTING,0,NULL);
        LARGE_INTEGER li; GetFileSizeEx(q,&li); fsize=(uint64_t)li.QuadPart; CloseHandle(q);
        h.assign(n,INVALID_HANDLE_VALUE);
        for(int i=0;i<n;++i) bf.push_back(fopen(p,"rb"));                                          // buffered EOF fallback
        for(int w=0;w<n;++w) pool.emplace_back([this,w]{ worker(w); });                           // spawn ONCE
        while(ready.load()<n) std::this_thread::yield();                                           // wait for all worker handles open
    }
    void doread(int w,const IOJob& j){
        if(j.aoff+j.asize<=fsize){ LARGE_INTEGER li; li.QuadPart=(LONGLONG)j.aoff;
            SetFilePointerEx(h[w],li,NULL,FILE_BEGIN); DWORD got=0; ReadFile(h[w],j.dst,j.asize,&got,NULL); }   // direct into RT buffer
        else { _fseeki64(bf[w],(int64_t)(j.aoff+j.head),SEEK_SET); fread(j.dst+j.head,1,j.rsize,bf[w]); }         // last-sector: buffered
    }
    void worker(int w){
        h[w]=CreateFileA(fpath.c_str(),GENERIC_READ,FILE_SHARE_READ,NULL,OPEN_EXISTING,FILE_FLAG_NO_BUFFERING|FILE_FLAG_RANDOM_ACCESS,NULL); // open in OWN thread
        ready.fetch_add(1);
        uint64_t seen=0;
        for(;;){
            { std::unique_lock<std::mutex> lk(m); cv_work.wait(lk,[&]{return stop||gen!=seen;}); if(stop) return; seen=gen; }
            for(;;){ int j=next_idx.fetch_add(1); if(j>=batch_total) break; doread(w,(*cur)[j]); }
            if(done_cnt.fetch_add(1)+1==n){ { std::lock_guard<std::mutex> lk(m); } cv_done.notify_one(); } // last finisher wakes run()
        }
    }
    void run(std::vector<IOJob>& jobs){
        if(jobs.empty()) return;
        int total=(int)jobs.size();
        if(n<=1){ for(int j=0;j<total;++j) doread(0,jobs[j]); return; }
        { std::lock_guard<std::mutex> lk(m); cur=&jobs; batch_total=total; next_idx.store(0); done_cnt.store(0); ++gen; }
        cv_work.notify_all();                                                                     // wake the warm pool
        { std::unique_lock<std::mutex> lk(m); cv_done.wait(lk,[&]{return done_cnt.load()==n;}); }  // barrier: all workers drained
    }
    ~DirectIO(){ { std::lock_guard<std::mutex> lk(m); stop=true; ++gen; } cv_work.notify_all(); for(auto&t:pool) if(t.joinable()) t.join(); }
};
static void run_jobs(DirectIO& io, std::vector<IOJob>& jobs){ io.run(jobs); }
// ---- explicit RAM tier: bounded LRU cache of expert byte-buffers (warm experts) ----
struct RamTier {
    int cap=0;                                         // max experts held in RAM
    std::map<int,int> key2slot;                        // (il*n_expert+e) -> ram slot
    std::vector<int> slot2key;                         // ram slot -> key (-1 empty)
    std::list<int> lru;                                // ram slots, front=recent
    std::vector<uint8_t*> gbuf,ubuf,dbuf;              // [ram slot] 4096-aligned buffers (lazy VirtualAlloc, sized to max stride+slack)
    std::vector<uint32_t> ghead,uhead,dhead;           // byte offset of real data within the aligned buffer (for device-direct reads)
};

static bool is_exp(const char* n){ return strstr(n,"_exps")!=nullptr; }

static void load_model(Model& M, HostExperts& H, const char* path, ggml_backend_t backend) {
    M.backend=backend;
    ggml_context* all=nullptr;
    gguf_context* gguf=gguf_init_from_file(path,{true,&all});
    if(!gguf){fprintf(stderr,"gguf fail\n");exit(1);}

    // ---- read hparams from GGUF metadata (generic over arch) ----
    { int64_t ai=gguf_find_key(gguf,"general.architecture"); ARCH = ai<0?"":gguf_get_val_str(gguf,ai); }
    auto KEY=[&](const char* s){ return ARCH+"."+s; };
    n_layer    = (int)gku32(gguf,KEY("block_count"),0);
    n_embd     = (int)gku32(gguf,KEY("embedding_length"),0);
    n_head     = (int)gku32(gguf,KEY("attention.head_count"),0);
    n_head_kv  = (int)gku32(gguf,KEY("attention.head_count_kv"),n_head);
    n_ff       = (int)gku32(gguf,KEY("feed_forward_length"),0);
    n_ff       = (int)gku32(gguf,KEY("expert_feed_forward_length"),n_ff);  // MoE per-expert FFN if present
    n_expert   = (int)gku32(gguf,KEY("expert_count"),0);
    n_used     = (int)gku32(gguf,KEY("expert_used_count"),0);
    if(const char* tk=getenv("TOPK")){ int t=atoi(tk); if(t>0 && t<n_used){ fprintf(stderr,"[topk] inference-time top-k reduction: %d -> %d active experts/layer (router renormalizes the survivors)\n",n_used,t); n_used=t; } } // footprint test: fewer active experts -> fewer streamed bytes/token
    n_ctx_orig = (int)gku32(gguf,KEY("context_length"),4096);
    freq_base  = gkf32(gguf,KEY("rope.freq_base"),10000.0f);
    eps        = gkf32(gguf,KEY("attention.layer_norm_rms_epsilon"),1e-5f);
    int q_proj_out=0;
    for(ggml_tensor* t=ggml_get_first_tensor(all); t; t=ggml_get_next_tensor(all,t)){
        if(!strcmp(t->name,"token_embd.weight")) n_vocab=(int)t->ne[1];
        if(!strcmp(t->name,"blk.0.attn_q.weight")) q_proj_out=(int)t->ne[1];   // = n_head*head_dim
        if(!strcmp(t->name,"blk.0.attn_q_norm.weight")){ has_qk_norm=true; qk_norm_dim=(int)t->ne[0]; }
    }
    // head_dim: prefer key_length; else derive from q-proj shape (Qwen3 head_dim 128 != n_embd/n_head 64)
    head_dim = (int)gku32(gguf,KEY("attention.key_length"), (n_head&&q_proj_out)?q_proj_out/n_head:(n_head?n_embd/n_head:0));
    n_embd_attn = n_head*head_dim;                       // attention inner dim (may differ from n_embd)
    qk_perhead  = has_qk_norm && (qk_norm_dim==head_dim); // Qwen3 per-head vs OLMoE full-projection
    // norm_topk_prob: llama.cpp hardcodes norm_w per-arch (qwen3moe=true, olmoe=false). Not a GGUF key.
    wnorm = (ARCH=="olmoe") ? false : true;              // OLMoE is the no-renorm exception; modern MoE renormalize
    eos_id = (int)gku32(gguf,"tokenizer.ggml.eos_token_id",50279);
    // ---- qwen35moe / qwen3next (hybrid Gated Delta Net + full attention) ----
    is_qwen35    = (ARCH=="qwen35moe" || ARCH=="qwen3next"); // shared GDN+shexp engine
    is_qwen3next = (ARCH=="qwen3next");                      // fused ssm_ba + NEOX rope variant
    if(is_qwen35){
        ssm_d_conv  = (int)gku32(gguf,KEY("ssm.conv_kernel"),4);
        ssm_d_inner = (int)gku32(gguf,KEY("ssm.inner_size"),0);
        ssm_d_state = (int)gku32(gguf,KEY("ssm.state_size"),0);
        ssm_dt_rank = (int)gku32(gguf,KEY("ssm.time_step_rank"),0);   // = num_v_heads (32)
        ssm_n_group = (int)gku32(gguf,KEY("ssm.group_count"),0);      // = num_k_heads (16)
        n_ff_shexp  = (int)gku32(gguf,KEY("expert_shared_feed_forward_length"),0);
        full_attn_interval = (int)gku32(gguf,KEY("full_attention_interval"),4);
        { int64_t ki=gguf_find_key(gguf,KEY("rope.dimension_sections").c_str());   // array of 4 i32
          if(ki>=0 && gguf_get_arr_n(gguf,ki)>=4){ const int32_t* a=(const int32_t*)gguf_get_arr_data(gguf,ki); for(int s=0;s<4;++s) rope_sections[s]=a[s]; } }
        is_recr.assign(n_layer,0); int nl=0,nf=0;
        for(int il=0;il<n_layer;++il){ is_recr[il] = ((il+1)%full_attn_interval!=0)?1:0; if(is_recr[il])nl++; else nf++; }
        fprintf(stderr,"[%s] ssm conv=%d inner=%d state=%d dt_rank/Vheads=%d n_group/Kheads=%d | shexp_ff=%d | full_attn_every %d -> %d linear, %d full | rope=%s sec=%d,%d,%d,%d\n",
            ARCH.c_str(),ssm_d_conv,ssm_d_inner,ssm_d_state,ssm_dt_rank,ssm_n_group,n_ff_shexp,full_attn_interval,nl,nf,
            is_qwen3next?"NEOX(64)":"IMRoPE",rope_sections[0],rope_sections[1],rope_sections[2],rope_sections[3]);
    }
    fprintf(stderr,"[hparams2] n_embd_attn=%d qk_norm_dim=%d qk_perhead=%d wnorm=%d\n",n_embd_attn,qk_norm_dim,(int)qk_perhead,(int)wnorm);
    fprintf(stderr,"[hparams] arch=%s layers=%d embd=%d heads=%d/%d head_dim=%d n_ff=%d experts=%d/%d vocab=%d rope=%.0f eps=%.1e qknorm=%d\n",
        ARCH.c_str(),n_layer,n_embd,n_head,n_head_kv,head_dim,n_ff,n_expert,n_used,n_vocab,freq_base,eps,(int)has_qk_norm);
    if(n_layer<=0||n_embd<=0||n_expert<=0){fprintf(stderr,"unsupported/missing hparams\n");exit(1);}

    H.goff.resize(n_layer); H.uoff.resize(n_layer); H.doff.resize(n_layer);
    H.gstride.resize(n_layer); H.ustride.resize(n_layer); H.dstride.resize(n_layer);
    H.gtype.resize(n_layer); H.utype.resize(n_layer); H.dtype.resize(n_layer);

    // VRAM ctx with ONLY non-expert tensors
    ggml_context* nectx=ggml_init({(size_t)(n_layer*24+32)*ggml_tensor_overhead(),nullptr,true}); // qwen35moe: ~16 non-expert tensors/layer
    std::map<std::string,ggml_tensor*> nesrc;
    for(ggml_tensor* t=ggml_get_first_tensor(all); t; t=ggml_get_next_tensor(all,t)){
        if(is_exp(t->name)) continue;
        ggml_type ct=(t->type==GGML_TYPE_BF16)?GGML_TYPE_F32:t->type; // Vulkan has no bf16 kernel -> upconvert (qwen3next ffn_gate_inp_shexp)
        ggml_tensor* c=ggml_new_tensor(nectx,ct,GGML_MAX_DIMS,t->ne); ggml_set_name(c,t->name); nesrc[t->name]=c;
    }
    M.wbuf=ggml_backend_alloc_ctx_tensors(nectx,backend);

    // Load non-expert weights UNBUFFERED. Buffered reads here would fill the OS page cache, and cached
    // file pages route every later NO_BUFFERING expert read through the cache manager -> NVMe throughput
    // collapses 3.5 -> ~1.9 GB/s (bisected with pooltest2). Experts stay on disk and stream unbuffered.
    HANDLE hload=CreateFileA(path,GENERIC_READ,FILE_SHARE_READ,NULL,OPEN_EXISTING,FILE_FLAG_NO_BUFFERING|FILE_FLAG_SEQUENTIAL_SCAN,NULL);
    const size_t SCRATCH=32u*1024*1024; uint8_t* scratch=(uint8_t*)VirtualAlloc(NULL,SCRATCH,MEM_COMMIT|MEM_RESERVE,PAGE_READWRITE);
    auto uload=[&](uint64_t foff,size_t nb,uint8_t* dst){   // ALWAYS NO_BUFFERING (no buffered fallback -> page cache stays clean even for the EOF tensor)
        size_t done=0;
        while(done<nb){ uint64_t cur=foff+done; uint64_t a=cur&~(uint64_t)4095; uint32_t head=(uint32_t)(cur-a); size_t want=nb-done;
            size_t asize=(head+want+4095)&~(size_t)4095; if(asize>SCRATCH) asize=SCRATCH;   // reads past EOF return a short got (valid)
            LARGE_INTEGER li; li.QuadPart=(LONGLONG)a; SetFilePointerEx(hload,li,NULL,FILE_BEGIN);
            DWORD got=0; if(!ReadFile(hload,scratch,(DWORD)asize,&got,NULL) || got<=head) break;
            size_t avail=(size_t)got-head; size_t take=avail<want?avail:want;
            memcpy(dst+done,scratch+head,take); done+=take; if((DWORD)asize!=got) break; }   // got<asize => hit EOF, all available consumed
    };
    H.f=nullptr; size_t doff=gguf_get_data_offset(gguf); std::vector<uint8_t> buf; int n_bf16=0;
    for(ggml_tensor* t=ggml_get_first_tensor(all); t; t=ggml_get_next_tensor(all,t)){
        int64_t ti=gguf_find_tensor(gguf,t->name); size_t off=gguf_get_tensor_offset(gguf,ti), nb=ggml_nbytes(t);
        if(is_exp(t->name)){
            // EXPERTS STAY ON DISK: record file offset of the tensor data + per-expert stride
            int il=-1; char which[16]={0}; sscanf(t->name,"blk.%d.ffn_%15[^_]_exps",&il,which);
            size_t fileoff=doff+off;
            if(!strcmp(which,"gate")){ H.goff[il]=fileoff; H.gstride[il]=nb/n_expert; H.gtype[il]=t->type; }
            else if(!strcmp(which,"up")){ H.uoff[il]=fileoff; H.ustride[il]=nb/n_expert; H.utype[il]=t->type; }
            else if(!strcmp(which,"down")){ H.doff[il]=fileoff; H.dstride[il]=nb/n_expert; H.dtype[il]=t->type; }
        } else {
            buf.resize(nb); uload(doff+off,nb,buf.data());
            if(t->type==GGML_TYPE_BF16){ // upconvert bf16->f32 (high 16 bits of the f32); Vulkan lacks a bf16 path
                int64_t ne=ggml_nelements(t); std::vector<float> f32((size_t)ne); const uint16_t* s=(const uint16_t*)buf.data();
                for(int64_t i=0;i<ne;++i){ uint32_t bits=((uint32_t)s[i])<<16; memcpy(&f32[i],&bits,4); }
                ggml_backend_tensor_set(nesrc[t->name],f32.data(),0,(size_t)ne*4); n_bf16++;
            } else ggml_backend_tensor_set(nesrc[t->name],buf.data(),0,nb);
            M.ten[t->name]=nesrc[t->name];
        }
    }
    CloseHandle(hload); VirtualFree(scratch,0,MEM_RELEASE);   // release the unbuffered loader (experts stream via DirectIO)
    M.ctx=nectx; M.gguf=gguf;
    if(n_bf16) fprintf(stderr,"[bf16->f32] upconverted %d bf16 tensors (Vulkan has no bf16 path)\n",n_bf16);
    fprintf(stderr,"loaded: %zu non-expert VRAM tensors; experts ON DISK (per-expert ~%.1f MB)\n",
        M.ten.size(),(H.gstride[0]+H.ustride[0]+H.dstride[0])/1e6);
}

// ---- KV cache ----
struct KV { std::vector<ggml_tensor*> k,v; ggml_context* ctx=nullptr; ggml_backend_buffer_t buf=nullptr; };
// ---- per-layer expert slot pool (VRAM) ----
struct Slots {
    int K;
    // [layer][slot] tensors
    std::vector<std::vector<ggml_tensor*>> gate,up,down;
    ggml_context* ctx=nullptr; ggml_backend_buffer_t buf=nullptr;
    // residency: [layer] expert_id currently in each slot (-1 empty), and LRU order (list of slots)
    std::vector<std::vector<int>> slot_expert;      // [layer][slot] -> expert id
    std::vector<std::map<int,int>> expert_slot;     // [layer] expert -> slot
    std::vector<std::list<int>> lru;                // [layer] slots, front=most recent
};

// decode a base64 line (conversation content is sent base64 so multi-line/code text survives the line protocol)
static std::string b64dec(const std::string& in){
    static const std::string T="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out; int val=0, bits=-8;
    for(unsigned char c:in){ if(c=='=') break; size_t p=T.find((char)c); if(p==std::string::npos) continue;
        val=(val<<6)|(int)p; bits+=6; if(bits>=0){ out.push_back(char((val>>bits)&0xFF)); bits-=8; } }
    return out;
}

int main(int argc,char**argv){
    if(argc<3){fprintf(stderr,"usage: %s model ids_csv ngen mode[naive|cache] [K] [ram_experts]\n       %s model chat [K] [ram_experts]   (interactive text REPL)\n",argv[0],argv[0]);return 2;}
    bool chatmode = (std::string(argv[2])=="chat");
    int ngen; std::string mode; int Karg; std::vector<int32_t> seq;
    if(chatmode){ mode="chat"; ngen=1024; Karg=(argc>3)?atoi(argv[3]):0; }
    else {
        if(argc<5){fprintf(stderr,"usage: %s model ids_csv ngen mode[naive|cache] [K] [ram_experts]\n",argv[0]);return 2;}
        ngen=atoi(argv[3]); mode=argv[4]; Karg=(argc>5)?atoi(argv[5]):0;
        char* s=strdup(argv[2]); for(char* p=strtok(s,","); p; p=strtok(nullptr,",")) seq.push_back(atoi(p)); free(s);
    }
    if(const char* sl=getenv("SYNTH_LEN")){   // prefill benchmark: synthetic diffuse-routing prompt of N tokens
        int n=atoi(sl); seq.clear(); uint32_t r=2463534242u;
        for(int i=0;i<n;++i){ r^=r<<13; r^=r>>17; r^=r<<5; seq.push_back((int32_t)(r%100000u)); }  // xorshift32 pseudo-random ids -> diffuse routing
        fprintf(stderr,"[synth] %d-token synthetic prompt (xorshift, diffuse routing)\n",n);
    }
    int T=(int)seq.size();
    // teacher-forcing validation: TFLEN=<prompt_len> -> treat first TFLEN ids as prompt,
    // follow the rest of seq verbatim (don't substitute argmax), dump top-5 logits per tip.
    const char* tflen=getenv("TFLEN"); bool tf=false; if(tflen){ T=atoi(tflen); tf=true; }
    bool dumpL = tf || getenv("DUMP")!=nullptr;

    ggml_backend_load_all();
    ggml_backend_dev_t vk0=nullptr;
    for(size_t i=0;i<ggml_backend_dev_count();++i){ggml_backend_dev_t d=ggml_backend_dev_get(i); if(!strcmp(ggml_backend_dev_name(d),"Vulkan0")){vk0=d;break;}}
    ggml_backend_t backend=ggml_backend_dev_init(vk0,nullptr);
    Model M; HostExperts H; load_model(M,H,argv[1],backend);
    // device-direct parallel reader: 8 workers (>= QD4 saturation), scratch sized to the largest expert sub-read.
    int ioqd = getenv("IOQD")?atoi(getenv("IOQD")):8; if(ioqd<1) ioqd=1;   // IOQD=N -> vary effective NVMe queue depth
    DirectIO io; io.init(argv[1],ioqd);
    size_t gbsz,ubsz,dbsz; { size_t mg=0,mu=0,md=0; for(int il=0;il<n_layer;++il){ if(H.gstride[il]>mg)mg=H.gstride[il]; if(H.ustride[il]>mu)mu=H.ustride[il]; if(H.dstride[il]>md)md=H.dstride[il]; }
        auto rup=[](size_t v){ return (v+8192+4095)&~(size_t)4095; }; gbsz=rup(mg); ubsz=rup(mu); dbsz=rup(md);
        fprintf(stderr,"[directIO] 8 workers, reads land DIRECTLY in aligned RT buffers (no scratch/memcpy); buf g/u/d=%zu/%zu/%zu KB\n",gbsz/1024,ubsz/1024,dbsz/1024); }
    if(const char* st=getenv("IOSELFTEST")){   // diagnostic: realized pool bandwidth (random 512KB reads) vs batch size; should hit the drive's ~3.5 GB/s with a clean page cache
        int M=atoi(st); if(M<=0)M=4096; const uint32_t BS=524288;
        std::vector<uint8_t*> bufs(M); std::vector<IOJob> all(M); uint64_t maxoff=io.fsize-BS-4096;
        for(int i=0;i<M;++i) bufs[i]=(uint8_t*)VirtualAlloc(NULL,BS,MEM_COMMIT|MEM_RESERVE,PAGE_READWRITE);
        // bufmode: 0=distinct cold buffers, 1=distinct pre-faulted (memset), 2=only 8 buffers reused (iobench-like)
        auto bw=[&](int chunk,unsigned seed,int bufmode)->double{
            for(int i=0;i<M;++i){ seed=seed*1103515245u+12345u; uint64_t off=(((uint64_t)seed*4096ull)%maxoff)&~(uint64_t)4095;
                uint8_t* d = (bufmode==2)? bufs[i&7] : bufs[i]; all[i]=IOJob{d,off,BS,0,BS}; }
            if(bufmode==1) for(int i=0;i<M;++i) memset(bufs[i],0,BS);   // pre-fault every page resident
            struct timespec a,c; clock_gettime(CLOCK_MONOTONIC,&a);
            if(chunk>=M) run_jobs(io,all);
            else for(int o=0;o<M;o+=chunk){ int n=(o+chunk<=M)?chunk:(M-o); std::vector<IOJob> b(all.begin()+o,all.begin()+o+n); run_jobs(io,b); }
            clock_gettime(CLOCK_MONOTONIC,&c); double sec=(c.tv_sec-a.tv_sec)+(c.tv_nsec-a.tv_nsec)/1e9; return (double)M*BS/1e9/sec; };
        fprintf(stderr,"[ioselftest] %d x 512KB random reads through the persistent pool (QD=%d):\n",M,ioqd);
        fprintf(stderr,"   big batch, distinct COLD buffers     : %.2f GB/s\n", bw(M,11u,0));
        fprintf(stderr,"   big batch, distinct PRE-FAULTED bufs : %.2f GB/s\n", bw(M,13u,1));
        fprintf(stderr,"   big batch, only 8 buffers reused     : %.2f GB/s\n", bw(M,17u,2));
        fprintf(stderr,"   batches of 9, PRE-FAULTED buffers    : %.2f GB/s\n", bw(9,37u,1));
        return 0;
    }
    int K = Karg?Karg:(mode=="naive"?n_used:16);   // naive needs >= n_used slots
    int total_experts = n_layer*n_expert;
    int ram_cap = chatmode ? ((argc>4)?atoi(argv[4]):total_experts) : ((argc>6)?atoi(argv[6]):total_experts);   // RAM tier size
    if(ram_cap>total_experts) ram_cap=total_experts; if(ram_cap<n_used) ram_cap=n_used;

    int max_kv = chatmode ? 4096 : (tf ? (int)seq.size()+2 : T+ngen+2); // tf: full given sequence
    if(const char* mk=getenv("MAXKV")){ int v=atoi(mk); if(v>max_kv) max_kv=v; } // bench: size KV for a target context (VRAM-cliff sweep) without generating it
    KV kv; kv.ctx=ggml_init({(size_t)2*n_layer*ggml_tensor_overhead()+1024,nullptr,true});
    for(int il=0;il<n_layer;++il){ bool need_kv = !is_qwen35 || !is_recr[il]; // qwen35: only full-attn layers use KV
        kv.k.push_back(need_kv?ggml_new_tensor_3d(kv.ctx,GGML_TYPE_F32,head_dim,n_head_kv,max_kv):nullptr);
        kv.v.push_back(need_kv?ggml_new_tensor_3d(kv.ctx,GGML_TYPE_F32,head_dim,n_head_kv,max_kv):nullptr); }
    kv.buf=ggml_backend_alloc_ctx_tensors(kv.ctx,backend);

    // ---- P2: qwen35moe recurrent state (Gated Delta Net) ----
    // per linear layer: GDN state [S_v,S_v,H_v] + conv rolling window [d_conv-1, conv_dim]. persistent VRAM, zeroed per gen.
    int gdn_Sv=ssm_d_state, gdn_Hv=ssm_dt_rank, gdn_Hk=ssm_n_group;          // 128, 32, 16
    int gdn_key_dim=ssm_d_state*ssm_n_group, gdn_val_dim=ssm_d_inner;        // 2048, 4096
    int gdn_conv_dim=gdn_key_dim*2+gdn_val_dim;                              // 8192
    ggml_context* rsctx=nullptr; ggml_backend_buffer_t rsbuf=nullptr;
    std::vector<ggml_tensor*> gdn_state(n_layer,nullptr), conv_state(n_layer,nullptr);
    if(is_qwen35){
        rsctx=ggml_init({(size_t)2*n_layer*ggml_tensor_overhead()+1024,nullptr,true});
        for(int il=0;il<n_layer;++il) if(is_recr[il]){
            gdn_state[il]=ggml_new_tensor_3d(rsctx,GGML_TYPE_F32,gdn_Sv,gdn_Sv,gdn_Hv);
            conv_state[il]=ggml_new_tensor_2d(rsctx,GGML_TYPE_F32,ssm_d_conv-1,gdn_conv_dim);
        }
        rsbuf=ggml_backend_alloc_ctx_tensors(rsctx,backend);
        fprintf(stderr,"[P2] GDN state %dx%dx%d + conv %dx%d per linear layer (%.0f MB total)\n",
            gdn_Sv,gdn_Sv,gdn_Hv,ssm_d_conv-1,gdn_conv_dim,
            (double)(30*((size_t)gdn_Sv*gdn_Sv*gdn_Hv+(size_t)(ssm_d_conv-1)*gdn_conv_dim)*4)/1e6);
    }
    auto reset_state=[&](){    // zero the recurrent state (call before each generation)
        if(!is_qwen35) return;
        std::vector<float> z((size_t)gdn_Sv*gdn_Sv*gdn_Hv,0.0f);
        for(int il=0;il<n_layer;++il) if(is_recr[il]){
            ggml_backend_tensor_set(gdn_state[il],z.data(),0,(size_t)gdn_Sv*gdn_Sv*gdn_Hv*4);
            ggml_backend_tensor_set(conv_state[il],z.data(),0,(size_t)(ssm_d_conv-1)*gdn_conv_dim*4);
        }
    };

    // slot pool
    Slots S; S.K=K;
    S.ctx=ggml_init({(size_t)(n_layer*K*3+16)*ggml_tensor_overhead(),nullptr,true});
    S.gate.resize(n_layer); S.up.resize(n_layer); S.down.resize(n_layer);
    S.slot_expert.assign(n_layer,std::vector<int>(K,-1));
    S.expert_slot.resize(n_layer); S.lru.resize(n_layer);
    for(int il=0;il<n_layer;++il) for(int s=0;s<K;++s){
        S.gate[il].push_back(ggml_new_tensor_2d(S.ctx,H.gtype[il],n_embd,n_ff));
        S.up[il].push_back(ggml_new_tensor_2d(S.ctx,H.utype[il],n_embd,n_ff));
        S.down[il].push_back(ggml_new_tensor_2d(S.ctx,H.dtype[il],n_ff,n_embd));
        S.lru[il].push_back(s);
    }
    S.buf=ggml_backend_alloc_ctx_tensors(S.ctx,backend);

    // ---- RAM tier (bounded LRU of expert byte-buffers); below it = NVMe (disk) ----
    RamTier RT; RT.cap=ram_cap; RT.slot2key.assign(ram_cap,-1);
    RT.gbuf.assign(ram_cap,nullptr); RT.ubuf.assign(ram_cap,nullptr); RT.dbuf.assign(ram_cap,nullptr);
    RT.ghead.assign(ram_cap,0); RT.uhead.assign(ram_cap,0); RT.dhead.assign(ram_cap,0);
    for(int s=0;s<ram_cap;++s) RT.lru.push_back(s);

    ggml_gallocr_t galloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));

    // metrics: three-tier access split + VRAM-slot copies
    long copies=0, reqs=0, vram_hit=0, ram_hit=0, nvme_fall=0;
    long long disk_ns=0; size_t disk_bytes=0;   // isolate NVMe: time + bytes in H.read (fseek+fread)
    std::set<int> touched;                 // distinct (il*n_expert+e) selected = WORKING SET
    bool trace=getenv("TRACE")!=nullptr;   // log per-(token,layer) selection -> prefetch-predictor ceiling
    std::vector<std::vector<std::vector<int>>> seltrace;   // [decode_tok][layer] = selected experts
    // --- benchmark instrumentation (all env-gated; a normal run sets none of these, so behaviour is unchanged) ---
    double throttle_Bps = getenv("THROTTLE_MBPS") ? atof(getenv("THROTTLE_MBPS"))*1e6 : 0.0; // simulate a slower drive
    const char* csvpath = getenv("CSV"); FILE* csvf=nullptr;                                  // per-token metrics CSV
    if(csvpath){ csvf=fopen(csvpath,"w"); if(csvf) fprintf(csvf,"tok,t_ms,dt_ms,vram_hit,ram_hit,nvme_fall,reqs,distinct,disk_mb\n"); }
    long long tok_first_ns=0, tok_prev_ns=0;   // per-token wall-clock -> hit-rate-over-time + latency percentiles
    // --- DB-reframe Stage-0/1 instrumentation (env-gated) ---
    FILE* gdf=nullptr; if(const char* gp=getenv("GATEDUMP")) gdf=fopen(gp,"w"); bool gatedump=(gdf!=nullptr); // per-(decode tok,layer): gate band + residency tier, for the OFFLINE residency-substitution ceiling
    if(gdf) fprintf(gdf,"# tok layer wk | id:prob:selected:restier(0=cold,1=vram,2=ram) ... (experts with prob>=wk-0.2)\n");
    FILE* pf=nullptr; if(const char* pp=getenv("PROFILE")) pf=fopen(pp,"w"); // lean access trace: per decode (tok,layer) the selected expert ids; '# turn' marks a new prompt/turn
    double nll_sum=0; long nll_n=0; std::vector<float> pr;   // teacher-forced NLL -> perplexity (quality probe); pr = host copy of full router softmax
    size_t vram = ggml_backend_buffer_get_size(M.wbuf)+ggml_backend_buffer_get_size(kv.buf)+ggml_backend_buffer_get_size(S.buf);
    if(getenv("DRYRUN")){ // VRAM-cliff probe: report static allocation (if the KV alloc above OOM'd we already crashed = past the cliff)
        size_t rs = rsbuf?ggml_backend_buffer_get_size(rsbuf):0; int nfull=0; for(int il=0;il<n_layer;++il) if(!is_qwen35||!is_recr[il]) nfull++;
        printf("DRYRUN K=%d max_kv=%d static_vram_mb=%.1f model_mb=%.1f kv_mb=%.1f slots_mb=%.1f gdn_mb=%.1f kv_layers=%d/%d\n",
            K,max_kv,(vram+rs)/1e6,ggml_backend_buffer_get_size(M.wbuf)/1e6,ggml_backend_buffer_get_size(kv.buf)/1e6,
            ggml_backend_buffer_get_size(S.buf)/1e6,rs/1e6,nfull,n_layer);
        return 0; }

    // fetch expert (il,e) bytes into the RAM tier; return RAM slot. RAM hit or NVMe fall-through.
    // copy expert (il,e) bytes (from RAM slot rs, at its head offset within the aligned buffer) into VRAM slot vs
    auto to_vram = [&](int il,int e,int rs,int vs){
        ggml_backend_tensor_set(S.gate[il][vs],RT.gbuf[rs]+RT.ghead[rs],0,H.gstride[il]);
        ggml_backend_tensor_set(S.up[il][vs],  RT.ubuf[rs]+RT.uhead[rs],0,H.ustride[il]);
        ggml_backend_tensor_set(S.down[il][vs],RT.dbuf[rs]+RT.dhead[rs],0,H.dstride[il]);
        copies++;
    };

    std::vector<float> x(n_embd);
    std::vector<int32_t> gen;
    struct timespec t0{},t1{},tg0{},tfirst{}; bool timing=false;
    size_t peak_compute=0;
    auto NOW=[](){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (long long)t.tv_sec*1000000000LL+t.tv_nsec; };
    long long tE=0,tA=0,tFetch=0,tDisk=0,tB=0,tHead=0;   // decode-only phase timers (ns)

    // ============================ BATCHED PREFILL (the 20-30x lever) ============================
    // Process the prompt in chunks of C tokens with batched graphs (per-batch instead of per-position),
    // streaming each routed expert ONCE per (layer,chunk) instead of once per position. Builds the SAME
    // KV/GDN/conv state as the per-position path (carried across chunks; state tensors persist), so it is
    // token-exact. Returns logits at the final prompt position into `out_logits`. ENV: BATCH_PREFILL=1, PREFILL_CHUNK=C.
    // Validated primitives: ggml_gated_delta_net / ggml_ssm_conv batched == sequential (bit-exact, batchgdn_test);
    // mul_mat_id over a slot pool (run_specloop). The only batched-only piece is the causal mask on full-attn.
    // resume_from>0: INCREMENTAL prefill -- the KV/GDN/conv state for [0,resume_from) is already valid (prior turn);
    // skip reset and prefill only [resume_from, Tp). Used for multi-turn chat to avoid re-prefilling the whole conversation.
    auto prefill_batched=[&](std::vector<int32_t>& seq,int Tp,int C,std::vector<float>& out_logits,int resume_from){
        if(resume_from<=0) reset_state();
        int Hv=gdn_Hv,Hk=gdn_Hk,Sv=gdn_Sv,kd=gdn_key_dim,vd=gdn_val_dim,cd=gdn_conv_dim;
        bool mrope=(is_qwen35 && !is_qwen3next);
        long long pe=0,pa=0,pn=0,pv=0,pbt=0,ph=0; bool ptime=getenv("PTIME")!=nullptr;   // prefill phase timers (ns): embed/graphA/nvme/poolstage/graphB/head
        std::vector<char> scratch((size_t)32*1024*1024);   // ONE persistent host metadata arena, reused for every per-layer ggml_init (no malloc/free churn)
        // ON-GPU RESIDUAL: keep the residual stream (Xg), router input (fxg) and post-attn residual (fig) in PERSISTENT VRAM
        // across the graphA->host-route->graphB boundary, so only the 4MB router probs leave the GPU per layer (vs ~230MB).
        int Cm=std::min(C,Tp-(resume_from>0?resume_from:0)); if(Cm<1)Cm=1;
        ggml_context* rctx=ggml_init({(size_t)4*ggml_tensor_overhead()+256,nullptr,true});
        ggml_tensor* Xg =ggml_new_tensor_2d(rctx,GGML_TYPE_F32,n_embd,Cm);   // residual stream
        ggml_tensor* fxg=ggml_new_tensor_2d(rctx,GGML_TYPE_F32,n_embd,Cm);   // post-attn-norm router/expert input
        ggml_tensor* fig=ggml_new_tensor_2d(rctx,GGML_TYPE_F32,n_embd,Cm);   // post-attn residual (added back after MoE)
        ggml_backend_buffer_t rbuf=ggml_backend_alloc_ctx_tensors(rctx,backend);
        if(!rbuf){ fprintf(stderr,"prefill_batched: residual VRAM alloc failed (Cm=%d)\n",Cm); ggml_free(rctx); out_logits.assign(n_vocab,0.f); return; }
        for(int base=(resume_from>0?resume_from:0); base<Tp; base+=C){
            int m=std::min(C,Tp-base); int nkv=base+m;
            // HOIST: causal mask [nkv,m] + positions are identical for every full-attn layer in this chunk -> build once
            std::vector<int32_t> pp(mrope?4*m:m); for(int t=0;t<m;++t){ if(mrope){ for(int s=0;s<4;++s) pp[(size_t)s*m+t]=base+t; } else pp[t]=base+t; }
            std::vector<float> mk((size_t)nkv*m); for(int t=0;t<m;++t)for(int j=0;j<nkv;++j) mk[(size_t)t*nkv+j]=(j<=base+t)?0.0f:-INFINITY;
            // ---- graph E: embed m tokens -> Xg[:, 0:m] (stays on GPU) ----
            { long long _te=NOW(); ggml_context* g=ggml_init({(size_t)32*1024*1024,scratch.data(),true}); ggml_cgraph* gf=ggml_new_graph(g);
              ggml_tensor* it=ggml_new_tensor_1d(g,GGML_TYPE_I32,m); ggml_set_input(it);
              ggml_tensor* e=ggml_get_rows(g,M.get("token_embd.weight"),it);
              ggml_tensor* cpy=ggml_cpy(g,e,ggml_view_2d(g,Xg,n_embd,m,Xg->nb[1],0));
              ggml_build_forward_expand(gf,cpy); ggml_gallocr_alloc_graph(galloc,gf);
              ggml_backend_tensor_set(it,seq.data()+base,0,(size_t)m*4); ggml_backend_graph_compute(backend,gf);
              ggml_free(g); pe+=NOW()-_te; }
            for(int il=0; il<n_layer; ++il){
                std::vector<float> probs((size_t)n_expert*m);   // only the router softmax comes to host (for expert selection)
                // ---- graph A: attention (full OR GDN) + post-attn norm + router, batched over m ----
                long long _ta=NOW();
                { ggml_context* g=ggml_init({(size_t)16*1024*1024,scratch.data(),true});
                  ggml_cgraph* gf=ggml_new_graph_custom(g,8192,false);
                  ggml_tensor* ix=ggml_view_2d(g,Xg,n_embd,m,Xg->nb[1],0);   // residual already on GPU (no H2D)
                  ggml_tensor* ip=ggml_new_tensor_1d(g,GGML_TYPE_I32,mrope?4*m:m); ggml_set_input(ip);
                  ggml_tensor* mask=ggml_new_tensor_2d(g,GGML_TYPE_F32,nkv,m); ggml_set_input(mask);   // causal [nkv,m]
                  ggml_tensor* xn=ggml_mul(g,ggml_rms_norm(g,ix,eps),M.blk(il,"attn_norm.weight"));
                  ggml_tensor* att; ggml_tensor* extra1=nullptr; ggml_tensor* extra2=nullptr;
                  if(!is_recr[il]){
                      ggml_tensor* Qg=ggml_mul_mat(g,M.blk(il,"attn_q.weight"),xn);          // [head_dim*n_head*2, m]
                      size_t es=ggml_element_size(Qg);
                      ggml_tensor* Qc=ggml_view_3d(g,Qg,head_dim,n_head,m,es*head_dim*2,es*head_dim*2*n_head,0);
                      Qc=ggml_mul(g,ggml_rms_norm(g,Qc,eps),M.blk(il,"attn_q_norm.weight"));
                      ggml_tensor* gt=ggml_view_3d(g,Qg,head_dim,n_head,m,es*head_dim*2,es*head_dim*2*n_head,es*head_dim);
                      gt=ggml_cont_2d(g,gt,n_embd_attn,m);
                      ggml_tensor* Kc=ggml_reshape_3d(g,ggml_mul_mat(g,M.blk(il,"attn_k.weight"),xn),head_dim,n_head_kv,m);
                      Kc=ggml_mul(g,ggml_rms_norm(g,Kc,eps),M.blk(il,"attn_k_norm.weight"));
                      ggml_tensor* Vc=ggml_reshape_3d(g,ggml_mul_mat(g,M.blk(il,"attn_v.weight"),xn),head_dim,n_head_kv,m);
                      if(is_qwen3next){ Qc=ggml_rope_ext(g,Qc,ip,nullptr,64,GGML_ROPE_TYPE_NEOX,n_ctx_orig,freq_base,1,0,1,32,1);
                                        Kc=ggml_rope_ext(g,Kc,ip,nullptr,64,GGML_ROPE_TYPE_NEOX,n_ctx_orig,freq_base,1,0,1,32,1); }
                      else { int sec[4]={rope_sections[0],rope_sections[1],rope_sections[2],rope_sections[3]};
                             Qc=ggml_rope_multi(g,Qc,ip,nullptr,64,sec,GGML_ROPE_TYPE_IMROPE,n_ctx_orig,freq_base,1,0,1,32,1);
                             Kc=ggml_rope_multi(g,Kc,ip,nullptr,64,sec,GGML_ROPE_TYPE_IMROPE,n_ctx_orig,freq_base,1,0,1,32,1); }
                      ggml_tensor* kc=kv.k[il]; ggml_tensor* vc=kv.v[il];
                      ggml_tensor* dk=ggml_view_3d(g,kc,head_dim,n_head_kv,m,kc->nb[1],kc->nb[2],(size_t)base*kc->nb[2]);
                      ggml_tensor* dv=ggml_view_3d(g,vc,head_dim,n_head_kv,m,vc->nb[1],vc->nb[2],(size_t)base*vc->nb[2]);
                      extra1=ggml_cpy(g,Kc,dk); extra2=ggml_cpy(g,Vc,dv);
                      ggml_tensor *Kall,*Vall;
                      if(base==0){Kall=Kc;Vall=Vc;} else {
                          Kall=ggml_concat(g,ggml_view_3d(g,kc,head_dim,n_head_kv,base,kc->nb[1],kc->nb[2],0),Kc,2);
                          Vall=ggml_concat(g,ggml_view_3d(g,vc,head_dim,n_head_kv,base,vc->nb[1],vc->nb[2],0),Vc,2); }
                      ggml_tensor* q=ggml_permute(g,Qc,0,2,1,3);
                      ggml_tensor* k=ggml_permute(g,Kall,0,2,1,3);
                      ggml_tensor* kq=ggml_soft_max_ext(g,ggml_mul_mat(g,k,q),mask,1.0f/sqrtf((float)head_dim),0.0f);
                      ggml_tensor* vv=ggml_cont(g,ggml_permute(g,Vall,1,2,0,3));
                      ggml_tensor* kqv=ggml_permute(g,ggml_mul_mat(g,vv,kq),0,2,1,3);
                      ggml_tensor* ao=ggml_cont_2d(g,kqv,n_embd_attn,m);
                      ao=ggml_mul(g,ao,ggml_sigmoid(g,gt));
                      att=ggml_mul_mat(g,M.blk(il,"attn_output.weight"),ao);
                  } else {
                      ggml_tensor* qkv=ggml_mul_mat(g,M.blk(il,"attn_qkv.weight"),xn);        // [cd, m]
                      ggml_tensor* z=ggml_mul_mat(g,M.blk(il,"attn_gate.weight"),xn);         // [vd, m]
                      ggml_tensor* beta; ggml_tensor* al;
                      if(is_qwen3next){ int vpg=Hv/Hk;
                          ggml_tensor* ba=ggml_reshape_3d(g,ggml_mul_mat(g,M.blk(il,"ssm_ba.weight"),xn),vpg*2,Hk,m); // [vpg*2,Hk,m]
                          ggml_tensor* br=ggml_cont(g,ggml_view_3d(g,ba,vpg,Hk,m,ba->nb[1],ba->nb[2],0));
                          ggml_tensor* ar=ggml_cont(g,ggml_view_3d(g,ba,vpg,Hk,m,ba->nb[1],ba->nb[2],(size_t)vpg*ggml_element_size(ba)));
                          beta=ggml_sigmoid(g,ggml_reshape_2d(g,br,Hv,m));
                          al  =ggml_reshape_2d(g,ar,Hv,m);
                      } else { beta=ggml_sigmoid(g,ggml_mul_mat(g,M.blk(il,"ssm_beta.weight"),xn)); // [Hv,m]
                               al  =ggml_mul_mat(g,M.blk(il,"ssm_alpha.weight"),xn); }                // [Hv,m]
                      beta=ggml_reshape_4d(g,beta,1,Hv,m,1);
                      al=ggml_softplus(g,ggml_add(g,al,M.blk(il,"ssm_dt.bias")));
                      ggml_tensor* gg=ggml_reshape_4d(g,ggml_mul(g,al,M.blk(il,"ssm_a")),1,Hv,m,1);
                      ggml_tensor* cs=conv_state[il];                                          // [d_conv-1, cd]
                      ggml_tensor* qkvT=ggml_cont(g,ggml_transpose(g,qkv));                    // [m, cd]
                      ggml_tensor* cin=ggml_concat(g,cs,qkvT,0);                               // [d_conv-1+m, cd]
                      ggml_tensor* convo=ggml_silu(g,ggml_ssm_conv(g,cin,M.blk(il,"ssm_conv1d.weight"))); // [cd, m]
                      extra2=ggml_cpy(g,ggml_view_2d(g,cin,ssm_d_conv-1,cd,cin->nb[1],(size_t)m*cin->nb[0]),cs); // roll conv state
                      size_t cz=ggml_element_size(convo);
                      ggml_tensor* qc=ggml_l2_norm(g,ggml_cont(g,ggml_view_3d(g,convo,Sv,Hk,m,cz*Sv,convo->nb[1],0)),eps);
                      ggml_tensor* kc2=ggml_l2_norm(g,ggml_cont(g,ggml_view_3d(g,convo,Sv,Hk,m,cz*Sv,convo->nb[1],cz*kd)),eps);
                      ggml_tensor* vc2=ggml_cont(g,ggml_view_3d(g,convo,Sv,Hv,m,cz*Sv,convo->nb[1],cz*kd*2));
                      qc=ggml_reshape_4d(g,qc,Sv,Hk,m,1); kc2=ggml_reshape_4d(g,kc2,Sv,Hk,m,1); vc2=ggml_reshape_4d(g,vc2,Sv,Hv,m,1);
                      if(is_qwen3next && Hv!=Hk){ int vpg=Hv/Hk;
                          qc =ggml_reshape_4d(g,ggml_repeat_4d(g,ggml_reshape_4d(g,qc ,Sv,1,Hk,m),Sv,vpg,Hk,m),Sv,Hv,m,1);
                          kc2=ggml_reshape_4d(g,ggml_repeat_4d(g,ggml_reshape_4d(g,kc2,Sv,1,Hk,m),Sv,vpg,Hk,m),Sv,Hv,m,1); }
                      ggml_tensor* s0=ggml_reshape_4d(g,gdn_state[il],Sv,Sv,Hv,1);
                      ggml_tensor* gdn=ggml_gated_delta_net(g,qc,kc2,vc2,gg,beta,s0,1);        // n_tokens=m, K=1
                      size_t rs1=ggml_row_size(gdn->type,Sv);
                      ggml_tensor* out=ggml_view_4d(g,gdn,Sv,Hv,m,1,rs1,ggml_row_size(gdn->type,Sv*Hv),ggml_row_size(gdn->type,Sv*Hv*m),0);
                      ggml_tensor* ns=ggml_view_4d(g,gdn,Sv,Sv,Hv,1,rs1,ggml_row_size(gdn->type,Sv*Sv),ggml_row_size(gdn->type,Sv*Sv*Hv),ggml_row_size(gdn->type,Sv*Hv*m));
                      extra1=ggml_cpy(g,ns,gdn_state[il]);
                      ggml_tensor* zr=ggml_reshape_4d(g,z,Sv,Hv,m,1);
                      ggml_tensor* on=ggml_mul(g,ggml_mul(g,ggml_rms_norm(g,out,eps),M.blk(il,"ssm_norm.weight")),ggml_silu(g,zr));
                      att=ggml_mul_mat(g,M.blk(il,"ssm_out.weight"),ggml_reshape_2d(g,on,vd,m));
                  }
                  ggml_tensor* ffn_inp=ggml_add(g,att,ix);
                  ggml_tensor* fx=ggml_mul(g,ggml_rms_norm(g,ffn_inp,eps),M.blk(il,"post_attention_norm.weight"));
                  ggml_tensor* probsT=ggml_soft_max(g,ggml_mul_mat(g,M.blk(il,"ffn_gate_inp.weight"),fx)); // [n_expert,m]
                  ggml_tensor* cfx=ggml_cpy(g,fx,ggml_view_2d(g,fxg,n_embd,m,fxg->nb[1],0));        // router/expert input -> persistent GPU
                  ggml_tensor* cfi=ggml_cpy(g,ffn_inp,ggml_view_2d(g,fig,n_embd,m,fig->nb[1],0));   // post-attn residual -> persistent GPU
                  ggml_set_output(probsT);
                  ggml_build_forward_expand(gf,cfx); ggml_build_forward_expand(gf,cfi); ggml_build_forward_expand(gf,probsT);
                  if(extra1) ggml_build_forward_expand(gf,extra1);
                  if(extra2) ggml_build_forward_expand(gf,extra2);
                  ggml_gallocr_alloc_graph(galloc,gf);
                  if(!is_recr[il]){ ggml_backend_tensor_set(ip,pp.data(),0,pp.size()*4);   // only positions + causal mask H2D (ix is Xg, already on GPU)
                      ggml_backend_tensor_set(mask,mk.data(),0,(size_t)nkv*m*4); }
                  ggml_backend_graph_compute(backend,gf);
                  ggml_backend_tensor_get(probsT,probs.data(),0,(size_t)n_expert*m*4);   // ONLY 4MB router probs leaves the GPU
                  ggml_free(g);
                }
                pa+=NOW()-_ta;
                // ---- host: per-token top-n_used + wnorm; expert UNION across the chunk ----
                std::vector<int32_t> sel((size_t)n_used*m); std::vector<float> wgt((size_t)n_used*m,0.f);
                std::set<int> uni;
                for(int t=0;t<m;++t){
                    const float* pr=probs.data()+(size_t)t*n_expert;
                    std::vector<std::pair<float,int>> cand(n_expert); for(int e=0;e<n_expert;++e)cand[e]={pr[e],e};
                    std::partial_sort(cand.begin(),cand.begin()+n_used,cand.end(),[](auto&a,auto&b){return a.first>b.first;});
                    float ws=0; for(int j=0;j<n_used;++j) ws+=cand[j].first;
                    if(wnorm){ if(ws<6.103515625e-5f) ws=6.103515625e-5f; } else ws=1.0f;
                    for(int j=0;j<n_used;++j){ int e=cand[j].second; sel[(size_t)t*n_used+j]=e; wgt[(size_t)t*n_used+j]=cand[j].first/ws; uni.insert(e); touched.insert(il*n_expert+e); }
                }
                // ---- stream the union ONCE into RAM tier (NVMe on miss), assign pool slots 0..U-1 ----
                std::vector<int> ulist(uni.begin(),uni.end()); int U=(int)ulist.size();
                std::map<int,int> e2slot; for(int u=0;u<U;++u) e2slot[ulist[u]]=u;
                std::vector<int> ramslot(U,-1); std::vector<IOJob> jobs; jobs.reserve(U*3);
                for(int u=0;u<U;++u){ int e=ulist[u]; reqs++; int key=il*n_expert+e; auto rit=RT.key2slot.find(key);
                    if(rit!=RT.key2slot.end()){ ram_hit++; RT.lru.remove(rit->second); RT.lru.push_front(rit->second); ramslot[u]=rit->second; }
                    else { int rs=RT.lru.back(); RT.lru.pop_back(); int old=RT.slot2key[rs]; if(old>=0) RT.key2slot.erase(old);
                        if(!RT.gbuf[rs]){ RT.gbuf[rs]=(uint8_t*)VirtualAlloc(NULL,gbsz,MEM_COMMIT|MEM_RESERVE,PAGE_READWRITE);
                            RT.ubuf[rs]=(uint8_t*)VirtualAlloc(NULL,ubsz,MEM_COMMIT|MEM_RESERVE,PAGE_READWRITE);
                            RT.dbuf[rs]=(uint8_t*)VirtualAlloc(NULL,dbsz,MEM_COMMIT|MEM_RESERVE,PAGE_READWRITE); }
                        RT.slot2key[rs]=key; RT.key2slot[key]=rs; RT.lru.push_front(rs); ramslot[u]=rs; nvme_fall++;
                        uint64_t go=H.goff[il]+(size_t)e*H.gstride[il], uo=H.uoff[il]+(size_t)e*H.ustride[il], od=H.doff[il]+(size_t)e*H.dstride[il];
                        uint32_t gs=(uint32_t)H.gstride[il], us=(uint32_t)H.ustride[il], ds=(uint32_t)H.dstride[il];
                        RT.ghead[rs]=(uint32_t)(go&4095); RT.uhead[rs]=(uint32_t)(uo&4095); RT.dhead[rs]=(uint32_t)(od&4095);
                        jobs.push_back(IOJob{RT.gbuf[rs], go&~(uint64_t)4095, (RT.ghead[rs]+gs+4095u)&~4095u, RT.ghead[rs], gs});
                        jobs.push_back(IOJob{RT.ubuf[rs], uo&~(uint64_t)4095, (RT.uhead[rs]+us+4095u)&~4095u, RT.uhead[rs], us});
                        jobs.push_back(IOJob{RT.dbuf[rs], od&~(uint64_t)4095, (RT.dhead[rs]+ds+4095u)&~4095u, RT.dhead[rs], ds}); }
                }
                { struct timespec d0,d1; clock_gettime(CLOCK_MONOTONIC,&d0); run_jobs(io,jobs); clock_gettime(CLOCK_MONOTONIC,&d1);
                  long long _dn=(d1.tv_sec-d0.tv_sec)*1000000000LL+(d1.tv_nsec-d0.tv_nsec); disk_ns+=_dn; pn+=_dn; for(auto&jb:jobs) disk_bytes+=jb.rsize; }
                // ---- graph B: expert FFN via mul_mat_id over a graph-local typed pool [.,.,U] + shared expert ----
                long long _tbt=NOW();
                { ggml_context* g=ggml_init({(size_t)16*1024*1024,scratch.data(),true});
                  ggml_cgraph* gf=ggml_new_graph_custom(g,4096,false);
                  ggml_tensor* GP=ggml_new_tensor_3d(g,H.gtype[il],n_embd,n_ff,U); ggml_set_input(GP);
                  ggml_tensor* UP=ggml_new_tensor_3d(g,H.utype[il],n_embd,n_ff,U); ggml_set_input(UP);
                  ggml_tensor* DP=ggml_new_tensor_3d(g,H.dtype[il],n_ff,n_embd,U); ggml_set_input(DP);
                  ggml_tensor* ifx=ggml_view_2d(g,fxg,n_embd,m,fxg->nb[1],0);   // expert input already on GPU (no H2D)
                  ggml_tensor* idT=ggml_new_tensor_2d(g,GGML_TYPE_I32,n_used,m); ggml_set_input(idT);
                  ggml_tensor* wT=ggml_new_tensor_3d(g,GGML_TYPE_F32,1,n_used,m); ggml_set_input(wT);
                  ggml_tensor* cx=ggml_reshape_3d(g,ifx,n_embd,1,m);
                  ggml_tensor* up=ggml_mul_mat_id(g,UP,cx,idT);
                  ggml_tensor* ga=ggml_silu(g,ggml_mul_mat_id(g,GP,cx,idT));
                  ggml_tensor* ex=ggml_mul_mat_id(g,DP,ggml_mul(g,up,ga),idT);          // [n_embd, n_used, m]
                  ex=ggml_mul(g,ex,wT);
                  ggml_tensor* mo=nullptr; for(int j=0;j<n_used;++j){ ggml_tensor* s=ggml_view_2d(g,ex,n_embd,m,ex->nb[2],(size_t)j*ex->nb[1]); mo=j==0?s:ggml_add(g,mo,s); }
                  ggml_tensor* she=nullptr;
                  if(is_qwen35){ ggml_tensor* sg=ggml_mul_mat(g,M.blk(il,"ffn_gate_shexp.weight"),ifx);
                      ggml_tensor* su=ggml_mul_mat(g,M.blk(il,"ffn_up_shexp.weight"),ifx);
                      ggml_tensor* sh=ggml_mul_mat(g,M.blk(il,"ffn_down_shexp.weight"),ggml_mul(g,ggml_silu(g,sg),su));
                      ggml_tensor* sgate=ggml_sigmoid(g,ggml_mul_mat(g,M.blk(il,"ffn_gate_inp_shexp.weight"),ifx));
                      she=ggml_mul(g,sh,sgate); }
                  ggml_tensor* fin=ggml_view_2d(g,fig,n_embd,m,fig->nb[1],0);                               // post-attn residual (on GPU)
                  ggml_tensor* xnew=is_qwen35? ggml_add(g,ggml_add(g,mo,she),fin) : ggml_add(g,mo,fin);     // moe + shared expert + residual
                  ggml_tensor* cx2=ggml_cpy(g,xnew,ggml_view_2d(g,Xg,n_embd,m,Xg->nb[1],0));                // new residual -> Xg (stays on GPU)
                  ggml_set_output(cx2); ggml_build_forward_expand(gf,cx2);
                  ggml_gallocr_alloc_graph(galloc,gf);
                  long long _tv=NOW();
                  for(int u=0;u<U;++u){ int rs=ramslot[u];
                      ggml_backend_tensor_set(GP,RT.gbuf[rs]+RT.ghead[rs],(size_t)u*H.gstride[il],H.gstride[il]);
                      ggml_backend_tensor_set(UP,RT.ubuf[rs]+RT.uhead[rs],(size_t)u*H.ustride[il],H.ustride[il]);
                      ggml_backend_tensor_set(DP,RT.dbuf[rs]+RT.dhead[rs],(size_t)u*H.dstride[il],H.dstride[il]); copies++; }
                  pv+=NOW()-_tv;
                  std::vector<int32_t> ids((size_t)n_used*m); for(size_t i=0;i<(size_t)n_used*m;++i) ids[i]=e2slot[sel[i]];
                  ggml_backend_tensor_set(idT,ids.data(),0,(size_t)n_used*m*4);   // ifx is fxg (on GPU) -> no expert-input H2D
                  ggml_backend_tensor_set(wT,wgt.data(),0,(size_t)n_used*m*4);
                  ggml_backend_graph_compute(backend,gf);
                  ggml_free(g); }                                                  // residual add (moe+shared+ffn_inp) done on GPU into Xg -- no D2H
                pbt+=NOW()-_tbt;
            }
            if(base+m>=Tp){   // final chunk: head on the LAST position
                long long _th=NOW(); out_logits.assign(n_vocab,0.f);
                ggml_context* g=ggml_init({(size_t)16*1024*1024,scratch.data(),true}); ggml_cgraph* gf=ggml_new_graph(g);
                ggml_tensor* ix=ggml_view_2d(g,Xg,n_embd,1,Xg->nb[1],(size_t)(m-1)*Xg->nb[1]);   // last residual column (on GPU)
                ggml_tensor* lg=ggml_mul_mat(g,M.get("output.weight"),ggml_mul(g,ggml_rms_norm(g,ix,eps),M.get("output_norm.weight"))); ggml_set_output(lg);
                ggml_build_forward_expand(gf,lg); ggml_gallocr_alloc_graph(galloc,gf);
                ggml_backend_graph_compute(backend,gf); ggml_backend_tensor_get(lg,out_logits.data(),0,(size_t)n_vocab*4); ggml_free(g); ph+=NOW()-_th;
            }
            if(ptime) fprintf(stderr,"[prefill] chunk %d..%d/%d done\n",base,base+m,Tp);
        }
        if(ptime) fprintf(stderr,"[PTIME] embed=%.1fs graphA=%.1fs nvme=%.1fs poolstage=%.1fs graphB=%.1fs head=%.1fs\n",
            pe/1e9,pa/1e9,pn/1e9,pv/1e9,pbt/1e9,ph/1e9);
        ggml_backend_buffer_free(rbuf); ggml_free(rctx);   // free the persistent residual tensors
    };

    // core streaming forward over `seq`, emitting each new token id to on_token.
    // params deliberately shadow seq/T/ngen so the loop body is unchanged; one-shot AND chat both call this.
    auto generate=[&](std::vector<int32_t>& seq,int T,int ngen,const std::function<void(int)>& on_token,int Cbatch,int resumeFrom){
    int p0=0;
    if(Cbatch>0 && !tf && T>0){
        // BATCHED PREFILL path: build KV/GDN/conv state for positions resumeFrom..T-1 (prefill_batched resets state
        // only when resumeFrom<=0), sample the first token from the final-position logits, then DECODE per-position
        // from p=T on the batched-built state (no reset). Output is identical to the per-position path (validated).
        clock_gettime(CLOCK_MONOTONIC,&tg0);
        std::vector<float> lg; prefill_batched(seq,T,Cbatch,lg,resumeFrom);
        int bi=0; { float bv=lg[0]; for(int v=1;v<n_vocab;++v) if(lg[v]>bv){bv=lg[v];bi=v;} }   // argmax
        if(g_temp>0.0f){ double mx=(double)lg[bi],cut=mx-12.0,sum=0;
            for(int v=0;v<n_vocab;++v) if((double)lg[v]>=cut) sum+=exp(((double)lg[v]-mx)/g_temp);
            double r=std::uniform_real_distribution<double>(0.0,sum)(g_rng),acc=0;
            for(int v=0;v<n_vocab;++v) if((double)lg[v]>=cut){ acc+=exp(((double)lg[v]-mx)/g_temp); if(acc>=r){bi=v;break;} } }
        clock_gettime(CLOCK_MONOTONIC,&tfirst);   // first token emitted => prefill (TTFT) done
        gen.push_back(bi); on_token(bi);
        static const bool noeos0=getenv("NOEOS")!=nullptr;
        if(g_stop||(int)gen.size()>=ngen||(bi==eos_id&&!noeos0)){ clock_gettime(CLOCK_MONOTONIC,&t1); return; }
        seq.push_back(bi); p0=T;                   // decode continues from position T (state already built)
    } else {
        reset_state();   // zero GDN recurrent + conv state at the start of each generation
        clock_gettime(CLOCK_MONOTONIC,&tg0);   // prefill TTFT clock: start before position 0
    }
    for(int p=p0;;++p){
        if(p>=(int)seq.size()) break;
        if(p>=max_kv-1){ on_token(eos_id); break; }   // KV cache full: stop cleanly
        int32_t tok=seq[p];
        if (p==T){ clock_gettime(CLOCK_MONOTONIC,&t0); timing=true; } // start timing at first decode step
        long long _te=NOW();
        // embedding (graph E)
        {
            ggml_context* g=ggml_init({(size_t)8*1024*1024,nullptr,true});
            ggml_cgraph* gf=ggml_new_graph(g);
            ggml_tensor* it=ggml_new_tensor_1d(g,GGML_TYPE_I32,1); ggml_set_input(it);
            ggml_tensor* e=ggml_get_rows(g,M.get("token_embd.weight"),it); ggml_set_output(e);
            ggml_build_forward_expand(gf,e); ggml_gallocr_alloc_graph(galloc,gf);
            ggml_backend_tensor_set(it,&tok,0,4); ggml_backend_graph_compute(backend,gf);
            ggml_backend_tensor_get(e,x.data(),0,(size_t)n_embd*4); ggml_free(g);
        }
        if(p>=T) tE+=NOW()-_te;
        if(mode!="chat"&&p==T-1){double n=0;for(float v:x)n+=v*v;fprintf(stderr,"[dbg] embd x_norm=%.3f x012=%.4f %.4f %.4f\n",sqrt(n),x[0],x[1],x[2]);}
        for(int il=0; il<n_layer; ++il){
            std::vector<float> ffnx(n_embd), ffninp(n_embd), wsel(n_used);
            std::vector<int32_t> sel(n_used);
            long long _ta=NOW();
            // ---- graph A: hybrid attention (full OR Gated Delta Net) + post-attn norm + router ----
            {
                ggml_context* g=ggml_init({(size_t)256*1024*1024,nullptr,true});
                ggml_cgraph* gf=ggml_new_graph_custom(g,8192,false);
                ggml_tensor* ix=ggml_new_tensor_2d(g,GGML_TYPE_F32,n_embd,1); ggml_set_input(ix);
                bool mrope=(is_qwen35 && !is_qwen3next);                         // qwen35moe uses IMRoPE (4 pos/token); qwen3next & others 1
                ggml_tensor* ip=ggml_new_tensor_1d(g,GGML_TYPE_I32,mrope?4:1); ggml_set_input(ip);
                ggml_tensor* xn=ggml_mul(g,ggml_rms_norm(g,ix,eps),M.blk(il,"attn_norm.weight"));
                ggml_tensor* att; ggml_tensor* extra1=nullptr; ggml_tensor* extra2=nullptr; // state writebacks
                if(!is_recr[il]){
                    // ===== P1: FULL ATTENTION — fused Q+gate, per-head q/k norm, IMRoPE, GQA, sigmoid gate =====
                    ggml_tensor* Qg=ggml_mul_mat(g,M.blk(il,"attn_q.weight"),xn);   // [head_dim*n_head*2,1]
                    size_t es=ggml_element_size(Qg);
                    ggml_tensor* Qc=ggml_view_3d(g,Qg,head_dim,n_head,1,es*head_dim*2,es*head_dim*2*n_head,0);
                    Qc=ggml_mul(g,ggml_rms_norm(g,Qc,eps),M.blk(il,"attn_q_norm.weight"));
                    ggml_tensor* gt=ggml_view_3d(g,Qg,head_dim,n_head,1,es*head_dim*2,es*head_dim*2*n_head,es*head_dim);
                    gt=ggml_cont_2d(g,gt,n_embd_attn,1);
                    ggml_tensor* Kc=ggml_reshape_3d(g,ggml_mul_mat(g,M.blk(il,"attn_k.weight"),xn),head_dim,n_head_kv,1);
                    Kc=ggml_mul(g,ggml_rms_norm(g,Kc,eps),M.blk(il,"attn_k_norm.weight"));
                    ggml_tensor* Vc=ggml_reshape_3d(g,ggml_mul_mat(g,M.blk(il,"attn_v.weight"),xn),head_dim,n_head_kv,1);
                    if(is_qwen3next){ // NEOX rope, rope.dimension_count=64 (partial: first 64 of 256-dim head), 1 pos/token
                        Qc=ggml_rope_ext(g,Qc,ip,nullptr,64,GGML_ROPE_TYPE_NEOX,n_ctx_orig,freq_base,1,0,1,32,1);
                        Kc=ggml_rope_ext(g,Kc,ip,nullptr,64,GGML_ROPE_TYPE_NEOX,n_ctx_orig,freq_base,1,0,1,32,1);
                    } else {          // qwen35moe: multi-section IMRoPE (4 pos/token)
                        int sec[4]={rope_sections[0],rope_sections[1],rope_sections[2],rope_sections[3]};
                        Qc=ggml_rope_multi(g,Qc,ip,nullptr,64,sec,GGML_ROPE_TYPE_IMROPE,n_ctx_orig,freq_base,1,0,1,32,1);
                        Kc=ggml_rope_multi(g,Kc,ip,nullptr,64,sec,GGML_ROPE_TYPE_IMROPE,n_ctx_orig,freq_base,1,0,1,32,1);
                    }
                    ggml_tensor* kc=kv.k[il]; ggml_tensor* vc=kv.v[il];
                    ggml_tensor* dk=ggml_view_3d(g,kc,head_dim,n_head_kv,1,kc->nb[1],kc->nb[2],(size_t)p*kc->nb[2]);
                    ggml_tensor* dv=ggml_view_3d(g,vc,head_dim,n_head_kv,1,vc->nb[1],vc->nb[2],(size_t)p*vc->nb[2]);
                    extra1=ggml_cpy(g,Kc,dk); extra2=ggml_cpy(g,Vc,dv);
                    ggml_tensor *Kall,*Vall;
                    if(p==0){Kall=Kc;Vall=Vc;} else {
                        Kall=ggml_concat(g,ggml_view_3d(g,kc,head_dim,n_head_kv,p,kc->nb[1],kc->nb[2],0),Kc,2);
                        Vall=ggml_concat(g,ggml_view_3d(g,vc,head_dim,n_head_kv,p,vc->nb[1],vc->nb[2],0),Vc,2);
                    }
                    ggml_tensor* q=ggml_permute(g,Qc,0,2,1,3);
                    ggml_tensor* k=ggml_permute(g,Kall,0,2,1,3);
                    ggml_tensor* kq=ggml_soft_max_ext(g,ggml_mul_mat(g,k,q),nullptr,1.0f/sqrtf((float)head_dim),0.0f);
                    ggml_tensor* vv=ggml_cont(g,ggml_permute(g,Vall,1,2,0,3));
                    ggml_tensor* kqv=ggml_permute(g,ggml_mul_mat(g,vv,kq),0,2,1,3);
                    ggml_tensor* ao=ggml_cont_2d(g,kqv,n_embd_attn,1);
                    ao=ggml_mul(g,ao,ggml_sigmoid(g,gt));                              // sigmoid gate on attn output
                    att=ggml_mul_mat(g,M.blk(il,"attn_output.weight"),ao);
                } else {
                    // ===== P3: GATED DELTA NET (linear attention) =====
                    int Sv=gdn_Sv,Hv=gdn_Hv,Hk=gdn_Hk,kd=gdn_key_dim,vd=gdn_val_dim,cd=gdn_conv_dim;
                    ggml_tensor* qkv=ggml_mul_mat(g,M.blk(il,"attn_qkv.weight"),xn);    // [cd=8192,1]
                    ggml_tensor* z=ggml_mul_mat(g,M.blk(il,"attn_gate.weight"),xn);     // [vd=4096,1]
                    ggml_tensor* beta; ggml_tensor* al;
                    if(is_qwen3next){ // fused ssm_ba [2*Hv]: layout is per k-head [vpg betas][vpg alphas] x Hk -> de-interleave
                        int vpg=Hv/Hk;                                                              // value heads per k-head group (2)
                        ggml_tensor* ba=ggml_reshape_2d(g,ggml_mul_mat(g,M.blk(il,"ssm_ba.weight"),xn),vpg*2,Hk); // [4,Hk]
                        ggml_tensor* br=ggml_cont(g,ggml_view_2d(g,ba,vpg,Hk,ba->nb[1],0));                       // betas  -> [vpg,Hk]
                        ggml_tensor* ar=ggml_cont(g,ggml_view_2d(g,ba,vpg,Hk,ba->nb[1],(size_t)vpg*ggml_element_size(ba))); // alphas
                        beta=ggml_sigmoid(g,ggml_reshape_2d(g,br,Hv,1));
                        al  =ggml_reshape_2d(g,ar,Hv,1);
                    } else {          // qwen35moe: separate ssm_beta / ssm_alpha projections
                        beta=ggml_sigmoid(g,ggml_mul_mat(g,M.blk(il,"ssm_beta.weight"),xn));     // [Hv]
                        al  =ggml_mul_mat(g,M.blk(il,"ssm_alpha.weight"),xn);                     // [Hv]
                    }
                    beta=ggml_reshape_4d(g,beta,1,Hv,1,1);
                    al=ggml_softplus(g,ggml_add(g,al,M.blk(il,"ssm_dt.bias")));
                    ggml_tensor* gg=ggml_reshape_4d(g,ggml_mul(g,al,M.blk(il,"ssm_a")),1,Hv,1,1);         // decay
                    // causal conv1d over [conv_state(3) ++ new(1)] then silu
                    ggml_tensor* cs=conv_state[il];                                    // [d_conv-1, cd]
                    ggml_tensor* qkvT=ggml_transpose(g,ggml_reshape_2d(g,qkv,cd,1));    // [1, cd]
                    ggml_tensor* cin=ggml_concat(g,cs,qkvT,0);                          // [d_conv, cd]
                    ggml_tensor* convo=ggml_silu(g,ggml_ssm_conv(g,cin,M.blk(il,"ssm_conv1d.weight"))); // [cd,1]
                    extra2=ggml_cpy(g,ggml_view_2d(g,cin,ssm_d_conv-1,cd,cin->nb[1],cin->nb[0]),cs);     // roll conv state
                    size_t cz=ggml_element_size(convo);
                    ggml_tensor* qc=ggml_l2_norm(g,ggml_cont(g,ggml_view_3d(g,convo,Sv,Hk,1,cz*Sv,cz*kd,0)),eps);
                    ggml_tensor* kc2=ggml_l2_norm(g,ggml_cont(g,ggml_view_3d(g,convo,Sv,Hk,1,cz*Sv,cz*kd,cz*kd)),eps);
                    ggml_tensor* vc2=ggml_cont(g,ggml_view_3d(g,convo,Sv,Hv,1,cz*Sv,cz*vd,cz*kd*2));
                    qc=ggml_reshape_4d(g,qc,Sv,Hk,1,1); kc2=ggml_reshape_4d(g,kc2,Sv,Hk,1,1); vc2=ggml_reshape_4d(g,vc2,Sv,Hv,1,1);
                    // qwen3next explicitly repeat-interleaves q/k from Hk to Hv heads BEFORE the fused GDN ([g0,g0,g1,g1,...]).
                    // qwen35moe does NOT (its fused path relies on the op's internal broadcast) -> gate strictly on arch, NOT on Hv!=Hk.
                    if(is_qwen3next && Hv!=Hk){ int vpg=Hv/Hk;
                        qc =ggml_reshape_4d(g,ggml_repeat_4d(g,ggml_reshape_4d(g,qc ,Sv,1,Hk,1),Sv,vpg,Hk,1),Sv,Hv,1,1);
                        kc2=ggml_reshape_4d(g,ggml_repeat_4d(g,ggml_reshape_4d(g,kc2,Sv,1,Hk,1),Sv,vpg,Hk,1),Sv,Hv,1,1); }
                    ggml_tensor* s0=ggml_reshape_4d(g,gdn_state[il],Sv,Sv,Hv,1);
                    ggml_tensor* gdn=ggml_gated_delta_net(g,qc,kc2,vc2,gg,beta,s0,1);
                    size_t rs1=ggml_row_size(gdn->type,Sv);
                    ggml_tensor* out=ggml_view_4d(g,gdn,Sv,Hv,1,1,rs1,ggml_row_size(gdn->type,Sv*Hv),ggml_row_size(gdn->type,Sv*Hv),0);
                    ggml_tensor* ns=ggml_view_4d(g,gdn,Sv,Sv,Hv,1,rs1,ggml_row_size(gdn->type,Sv*Sv),ggml_row_size(gdn->type,Sv*Sv*Hv),ggml_row_size(gdn->type,Sv*Hv));
                    extra1=ggml_cpy(g,ns,gdn_state[il]);                               // write back GDN state
                    ggml_tensor* zr=ggml_reshape_4d(g,z,Sv,Hv,1,1);
                    ggml_tensor* on=ggml_mul(g,ggml_mul(g,ggml_rms_norm(g,out,eps),M.blk(il,"ssm_norm.weight")),ggml_silu(g,zr));
                    att=ggml_mul_mat(g,M.blk(il,"ssm_out.weight"),ggml_reshape_2d(g,on,vd,1));
                }
                ggml_tensor* ffn_inp=ggml_add(g,att,ix);
                ggml_tensor* fx=ggml_mul(g,ggml_rms_norm(g,ffn_inp,eps),M.blk(il,"post_attention_norm.weight"));
                ggml_tensor* rlog=ggml_mul_mat(g,M.blk(il,"ffn_gate_inp.weight"),fx);
                ggml_tensor* probs=ggml_soft_max(g,rlog);
                ggml_tensor* selT=ggml_top_k(g,probs,n_used);
                ggml_tensor* wT=ggml_get_rows(g,ggml_reshape_3d(g,probs,1,n_expert,1),selT);
                ggml_set_output(fx); ggml_set_output(ffn_inp); ggml_set_output(selT); ggml_set_output(wT);
                if(gatedump){ ggml_set_output(probs); ggml_build_forward_expand(gf,probs); }
                ggml_build_forward_expand(gf,fx); ggml_build_forward_expand(gf,ffn_inp);
                ggml_build_forward_expand(gf,selT); ggml_build_forward_expand(gf,wT);
                if(extra1) ggml_build_forward_expand(gf,extra1);
                if(extra2) ggml_build_forward_expand(gf,extra2);
                ggml_gallocr_alloc_graph(galloc,gf);
                ggml_backend_tensor_set(ix,x.data(),0,(size_t)n_embd*4);
                if(!is_recr[il]){ int32_t pp[4]={p,p,p,p}; ggml_backend_tensor_set(ip,pp,0,(mrope?4:1)*4); } // mrope: 4 sections all=p (text); else single pos
                ggml_backend_graph_compute(backend,gf);
                ggml_backend_tensor_get(fx,ffnx.data(),0,(size_t)n_embd*4);
                ggml_backend_tensor_get(ffn_inp,ffninp.data(),0,(size_t)n_embd*4);
                ggml_backend_tensor_get(selT,sel.data(),0,(size_t)n_used*4);
                if(p>=T-1) for(int j=0;j<n_used;++j) touched.insert(il*n_expert+sel[j]); // working set over DECODE
                if(trace && p>=T-1){ if(il==0) seltrace.emplace_back(); seltrace.back().emplace_back(sel.begin(),sel.begin()+n_used); }
                if(pf && p>=T-1){ fprintf(pf,"%d %d",(int)gen.size(),il); for(int j=0;j<n_used;++j) fprintf(pf," %d",sel[j]); fprintf(pf,"\n"); } // lean access trace
                ggml_backend_tensor_get(wT,wsel.data(),0,(size_t)n_used*4);
                if(gatedump && p>=T-1){ pr.resize(n_expert); ggml_backend_tensor_get(probs,pr.data(),0,(size_t)n_expert*4); } // full router softmax -> host (read before ggml_free)
                if(wnorm){ float ws=0; for(int j=0;j<n_used;++j) ws+=wsel[j];
                    if(ws<6.103515625e-5f) ws=6.103515625e-5f;              // clamp (matches llama.cpp F16 min)
                    for(int j=0;j<n_used;++j) wsel[j]/=ws; }                // renormalize top-k weights to sum 1
                size_t cb=ggml_gallocr_get_buffer_size(galloc,0); if(cb>peak_compute)peak_compute=cb;
                ggml_free(g);
            }
            if(p>=T) tA+=NOW()-_ta;
            if(gatedump && p>=T-1 && (int)pr.size()==n_expert){ // residency-substitution ceiling data: gate band + decision-time residency tier (snapshot BEFORE this token's fetch/eviction)
                float wk=1e30f; for(int j=0;j<n_used;++j){ float w=pr[sel[j]]; if(w<wk) wk=w; }
                float band=wk-0.2f;
                fprintf(gdf,"%d %d %.6g",(int)gen.size(),il,wk);
                for(int e=0;e<n_expert;++e){ if(pr[e]<band) continue;
                    int issel=0; for(int j=0;j<n_used;++j) if(sel[j]==e){issel=1;break;}
                    int res = S.expert_slot[il].count(e)?1 : (RT.key2slot.count(il*n_expert+e)?2:0);
                    fprintf(gdf," %d:%.6g:%d:%d",e,pr[e],issel,res); }
                fprintf(gdf,"\n");
            }
            long long _tf=NOW();
            // ---- stream selected experts: parallel device-direct reads, then stage RAM->VRAM ----
            std::vector<int> slotidx(n_used);
            {
                std::vector<int> ramslot(n_used,-1); std::vector<char> vhit(n_used,0);
                std::vector<IOJob> jobs; jobs.reserve(n_used*3);
                // Phase A (serial): classify each expert, reserve RAM slots, build disk-read jobs (no I/O yet)
                for(int j=0;j<n_used;++j){ int e=sel[j]; reqs++;
                    if(mode!="naive"){ auto it=S.expert_slot[il].find(e);
                        if(it!=S.expert_slot[il].end()){ vram_hit++; S.lru[il].remove(it->second); S.lru[il].push_front(it->second); slotidx[j]=it->second; vhit[j]=1; continue; } }
                    int key=il*n_expert+e; auto rit=RT.key2slot.find(key);
                    if(rit!=RT.key2slot.end()){ ram_hit++; RT.lru.remove(rit->second); RT.lru.push_front(rit->second); ramslot[j]=rit->second; }
                    else { int rs=RT.lru.back(); RT.lru.pop_back(); int old=RT.slot2key[rs]; if(old>=0) RT.key2slot.erase(old);
                        if(!RT.gbuf[rs]){ RT.gbuf[rs]=(uint8_t*)VirtualAlloc(NULL,gbsz,MEM_COMMIT|MEM_RESERVE,PAGE_READWRITE);   // lazy aligned alloc
                            RT.ubuf[rs]=(uint8_t*)VirtualAlloc(NULL,ubsz,MEM_COMMIT|MEM_RESERVE,PAGE_READWRITE);
                            RT.dbuf[rs]=(uint8_t*)VirtualAlloc(NULL,dbsz,MEM_COMMIT|MEM_RESERVE,PAGE_READWRITE); }
                        RT.slot2key[rs]=key; RT.key2slot[key]=rs; RT.lru.push_front(rs); ramslot[j]=rs; nvme_fall++;
                        uint64_t go=H.goff[il]+(size_t)e*H.gstride[il], uo=H.uoff[il]+(size_t)e*H.ustride[il], od=H.doff[il]+(size_t)e*H.dstride[il];
                        uint32_t gs=(uint32_t)H.gstride[il], us=(uint32_t)H.ustride[il], ds=(uint32_t)H.dstride[il];
                        RT.ghead[rs]=(uint32_t)(go&4095); RT.uhead[rs]=(uint32_t)(uo&4095); RT.dhead[rs]=(uint32_t)(od&4095);
                        jobs.push_back(IOJob{RT.gbuf[rs], go&~(uint64_t)4095, (RT.ghead[rs]+gs+4095u)&~4095u, RT.ghead[rs], gs}); // read direct into aligned buf
                        jobs.push_back(IOJob{RT.ubuf[rs], uo&~(uint64_t)4095, (RT.uhead[rs]+us+4095u)&~4095u, RT.uhead[rs], us});
                        jobs.push_back(IOJob{RT.dbuf[rs], od&~(uint64_t)4095, (RT.dhead[rs]+ds+4095u)&~4095u, RT.dhead[rs], ds});
                    }
                }
                // Phase B (parallel device-direct): execute the cold reads at QD up to io.n
                struct timespec d0,d1; clock_gettime(CLOCK_MONOTONIC,&d0);
                run_jobs(io,jobs);
                clock_gettime(CLOCK_MONOTONIC,&d1);
                { long long _dd=(d1.tv_sec-d0.tv_sec)*1000000000LL+(d1.tv_nsec-d0.tv_nsec); disk_ns+=_dd; if(p>=T) tDisk+=_dd;
                  // storage-bandwidth throttle: cap effective drive speed by sleeping out the rest of this batch's
                  // budget. Gated to the timed decode region (p>=T) so one-time prefill cold-load isn't throttled.
                  if(throttle_Bps>0 && p>=T){ size_t bb=0; for(auto&jb:jobs) bb+=jb.rsize;
                      if(bb>0){ long long tgt=(long long)((double)bb/throttle_Bps*1e9), slp=tgt-_dd;
                          if(slp>0){ struct timespec ts{(time_t)(slp/1000000000LL),(long)(slp%1000000000LL)}; nanosleep(&ts,nullptr);
                              disk_ns+=slp; if(p>=T) tDisk+=slp; } } } }
                for(auto&jb:jobs) disk_bytes += jb.rsize;
                // Phase C (serial): stage RAM->VRAM (ggml/Vulkan backend is single-threaded)
                for(int j=0;j<n_used;++j){ if(vhit[j]) continue; int e=sel[j], rs=ramslot[j];
                    if(mode=="naive"){ to_vram(il,e,rs,j); slotidx[j]=j; }
                    else { int slot=S.lru[il].back(); S.lru[il].pop_back(); int old=S.slot_expert[il][slot]; if(old>=0) S.expert_slot[il].erase(old);
                        to_vram(il,e,rs,slot); S.slot_expert[il][slot]=e; S.expert_slot[il][e]=slot; S.lru[il].push_front(slot); slotidx[j]=slot; }
                }
            }
            if(p>=T) tFetch+=NOW()-_tf;
            long long _tb=NOW();
            // ---- graph B: expert FFN over the 8 slots ----
            std::vector<float> eout((size_t)n_embd*n_used);
            std::vector<float> shex(is_qwen35?n_embd:0,0.0f);
            {
                ggml_context* g=ggml_init({(size_t)64*1024*1024,nullptr,true});
                ggml_cgraph* gf=ggml_new_graph_custom(g,2048,false);
                ggml_tensor* ifx=ggml_new_tensor_2d(g,GGML_TYPE_F32,n_embd,1); ggml_set_input(ifx);
                std::vector<ggml_tensor*> ys;
                for(int j=0;j<n_used;++j){ int s=slotidx[j];
                    ggml_tensor* gj=ggml_mul_mat(g,S.gate[il][s],ifx);
                    ggml_tensor* uj=ggml_mul_mat(g,S.up[il][s],ifx);
                    ggml_tensor* hj=ggml_mul(g,ggml_silu(g,gj),uj);
                    ys.push_back(ggml_mul_mat(g,S.down[il][s],hj)); // [n_embd,1]
                }
                ggml_tensor* stack=ys[0];
                for(int j=1;j<n_used;++j) stack=ggml_concat(g,stack,ys[j],1); // [n_embd,n_used]
                ggml_set_output(stack); ggml_build_forward_expand(gf,stack);
                ggml_tensor* she=nullptr;
                if(is_qwen35){ // gated shared expert: sigmoid(gate_inp_shexp.x) * down(silu(gate.x)*up.x)
                    ggml_tensor* sg=ggml_mul_mat(g,M.blk(il,"ffn_gate_shexp.weight"),ifx);
                    ggml_tensor* su=ggml_mul_mat(g,M.blk(il,"ffn_up_shexp.weight"),ifx);
                    ggml_tensor* sh=ggml_mul_mat(g,M.blk(il,"ffn_down_shexp.weight"),ggml_mul(g,ggml_silu(g,sg),su));
                    ggml_tensor* sgate=ggml_sigmoid(g,ggml_mul_mat(g,M.blk(il,"ffn_gate_inp_shexp.weight"),ifx));
                    she=ggml_mul(g,sh,sgate);
                    ggml_set_output(she); ggml_build_forward_expand(gf,she);
                }
                ggml_gallocr_alloc_graph(galloc,gf);
                ggml_backend_tensor_set(ifx,ffnx.data(),0,(size_t)n_embd*4);
                ggml_backend_graph_compute(backend,gf);
                ggml_backend_tensor_get(stack,eout.data(),0,(size_t)n_embd*n_used*4);
                if(she) ggml_backend_tensor_get(she,shex.data(),0,(size_t)n_embd*4);
                ggml_free(g);
            }
            // host weighted sum + (gated shared expert) + residual
            for(int d=0;d<n_embd;++d){ float acc=0; for(int j=0;j<n_used;++j) acc+=wsel[j]*eout[(size_t)j*n_embd+d];
                x[d]=acc+(is_qwen35?shex[d]:0.0f)+ffninp[d]; }
            if(p>=T) tB+=NOW()-_tb;
        }
        long long _th=NOW();
        // ---- head ----
        std::vector<float> logits(n_vocab);
        {
            ggml_context* g=ggml_init({(size_t)64*1024*1024,nullptr,true});
            ggml_cgraph* gf=ggml_new_graph(g);
            ggml_tensor* ix=ggml_new_tensor_2d(g,GGML_TYPE_F32,n_embd,1); ggml_set_input(ix);
            ggml_tensor* rn=ggml_mul(g,ggml_rms_norm(g,ix,eps),M.get("output_norm.weight"));
            ggml_tensor* lg=ggml_mul_mat(g,M.get("output.weight"),rn); ggml_set_output(lg);
            ggml_build_forward_expand(gf,lg); ggml_gallocr_alloc_graph(galloc,gf);
            ggml_backend_tensor_set(ix,x.data(),0,(size_t)n_embd*4);
            ggml_backend_graph_compute(backend,gf);
            ggml_backend_tensor_get(lg,logits.data(),0,(size_t)n_vocab*4); ggml_free(g);
        }
        if(p>=T) tHead+=NOW()-_th;
        if(p>=T-1){
            if(dumpL){ // top-5 logits to expose near-ties / compare vs reference
                int t5[5]; for(int k=0;k<5;++k){int bi=-1;float bv=-1e30f;
                    for(int v=0;v<n_vocab;++v){bool u=false;for(int q=0;q<k;++q)if(t5[q]==v)u=true; if(!u&&logits[v]>bv){bv=logits[v];bi=v;}} t5[k]=bi;}
                fprintf(stderr,"  pos%d top5:",p-(T-1)); for(int k=0;k<5;++k)fprintf(stderr," %d=%.3f",t5[k],logits[t5[k]]);
                fprintf(stderr,"  (gap %.3f)\n",logits[t5[0]]-logits[t5[1]]);
            }
            int bi=0; { float bv=logits[0]; for(int v=1;v<n_vocab;++v) if(logits[v]>bv){bv=logits[v];bi=v;} } // argmax (also the temp fallback)
            if(g_temp>0.0f && !tf){   // temperature sampling over the head of the distribution (logit floor avoids tail garbage)
                double mx=(double)logits[bi], cut=mx-12.0, sum=0;
                for(int v=0;v<n_vocab;++v) if((double)logits[v]>=cut) sum+=exp(((double)logits[v]-mx)/g_temp);
                double r=std::uniform_real_distribution<double>(0.0,sum)(g_rng), acc=0;
                for(int v=0;v<n_vocab;++v) if((double)logits[v]>=cut){ acc+=exp(((double)logits[v]-mx)/g_temp); if(acc>=r){ bi=v; break; } }
            }
            gen.push_back(bi); on_token(bi);
            if(gen.size()==1) clock_gettime(CLOCK_MONOTONIC,&tfirst);   // first token out => prefill (TTFT) done
            if(tf && p+1<(int)seq.size()){ int nt=seq[p+1];   // teacher-forced NLL of the true next token -> perplexity (quality probe)
                double mx=logits[0]; for(int v=1;v<n_vocab;++v) if((double)logits[v]>mx) mx=logits[v];
                double se=0; for(int v=0;v<n_vocab;++v) se+=exp((double)logits[v]-mx);
                nll_sum += (mx+log(se))-(double)logits[nt]; nll_n++; }
            if(csvf){ long long now=NOW(); if(!tok_first_ns) tok_first_ns=now;
                double t_ms=(now-tok_first_ns)/1e6, dt_ms=tok_prev_ns?(now-tok_prev_ns)/1e6:0; tok_prev_ns=now;
                fprintf(csvf,"%zu,%.3f,%.3f,%ld,%ld,%ld,%ld,%zu,%.3f\n",
                    gen.size(),t_ms,dt_ms,vram_hit,ram_hit,nvme_fall,reqs,touched.size(),disk_bytes/1e6); fflush(csvf); }
            if(mode!="chat" && gen.size()%16==0) // saturation curve: cumulative distinct experts vs token position
                fprintf(stderr,"[ws@%3zu tok] %4zu distinct experts (%.1f%% of model, %.0f MB)\n",
                    gen.size(),touched.size(),100.0*touched.size()/(double)(n_layer*n_expert),
                    [&]{double mb=0;for(int key:touched)mb+=H.gstride[key/n_expert]+H.ustride[key/n_expert]+H.dstride[key/n_expert];return mb/1e6;}());
            if(tf){ if(p+1>=(int)seq.size()) break; }              // teacher-force: follow given seq
            else { static const bool noeos=getenv("NOEOS")!=nullptr; if(g_stop||(int)gen.size()>=ngen||(bi==eos_id&&!noeos)) break; seq.push_back(bi); } // NOEOS: keep generating to ngen ; g_stop: Stop button
        }
    }
    clock_gettime(CLOCK_MONOTONIC,&t1);
    };  // ---- end generate lambda ----

    if(mode=="chat"){
        // ---------------- CHAT REPL: multi-line text in, streamed text out (model loaded once) ----------------
        llama_model_params vp=llama_model_default_params(); vp.vocab_only=true;
        llama_model* lm=llama_model_load_from_file(argv[1],vp);
        if(!lm){ fprintf(stderr,"vocab load failed\n"); return 1; }
        const llama_vocab* vocab=llama_model_get_vocab(lm);
        std::vector<int32_t> hist;                              // running conversation tokens (multi-turn)
        int g_state_pos=0;   // INCREMENTAL: positions [0,g_state_pos) already have valid KV/GDN/conv state (carried across turns)
        auto respond=[&](const std::string& msg){
            // qwen3.5: disable thinking by pre-filling an empty <think></think> block (enable_thinking=false).
            // qwen3: the /no_think soft switch works. Either way -> concise answers, not 1k tokens of hidden reasoning.
            std::string turn = is_qwen3next
                ? "<|im_start|>user\n"+msg+"<|im_end|>\n<|im_start|>assistant\n"                       // qwen3next-Instruct: no thinking, plain turn
                : is_qwen35
                ? "<|im_start|>user\n"+msg+"<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"  // qwen3.5: empty <think> = thinking off
                : "<|im_start|>user\n"+msg+" /no_think<|im_end|>\n<|im_start|>assistant\n";             // qwen3: /no_think soft switch
            int32_t need=llama_tokenize(vocab,turn.c_str(),(int)turn.size(),nullptr,0,false,true);
            std::vector<llama_token> tt(need<0?-need:need);
            llama_tokenize(vocab,turn.c_str(),(int)turn.size(),tt.data(),(int)tt.size(),false,true);
            std::vector<int32_t> sq=hist; for(auto t:tt) sq.push_back((int32_t)t);
            int Tp=(int)sq.size();
            int think_open=(is_qwen35&&!is_qwen3next)?248068:151667, think_close=(is_qwen35&&!is_qwen3next)?248069:151668; // <think>/</think> ids (qwen35moe vocab vs standard Qwen)
            gen.clear(); bool inthink=false; printf("\n");
            int maxgen=getenv("MAXGEN")?atoi(getenv("MAXGEN")):1024;   // cap chat generation (corpus collection wants bounded traces)
            int batched=getenv("BATCH_PREFILL")?(getenv("PREFILL_CHUNK")?atoi(getenv("PREFILL_CHUNK")):256):0;
            int resume=(batched && g_state_pos>0 && g_state_pos<Tp && getenv("NO_INCREMENTAL")==nullptr)?g_state_pos:0; // incremental: only prefill new tokens
            if(batched) fprintf(stderr,"[turn] prefill %d new tokens of %d ctx%s\n",Tp-resume,Tp,resume?" [incremental resume]":" [cold]");
            generate(sq,Tp,maxgen,[&](int id){
                if(id==think_open){inthink=true;return;} if(id==think_close){inthink=false;return;}  // hide <think>...</think>
                if(inthink||id==eos_id) return;
                char b[256]; int k=llama_token_to_piece(vocab,id,b,256,0,false); if(k>0){ fwrite(b,1,k,stdout); fflush(stdout); } },
                batched,resume);
            printf("\n");
            g_state_pos=(int)sq.size();   // decode left valid KV/GDN/conv state for all of sq (the eos appended below is processed next turn)
            if(sq.empty()||sq.back()!=eos_id) sq.push_back(eos_id);
            hist=sq;
            double s=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
            fprintf(stderr,"[%zu tokens, %.2f tok/s]\n", gen.size(), s>0?gen.size()/s:0);
        };
        // restore a saved conversation's context: rebuild the chat template from stored turns and set hist.
        // The next respond() re-prefills it (engine has no batched prefill, so cost ~ conversation length).
        auto load_history=[&](const std::vector<std::pair<char,std::string>>& msgs){
            std::string s;
            for(auto& m:msgs){
                if(m.first=='U') s += "<|im_start|>user\n"+m.second+"<|im_end|>\n";
                else s += "<|im_start|>assistant\n"+((is_qwen35&&!is_qwen3next)?std::string("<think>\n\n</think>\n\n"):std::string())+m.second+"<|im_end|>\n";
            }
            int32_t need=llama_tokenize(vocab,s.c_str(),(int)s.size(),nullptr,0,false,true);
            std::vector<llama_token> tt(need<0?-need:need);
            if(!tt.empty()) llama_tokenize(vocab,s.c_str(),(int)s.size(),tt.data(),(int)tt.size(),false,true);
            hist.assign(tt.begin(),tt.end()); g_state_pos=0;   // restored tokens have NO built state -> next respond() cold-prefills them
            fprintf(stderr,"[load] %zu turns -> %zu ctx tokens\n",msgs.size(),hist.size());
        };
        if(getenv("SERVE")){
            // headless server protocol (driven by server/serve.py): messages are <<<SEND>>>-delimited (content-safe,
            // unlike blank-line), <<<RESET>>> starts a new conversation; each turn ends with a 0x04 byte on stdout
            // (ASCII End-Of-Transmission, never produced by the tokenizer) so the reader knows the answer is complete.
            fprintf(stderr,"[serve] ready: %s K=%d %d/%d experts resident\n",ARCH.c_str(),K,ram_cap,total_experts); fflush(stderr);
            // dedicated stdin reader so <<<STOP>>> can be seen DURING generation; everything else is queued.
            std::deque<std::string> q; std::mutex qm; std::condition_variable qcv; bool eof=false;
            std::thread reader([&]{
                std::string l;
                while(std::getline(std::cin,l)){
                    if(l=="<<<STOP>>>"){ g_stop=true; continue; }                                          // out-of-band: interrupt generation
                    { std::lock_guard<std::mutex> lk(qm); q.push_back(l); } qcv.notify_one();
                }
                { std::lock_guard<std::mutex> lk(qm); eof=true; } qcv.notify_one();
            });
            std::string msg,line;
            std::vector<std::pair<char,std::string>> loadmsgs; bool loading=false;
            for(;;){
                { std::unique_lock<std::mutex> lk(qm); qcv.wait(lk,[&]{return !q.empty()||eof;}); if(q.empty()){ break; } line=q.front(); q.pop_front(); }
                if(line=="<<<LOAD>>>"){ loadmsgs.clear(); loading=true; continue; }                        // begin context restore
                if(line=="<<<ENDLOAD>>>"){ loading=false; load_history(loadmsgs); putchar(4); fflush(stdout); continue; }
                if(loading){ if(line.size()>=2 && (line[0]=='U'||line[0]=='A') && line[1]==' ') loadmsgs.push_back({line[0],b64dec(line.substr(2))}); continue; } // "U <b64>" / "A <b64>"
                if(line=="<<<RESET>>>"){ hist.clear(); g_state_pos=0; msg.clear(); putchar(4); fflush(stdout); continue; }
                if(line.rfind("<<<TEMP ",0)==0){ g_temp=(float)atof(line.c_str()+8); fprintf(stderr,"[temp] %.2f\n",g_temp); continue; } // "<<<TEMP 0.7>>>"
                if(line=="<<<SEND>>>"){ if(!msg.empty()){ if(pf){ fprintf(pf,"# turn\n"); fflush(pf);} g_stop=false; respond(msg); msg.clear(); } putchar(4); fflush(stdout); continue; }
                if(line=="/quit") break;
                if(!msg.empty()) msg+="\n"; msg+=line;
            }
            g_stop=true; reader.detach();   // process is terminating; don't block on the reader's getline
            if(pf) fclose(pf); llama_model_free(lm); return 0;
        }
        fprintf(stderr,"\n=== %s streaming chat (K=%d, %d/%d experts resident, rest stream from NVMe) ===\n",ARCH.c_str(),K,ram_cap,total_experts);
        fprintf(stderr,"multi-line ok (paste code); blank line sends, /quit exits.\n");
        std::string msg,line;
        while(true){
            fprintf(stderr, msg.empty()?"\nuser> ":"...   "); fflush(stderr);
            if(!std::getline(std::cin,line)){ if(!msg.empty()) respond(msg); break; }      // EOF: send any pending
            if(line=="/quit"||line=="/exit") break;
            if(line.empty()){ if(!msg.empty()){ respond(msg); msg.clear(); } continue; }    // blank line: send
            if(!msg.empty()) msg+="\n"; msg+=line;
        }
        llama_model_free(lm); fprintf(stderr,"\nbye.\n"); return 0;
    }

    // batched prefill is enabled by BATCH_PREFILL=1 (chunk PREFILL_CHUNK, default 256); it builds the prompt
    // state in batched graphs then decodes per-position from p=T. Output is token-identical to per-position.
    int batchC = getenv("BATCH_PREFILL")?(getenv("PREFILL_CHUNK")?atoi(getenv("PREFILL_CHUNK")):256):0;
    if(getenv("BENCH")){   // prefill benchmark: run prefill_batched K times in-process (rep0 cold, rest warm RAM tier) -> noise-averaged TTFT
        int K=atoi(getenv("BENCH")); if(K<1)K=1; int C=batchC>0?batchC:T;
        fprintf(stderr,"\nBENCH prefill: T=%d  C=%d  %d reps\n",T,C,K);
        double warm=0,coldttft=0;
        for(int r=0;r<K;++r){ std::vector<float> lg; struct timespec a,b; clock_gettime(CLOCK_MONOTONIC,&a);
            prefill_batched(seq,T,C,lg,0); clock_gettime(CLOCK_MONOTONIC,&b);
            double s=(b.tv_sec-a.tv_sec)+(b.tv_nsec-a.tv_nsec)/1e9;
            fprintf(stderr,"  rep%d  TTFT=%7.2fs  %6.1f ms/tok  %.1f tok/s%s\n",r,s,T>0?s*1000.0/T:0,s>0?T/s:0,r==0?"  [cold]":"  [warm]");
            if(r==0)coldttft=s; else warm+=s; }
        if(K>1) fprintf(stderr,"BENCH T=%d: cold=%.2fs  warm-mean=%.2fs (%.1f ms/tok, %.1f tok/s)\n",T,coldttft,warm/(K-1),T>0?warm/(K-1)*1000.0/T:0,warm>0?(K-1)*T/warm:0);
        return 0;
    }
    if(getenv("VALIDATE_PREFILL")){   // full-sequence token-exact check: per-position decode vs batched-prefill+decode
        int nv=getenv("VNGEN")?atoi(getenv("VNGEN")):16;
        std::vector<int> gp,gb;
        { std::vector<int32_t> s1(seq.begin(),seq.begin()+T); gen.clear(); generate(s1,T,nv,[&](int id){gp.push_back(id);},0,0); }
        { std::vector<int32_t> s2(seq.begin(),seq.begin()+T); gen.clear(); generate(s2,T,nv,[&](int id){gb.push_back(id);},batchC>0?batchC:256,0); }
        bool match=(gp==gb);
        printf("VALIDATE_PREFILL: per-pos vs batched(C=%d) over %d gen tokens => %s\n",batchC>0?batchC:256,(int)gp.size(),match?"MATCH":"*** MISMATCH ***");
        printf("  per-pos:"); for(int id:gp)printf(" %d",id); printf("\n  batched:"); for(int id:gb)printf(" %d",id); printf("\n");
        return match?0:1;
    }
    // ---- one-shot generation (benchmark / validation) ----
    gen.clear();
    generate(seq,T,ngen,[](int){},batchC,0);
    if(csvf) fclose(csvf);
    if(gdf) fclose(gdf);
    if(pf) fclose(pf);
    if(nll_n>0) printf("QUALITY: teacher-forced over %ld positions: mean NLL=%.5f  perplexity=%.4f\n", nll_n, nll_sum/nll_n, exp(nll_sum/nll_n));
    double secs=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
    double prefill_s=(tfirst.tv_sec-tg0.tv_sec)+(tfirst.tv_nsec-tg0.tv_nsec)/1e9;
    printf("PREFILL: %d prompt tokens, TTFT=%.2f s = %.1f tok/s prefill = %.1f ms/token\n",
        T, prefill_s, prefill_s>0?T/prefill_s:0, T>0?prefill_s*1000.0/T:0);
    vram += peak_compute;

    double per_exp=(H.gstride[0]+H.ustride[0]+H.dstride[0]);
    printf("GEN:"); for(int32_t id:gen) printf(" %d",id); printf("\n");
    printf("mode=%s K=%d ram_cap=%d/%d experts (%.0f%% of model in RAM)  tok/s=%.2f\n",
        mode.c_str(),K,ram_cap,total_experts,100.0*ram_cap/total_experts,gen.size()/secs);
    printf("peak_VRAM=%.1f MB ; RAM tier=%.0f MB ; experts on disk=%.0f MB\n",
        vram/1e6, ram_cap*per_exp/1e6, total_experts*per_exp/1e6);
    printf("THREE-TIER access split (%ld requests):  VRAM=%.1f%%  RAM=%.1f%%  NVMe=%.1f%%\n",
        reqs, reqs?100.0*vram_hit/reqs:0, reqs?100.0*ram_hit/reqs:0, reqs?100.0*nvme_fall/reqs:0);
    long miss=ram_hit+nvme_fall;
    printf("  of VRAM misses (%ld): RAM caught %.1f%%, NVMe fall-through %.1f%%   [disk reads=%ld, %.0f MB]\n",
        miss, miss?100.0*ram_hit/miss:0, miss?100.0*nvme_fall/miss:0, nvme_fall, nvme_fall*per_exp/1e6);
    // ---- NVMe latency (isolated from compute): time/bytes spent in disk reads ----
    if(nvme_fall>0){
        double dms=disk_ns/1e6, avg=dms/nvme_fall, gbps=(double)disk_bytes/(disk_ns>0?disk_ns:1); // bytes/ns = GB/s
        printf("NVMe: %ld disk reads, %.0f MB in %.1f s = %.2f GB/s ; avg %.3f ms/expert-read (%.0f KB each); disk = %.0f%% of wall\n",
            nvme_fall, disk_bytes/1e6, dms/1000.0, gbps, avg, (double)disk_bytes/nvme_fall/1e3, 100.0*disk_ns/1e9/secs);
    }
    // ---- DECODE BREAKDOWN: where the per-token wall actually goes (gate the next optimization) ----
    { double st=(tFetch-tDisk)/1e9;
      printf("DECODE BREAKDOWN (%.2fs decode): graphA(attn+router) %.0f%% | fetch %.0f%% [disk %.0f%% + H2D-stage %.0f%%] | graphB(expertFFN)+sum %.0f%% | embed %.0f%% | head %.0f%%\n",
        secs, 100.0*tA/1e9/secs, 100.0*tFetch/1e9/secs, 100.0*tDisk/1e9/secs, 100.0*st/secs, 100.0*tB/1e9/secs, 100.0*tE/1e9/secs, 100.0*tHead/1e9/secs);
      // absolute per-token phase times (ms) -- resolves the spill tax that integer percentages hide:
      // graphB reads expert operands from VRAM slots; when slots spill to shared RAM (K past the 8 GB
      // frontier), graphB ms/tok rises iff the spill costs PCIe bandwidth. Flat => spill is ~free.
      printf("PHASE ms/tok: graphA=%.3f fetch=%.3f disk=%.3f H2D=%.3f graphB=%.3f embed=%.3f head=%.3f (ngen=%d)\n",
        tA/1e6/ngen, tFetch/1e6/ngen, tDisk/1e6/ngen, st*1e3/ngen, tB/1e6/ngen, tE/1e6/ngen, tHead/1e6/ngen, ngen);
    }
    // ---- PREFETCH PREDICTOR CEILING: cross-token routing locality ----
    // predict token t+1's layer-L experts from token t's layer-L selection; ceiling = avg overlap/n_used.
    if(trace && seltrace.size()>=2){
        int nL=(int)seltrace[0].size();
        std::vector<double> ovL(nL,0); std::vector<int> cntL(nL,0); double tot=0; int totc=0;
        for(size_t t=1;t<seltrace.size();++t) for(int L=0;L<nL && L<(int)seltrace[t].size();++L){
            std::set<int> prev(seltrace[t-1][L].begin(),seltrace[t-1][L].end());
            int inter=0; for(int e:seltrace[t][L]) if(prev.count(e)) inter++;
            ovL[L]+=inter; cntL[L]++; tot+=inter; totc++;
        }
        double lo=1e9,hi=-1; int loL=0,hiL=0;
        for(int L=0;L<nL;++L){ double a=ovL[L]/cntL[L]; if(a<lo){lo=a;loL=L;} if(a>hi){hi=a;hiL=L;} }
        fprintf(stderr,"PREFETCH CEILING: avg consecutive-token top-%d overlap = %.2f/%d = %.1f%% (predict t+1 from t)\n",
            n_used, tot/totc, n_used, 100.0*tot/totc/n_used);
        fprintf(stderr,"  per-layer range: most-stable L%d=%.1f%%  least-stable L%d=%.1f%%  (over %zu decode tokens)\n",
            hiL,100.0*hi/n_used, loL,100.0*lo/n_used, seltrace.size());
        // a few sample layers (early / mid / late)
        for(int L : {0, nL/4, nL/2, 3*nL/4, nL-1}) fprintf(stderr,"    L%-2d %.1f%%\n",L,100.0*ovL[L]/cntL[L]/n_used);
    }
    // ---- WORKING SET (the green/cliff determinant): distinct experts touched over decode ----
    double ws=0; for(int key:touched) ws += H.gstride[key/n_expert]+H.ustride[key/n_expert]+H.dstride[key/n_expert];
    printf("WORKING SET over %d decode tokens: %zu distinct experts = %.0f MB (%.0f%% of %d experts; full model experts=%.0f MB)\n",
        (int)gen.size(), touched.size(), ws/1e6, 100.0*touched.size()/total_experts, total_experts, total_experts*per_exp/1e6);
    return 0;
}
