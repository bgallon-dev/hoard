// chat.cpp — interactive text interface to the from-scratch streaming-MoE engine.
//
// Text in / text out: tokenize with llama.dll (same path as ref_gen.cpp), run the hand-built
// streaming MoE forward pass (the exact per-token decode from run_moe_stream.cpp — non-expert
// weights resident in VRAM, experts streamed from host RAM into a bounded per-layer VRAM slot
// pool with an LRU residency cache), detokenize each greedy argmax and stream it to the user.
//
//   chat <model.gguf> [--raw] [--ctx N] [--max N] [-K N] [--naive] [--show-prompt]
//     default : instruct chat (OLMoE template), multi-turn, KV persists across turns.
//     --raw   : no template — your text is fed verbatim and the model continues it.
//     -K N    : expert slots per layer (cache mode). Bigger K = higher hit rate, more VRAM.
//     --naive : re-copy the 8 selected experts every token (0% cache) instead of LRU cache.
//
// Engine constants, structs, load_model, and the per-token forward graphs are copied
// faithfully from src/run_moe_stream.cpp (the M1-M3 validated engine) so behaviour is identical.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"
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
#include <iostream>

#ifndef GGML_ROPE_TYPE_NEOX
#define GGML_ROPE_TYPE_NEOX 2
#endif
static const int n_layer=16,n_embd=2048,n_head=16,head_dim=128,n_ff=1024;
static const int n_expert=64,n_used=8,n_vocab=50304,n_ctx_orig=4096;
static const float eps=1e-5f,freq_base=10000.0f;

// ---------------------------------------------------------------------------
// Engine state + loader (verbatim from run_moe_stream.cpp)
// ---------------------------------------------------------------------------
struct Model {
    gguf_context* gguf=nullptr; ggml_context* ctx=nullptr;
    ggml_backend_t backend=nullptr; ggml_backend_buffer_t wbuf=nullptr;
    std::map<std::string, ggml_tensor*> ten;
    ggml_tensor* get(const std::string& n){ auto it=ten.find(n); if(it==ten.end()){fprintf(stderr,"MISSING %s\n",n.c_str());exit(1);} return it->second; }
    ggml_tensor* blk(int il,const char* s){ char b[80]; snprintf(b,sizeof b,"blk.%d.%s",il,s); return get(b); }
};
struct HostExperts {
    std::vector<std::vector<uint8_t>> gate,up,down;
    std::vector<size_t> gstride,ustride,dstride;
    std::vector<ggml_type> gtype,utype,dtype;   // PER LAYER (Q4_K_M mixes q4_K/q6_K across layers)
};
static bool is_exp(const char* n){ return strstr(n,"_exps")!=nullptr; }

static void load_model(Model& M, HostExperts& H, const char* path, ggml_backend_t backend) {
    M.backend=backend;
    H.gate.resize(n_layer); H.up.resize(n_layer); H.down.resize(n_layer);
    H.gstride.resize(n_layer); H.ustride.resize(n_layer); H.dstride.resize(n_layer);
    H.gtype.resize(n_layer); H.utype.resize(n_layer); H.dtype.resize(n_layer);
    ggml_context* all=nullptr;
    gguf_context* gguf=gguf_init_from_file(path,{true,&all});
    if(!gguf){fprintf(stderr,"gguf fail\n");exit(1);}

    ggml_context* nectx=ggml_init({(size_t)(n_layer*12+16)*ggml_tensor_overhead(),nullptr,true});
    std::map<std::string,ggml_tensor*> nesrc;
    for(ggml_tensor* t=ggml_get_first_tensor(all); t; t=ggml_get_next_tensor(all,t)){
        if(is_exp(t->name)) continue;
        ggml_tensor* c=ggml_new_tensor(nectx,t->type,GGML_MAX_DIMS,t->ne); ggml_set_name(c,t->name); nesrc[t->name]=c;
    }
    M.wbuf=ggml_backend_alloc_ctx_tensors(nectx,backend);

    FILE* f=fopen(path,"rb"); size_t doff=gguf_get_data_offset(gguf); std::vector<uint8_t> buf;
    for(ggml_tensor* t=ggml_get_first_tensor(all); t; t=ggml_get_next_tensor(all,t)){
        int64_t ti=gguf_find_tensor(gguf,t->name); size_t off=gguf_get_tensor_offset(gguf,ti), nb=ggml_nbytes(t);
        buf.resize(nb); _fseeki64(f,(int64_t)(doff+off),SEEK_SET);
        if(fread(buf.data(),1,nb,f)!=nb){fprintf(stderr,"read fail %s\n",t->name);exit(1);}
        if(is_exp(t->name)){
            int il=-1; char which[16]={0}; sscanf(t->name,"blk.%d.ffn_%15[^_]_exps",&il,which);
            if(!strcmp(which,"gate")){ H.gate[il]=buf; H.gstride[il]=nb/n_expert; H.gtype[il]=t->type; }
            else if(!strcmp(which,"up")){ H.up[il]=buf; H.ustride[il]=nb/n_expert; H.utype[il]=t->type; }
            else if(!strcmp(which,"down")){ H.down[il]=buf; H.dstride[il]=nb/n_expert; H.dtype[il]=t->type; }
        } else { ggml_backend_tensor_set(nesrc[t->name],buf.data(),0,nb); M.ten[t->name]=nesrc[t->name]; }
    }
    fclose(f); M.ctx=nectx; M.gguf=gguf;
    fprintf(stderr,"[engine] %zu non-expert VRAM tensors; experts in host RAM (gstride=%zu)\n",M.ten.size(),H.gstride[0]);
}

struct KV { std::vector<ggml_tensor*> k,v; ggml_context* ctx=nullptr; ggml_backend_buffer_t buf=nullptr; };
struct Slots {
    int K;
    std::vector<std::vector<ggml_tensor*>> gate,up,down;
    ggml_context* ctx=nullptr; ggml_backend_buffer_t buf=nullptr;
    std::vector<std::vector<int>> slot_expert;      // [layer][slot] -> expert id
    std::vector<std::map<int,int>> expert_slot;     // [layer] expert -> slot
    std::vector<std::list<int>> lru;                // [layer] slots, front=most recent
};

// ---------------------------------------------------------------------------
// Engine: holds all decode state; step() runs one token and returns logits.
// ---------------------------------------------------------------------------
struct Engine {
    Model M; HostExperts H; KV kv; Slots S;
    ggml_gallocr_t galloc=nullptr; ggml_backend_t backend=nullptr;
    std::string mode="cache"; int K=32;
    long copies=0,hits=0,reqs=0; size_t peak_compute=0;
    int max_kv=0;

    int ensure(int il,int e){               // LRU residency: returns slot holding expert e (loads on miss)
        reqs++;
        if(mode=="naive") return -1;
        auto it=S.expert_slot[il].find(e);
        if(it!=S.expert_slot[il].end()){ hits++; S.lru[il].remove(it->second); S.lru[il].push_front(it->second); return it->second; }
        int slot=S.lru[il].back(); S.lru[il].pop_back();
        int old=S.slot_expert[il][slot]; if(old>=0) S.expert_slot[il].erase(old);
        ggml_backend_tensor_set(S.gate[il][slot], H.gate[il].data()+(size_t)e*H.gstride[il],0,H.gstride[il]);
        ggml_backend_tensor_set(S.up[il][slot],   H.up[il].data()+(size_t)e*H.ustride[il],0,H.ustride[il]);
        ggml_backend_tensor_set(S.down[il][slot], H.down[il].data()+(size_t)e*H.dstride[il],0,H.dstride[il]);
        copies++; S.slot_expert[il][slot]=e; S.expert_slot[il][e]=slot; S.lru[il].push_front(slot);
        return slot;
    }

    // one decode step at position p; fills logits[n_vocab].
    void step(int32_t tok,int p,std::vector<float>& logits){
        std::vector<float> x(n_embd);
        // ---- graph E: embedding ----
        {
            ggml_context* g=ggml_init({(size_t)8*1024*1024,nullptr,true});
            ggml_cgraph* gf=ggml_new_graph(g);
            ggml_tensor* it=ggml_new_tensor_1d(g,GGML_TYPE_I32,1); ggml_set_input(it);
            ggml_tensor* e=ggml_get_rows(g,M.get("token_embd.weight"),it); ggml_set_output(e);
            ggml_build_forward_expand(gf,e); ggml_gallocr_alloc_graph(galloc,gf);
            ggml_backend_tensor_set(it,&tok,0,4); ggml_backend_graph_compute(backend,gf);
            ggml_backend_tensor_get(e,x.data(),0,(size_t)n_embd*4); ggml_free(g);
        }
        for(int il=0; il<n_layer; ++il){
            std::vector<float> ffnx(n_embd), ffninp(n_embd), wsel(n_used);
            std::vector<int32_t> sel(n_used);
            // ---- graph A: attention (KV cache) + router ----
            {
                ggml_context* g=ggml_init({(size_t)128*1024*1024,nullptr,true});
                ggml_cgraph* gf=ggml_new_graph_custom(g,4096,false);
                ggml_tensor* ix=ggml_new_tensor_2d(g,GGML_TYPE_F32,n_embd,1); ggml_set_input(ix);
                ggml_tensor* ip=ggml_new_tensor_1d(g,GGML_TYPE_I32,1); ggml_set_input(ip);
                ggml_tensor* cur=ggml_mul(g,ggml_rms_norm(g,ix,eps),M.blk(il,"attn_norm.weight"));
                ggml_tensor* Qc=ggml_mul_mat(g,M.blk(il,"attn_q.weight"),cur);
                ggml_tensor* Kc=ggml_mul_mat(g,M.blk(il,"attn_k.weight"),cur);
                ggml_tensor* Vc=ggml_mul_mat(g,M.blk(il,"attn_v.weight"),cur);
                Qc=ggml_mul(g,ggml_rms_norm(g,Qc,eps),M.blk(il,"attn_q_norm.weight"));
                Kc=ggml_mul(g,ggml_rms_norm(g,Kc,eps),M.blk(il,"attn_k_norm.weight"));
                Qc=ggml_reshape_3d(g,Qc,head_dim,n_head,1); Kc=ggml_reshape_3d(g,Kc,head_dim,n_head,1); Vc=ggml_reshape_3d(g,Vc,head_dim,n_head,1);
                Qc=ggml_rope_ext(g,Qc,ip,nullptr,head_dim,GGML_ROPE_TYPE_NEOX,n_ctx_orig,freq_base,1,0,1,32,1);
                Kc=ggml_rope_ext(g,Kc,ip,nullptr,head_dim,GGML_ROPE_TYPE_NEOX,n_ctx_orig,freq_base,1,0,1,32,1);
                ggml_tensor* kc=kv.k[il]; ggml_tensor* vc=kv.v[il];
                ggml_tensor* dk=ggml_view_3d(g,kc,head_dim,n_head,1,kc->nb[1],kc->nb[2],(size_t)p*kc->nb[2]);
                ggml_tensor* dv=ggml_view_3d(g,vc,head_dim,n_head,1,vc->nb[1],vc->nb[2],(size_t)p*vc->nb[2]);
                ggml_tensor* sk=ggml_cpy(g,Kc,dk); ggml_tensor* sv=ggml_cpy(g,Vc,dv);
                ggml_tensor *Kall,*Vall;
                if(p==0){Kall=Kc;Vall=Vc;} else {
                    Kall=ggml_concat(g,ggml_view_3d(g,kc,head_dim,n_head,p,kc->nb[1],kc->nb[2],0),Kc,2);
                    Vall=ggml_concat(g,ggml_view_3d(g,vc,head_dim,n_head,p,vc->nb[1],vc->nb[2],0),Vc,2);
                }
                ggml_tensor* q=ggml_permute(g,Qc,0,2,1,3);
                ggml_tensor* k=ggml_permute(g,Kall,0,2,1,3);
                ggml_tensor* kq=ggml_soft_max_ext(g,ggml_mul_mat(g,k,q),nullptr,1.0f/sqrtf((float)head_dim),0.0f);
                ggml_tensor* vv=ggml_cont(g,ggml_permute(g,Vall,1,2,0,3));
                ggml_tensor* kqv=ggml_permute(g,ggml_mul_mat(g,vv,kq),0,2,1,3);
                cur=ggml_mul_mat(g,M.blk(il,"attn_output.weight"),ggml_cont_2d(g,kqv,n_embd,1));
                ggml_tensor* ffn_inp=ggml_add(g,cur,ix);
                ggml_tensor* fx=ggml_mul(g,ggml_rms_norm(g,ffn_inp,eps),M.blk(il,"ffn_norm.weight"));
                ggml_tensor* rlog=ggml_mul_mat(g,M.blk(il,"ffn_gate_inp.weight"),fx);
                ggml_tensor* probs=ggml_soft_max(g,rlog);
                ggml_tensor* selT=ggml_top_k(g,probs,n_used);
                ggml_tensor* wT=ggml_get_rows(g,ggml_reshape_3d(g,probs,1,n_expert,1),selT);
                ggml_set_output(fx); ggml_set_output(ffn_inp); ggml_set_output(selT); ggml_set_output(wT);
                ggml_build_forward_expand(gf,fx); ggml_build_forward_expand(gf,ffn_inp);
                ggml_build_forward_expand(gf,selT); ggml_build_forward_expand(gf,wT);
                ggml_build_forward_expand(gf,sk); ggml_build_forward_expand(gf,sv);
                ggml_gallocr_alloc_graph(galloc,gf);
                ggml_backend_tensor_set(ix,x.data(),0,(size_t)n_embd*4);
                int32_t pp=p; ggml_backend_tensor_set(ip,&pp,0,4);
                ggml_backend_graph_compute(backend,gf);
                ggml_backend_tensor_get(fx,ffnx.data(),0,(size_t)n_embd*4);
                ggml_backend_tensor_get(ffn_inp,ffninp.data(),0,(size_t)n_embd*4);
                ggml_backend_tensor_get(selT,sel.data(),0,(size_t)n_used*4);
                ggml_backend_tensor_get(wT,wsel.data(),0,(size_t)n_used*4);
                size_t cb=ggml_gallocr_get_buffer_size(galloc,0); if(cb>peak_compute)peak_compute=cb;
                ggml_free(g);
            }
            // ---- stream the 8 selected experts into slots ----
            int slotidx[n_used];
            if(mode=="naive"){
                for(int j=0;j<n_used;++j){ int e=sel[j];
                    ggml_backend_tensor_set(S.gate[il][j],H.gate[il].data()+(size_t)e*H.gstride[il],0,H.gstride[il]);
                    ggml_backend_tensor_set(S.up[il][j],  H.up[il].data()+(size_t)e*H.ustride[il],0,H.ustride[il]);
                    ggml_backend_tensor_set(S.down[il][j],H.down[il].data()+(size_t)e*H.dstride[il],0,H.dstride[il]);
                    copies++; reqs++; slotidx[j]=j;
                }
            } else {
                for(int j=0;j<n_used;++j) slotidx[j]=ensure(il,sel[j]);
            }
            // ---- graph B: expert FFN over the 8 slots ----
            std::vector<float> eout((size_t)n_embd*n_used);
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
                ggml_gallocr_alloc_graph(galloc,gf);
                ggml_backend_tensor_set(ifx,ffnx.data(),0,(size_t)n_embd*4);
                ggml_backend_graph_compute(backend,gf);
                ggml_backend_tensor_get(stack,eout.data(),0,(size_t)n_embd*n_used*4);
                ggml_free(g);
            }
            for(int d=0;d<n_embd;++d){ float acc=0; for(int j=0;j<n_used;++j) acc+=wsel[j]*eout[(size_t)j*n_embd+d]; x[d]=acc+ffninp[d]; }
        }
        // ---- head ----
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
    }
};

// ---------------------------------------------------------------------------
// Tokenizer helpers (llama.dll) — same path as ref_gen.cpp
// ---------------------------------------------------------------------------
static std::vector<int32_t> tokenize(const llama_vocab* v,const std::string& s,bool add_special){
    if(s.empty()) return {};
    int need = llama_tokenize(v,s.data(),(int)s.size(),nullptr,0,add_special,/*parse_special=*/false);
    std::vector<int32_t> t(need<0?-need:need);
    llama_tokenize(v,s.data(),(int)s.size(),t.data(),(int)t.size(),add_special,false);
    return t;
}
static std::string piece(const llama_vocab* v,int32_t tok){
    char buf[256]; int n=llama_token_to_piece(v,tok,buf,sizeof buf,0,/*special=*/false);
    if(n>=0) return std::string(buf,n);
    std::string s(-n,'\0'); llama_token_to_piece(v,tok,&s[0],(int)s.size(),0,false); return s;
}
static int argmax(const std::vector<float>& l){ int bi=0; float bv=l[0]; for(int v=1;v<(int)l.size();++v) if(l[v]>bv){bv=l[v];bi=v;} return bi; }

int main(int argc,char**argv){
    if(argc<2){ fprintf(stderr,"usage: %s <model.gguf> [--raw] [--ctx N] [--max N] [-K N] [--naive] [--show-prompt]\n",argv[0]); return 2; }
    const char* model_path=argv[1];
    bool raw=false, show_prompt=false; int n_ctx=2048, max_gen=512, K=32; std::string mode="cache";
    for(int i=2;i<argc;++i){
        std::string a=argv[i];
        if(a=="--raw") raw=true;
        else if(a=="--naive"){ mode="naive"; K=n_used; }
        else if(a=="--show-prompt") show_prompt=true;
        else if(a=="--ctx"&&i+1<argc) n_ctx=atoi(argv[++i]);
        else if(a=="--max"&&i+1<argc) max_gen=atoi(argv[++i]);
        else if(a=="-K"&&i+1<argc) K=atoi(argv[++i]);
        else { fprintf(stderr,"unknown arg: %s\n",a.c_str()); return 2; }
    }

    ggml_backend_load_all();
    ggml_backend_dev_t vk0=nullptr;
    for(size_t i=0;i<ggml_backend_dev_count();++i){ ggml_backend_dev_t d=ggml_backend_dev_get(i); if(!strcmp(ggml_backend_dev_name(d),"Vulkan0")){vk0=d;break;} }
    if(!vk0){ fprintf(stderr,"Vulkan0 not found\n"); return 1; }
    ggml_backend_t backend=ggml_backend_dev_init(vk0,nullptr);

    // ---- tokenizer (vocab only — no weights, fast/cheap) ----
    llama_model_params mp=llama_model_default_params(); mp.vocab_only=true;
    llama_model* lm=llama_model_load_from_file(model_path,mp);
    if(!lm){ fprintf(stderr,"tokenizer load failed\n"); return 1; }
    const llama_vocab* vocab=llama_model_get_vocab(lm);
    const llama_token BOS=llama_vocab_bos(vocab);   // 50279 = <|endoftext|>

    // ---- engine (weights) ----
    Engine E; E.backend=backend; E.mode=mode; E.K=K;
    load_model(E.M,E.H,model_path,backend);

    // KV cache sized to the full context window
    E.max_kv=n_ctx;
    E.kv.ctx=ggml_init({(size_t)2*n_layer*ggml_tensor_overhead()+1024,nullptr,true});
    for(int il=0;il<n_layer;++il){ E.kv.k.push_back(ggml_new_tensor_3d(E.kv.ctx,GGML_TYPE_F32,head_dim,n_head,n_ctx)); E.kv.v.push_back(ggml_new_tensor_3d(E.kv.ctx,GGML_TYPE_F32,head_dim,n_head,n_ctx)); }
    E.kv.buf=ggml_backend_alloc_ctx_tensors(E.kv.ctx,backend);

    // per-layer expert slot pool
    E.S.K=K;
    E.S.ctx=ggml_init({(size_t)(n_layer*K*3+16)*ggml_tensor_overhead(),nullptr,true});
    E.S.gate.resize(n_layer); E.S.up.resize(n_layer); E.S.down.resize(n_layer);
    E.S.slot_expert.assign(n_layer,std::vector<int>(K,-1));
    E.S.expert_slot.resize(n_layer); E.S.lru.resize(n_layer);
    for(int il=0;il<n_layer;++il) for(int s=0;s<K;++s){
        E.S.gate[il].push_back(ggml_new_tensor_2d(E.S.ctx,E.H.gtype[il],n_embd,n_ff));
        E.S.up[il].push_back(ggml_new_tensor_2d(E.S.ctx,E.H.utype[il],n_embd,n_ff));
        E.S.down[il].push_back(ggml_new_tensor_2d(E.S.ctx,E.H.dtype[il],n_ff,n_embd));
        E.S.lru[il].push_back(s);
    }
    E.S.buf=ggml_backend_alloc_ctx_tensors(E.S.ctx,backend);
    E.galloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));

    size_t vram_static = ggml_backend_buffer_get_size(E.M.wbuf)+ggml_backend_buffer_get_size(E.kv.buf)+ggml_backend_buffer_get_size(E.S.buf);
    fprintf(stderr,"[engine] mode=%s K=%d ctx=%d  static VRAM=%.1f MB (model on disk 4212 MB)\n",
            mode.c_str(),K,n_ctx,vram_static/1e6);
    fprintf(stderr,"[chat] %s mode. Type a message and press Enter. Ctrl-D (or /quit) to exit.\n\n",
            raw?"RAW completion":"instruct");

    std::vector<float> logits(n_vocab);
    int pos=0; bool first=true;

    std::string line;
    while(true){
        fputs(raw?"... ":"> ",stdout); fflush(stdout);
        if(!std::getline(std::cin,line)) break;          // Ctrl-D / EOF
        // sanitize: strip a leading UTF-8 BOM and a trailing CR (Windows CRLF / piped stdin)
        if(line.size()>=3 && (unsigned char)line[0]==0xEF && (unsigned char)line[1]==0xBB && (unsigned char)line[2]==0xBF) line.erase(0,3);
        if(!line.empty() && line.back()=='\r') line.pop_back();
        if(!raw && (line=="/quit"||line=="/exit")) break;
        if(line.empty()) continue;

        // ---- build the token delta for this turn ----
        std::vector<int32_t> in;
        if(raw){
            in = tokenize(vocab,line,/*add_special=*/first);   // first turn may add configured specials (none for OLMoE)
        } else {
            if(first && BOS>=0) in.push_back(BOS);               // <|endoftext|> per the chat template
            std::vector<int32_t> p = tokenize(vocab,"<|user|>\n"+line+"<|assistant|>",false);
            in.insert(in.end(),p.begin(),p.end());
        }
        first=false;

        if(show_prompt){ fprintf(stderr,"[prompt ids]"); for(int32_t t:in) fprintf(stderr," %d",t); fprintf(stderr,"\n"); }
        if(pos+(int)in.size() >= n_ctx-1){ fprintf(stderr,"\n[context full: %d/%d tokens — restart the program with a larger --ctx]\n",pos,n_ctx); break; }

        // ---- prefill: feed the prompt tokens; keep logits after the last ----
        for(size_t i=0;i<in.size();++i) E.step(in[i],pos++,logits);

        // ---- decode: greedy, stream detokenized text ----
        struct timespec t0{},t1{}; clock_gettime(CLOCK_MONOTONIC,&t0);
        long copies0=E.copies, hits0=E.hits, reqs0=E.reqs;
        int gen=0; bool started=false;
        int next=argmax(logits);
        while(gen<max_gen && pos<n_ctx-1){
            if(llama_vocab_is_eog(vocab,next)) break;
            std::string s=piece(vocab,next);
            if(!started){ size_t b=s.find_first_not_of("\n"); s = (b==std::string::npos)? std::string() : s.substr(b); started=true; } // trim the model's leading newline
            fputs(s.c_str(),stdout); fflush(stdout);
            E.step(next,pos++,logits); gen++;
            next=argmax(logits);
        }
        if(llama_vocab_is_eog(vocab,next) && pos<n_ctx-1) E.step(next,pos++,logits); // keep <eos> in context for the next turn
        putchar('\n');

        clock_gettime(CLOCK_MONOTONIC,&t1);
        double secs=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
        long dc=E.copies-copies0, dr=E.reqs-reqs0, dh=E.hits-hits0;
        fprintf(stderr,"[%d tok, %.1f tok/s, hit %.0f%% (%ld/%ld copies this turn), ctx %d/%d]\n\n",
                gen, gen/ (secs>0?secs:1e-9), dr? 100.0*dh/dr : 0.0, dc, dr, pos, n_ctx);
    }

    fprintf(stderr,"\n[chat] bye.\n");
    return 0;
}
