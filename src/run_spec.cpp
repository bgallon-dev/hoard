// M4: residency-aware speculation (measurement).
// Draft = same model, TOP-1 routing restricted to RESIDENT (cached) experts: zero streaming
// copies, ~1/8 the expert work. We measure, on the same streaming engine as M2/M3:
//   - acceptance: per position (given the target's verified context), does the residency-biased
//     top-1 draft predict the same next token as the top-8 target?
//   - target vs draft decode tok/s (the speed differential).
//   - projected joint tok/s from the standard greedy spec-decode model.
//   run_spec <model> <ids_csv> <ngen> <K>
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
#include <algorithm>
#ifndef GGML_ROPE_TYPE_NEOX
#define GGML_ROPE_TYPE_NEOX 2
#endif
static const int n_layer=16,n_embd=2048,n_head=16,head_dim=128,n_ff=1024;
static const int n_expert=64,n_vocab=50304,n_ctx_orig=4096;
static const float eps=1e-5f,freq_base=10000.0f;

struct Model{ggml_context*ctx=nullptr;ggml_backend_t backend=nullptr;ggml_backend_buffer_t wbuf=nullptr;std::map<std::string,ggml_tensor*>ten;
    ggml_tensor*get(const std::string&n){auto it=ten.find(n);if(it==ten.end()){fprintf(stderr,"MISSING %s\n",n.c_str());exit(1);}return it->second;}
    ggml_tensor*blk(int il,const char*s){char b[80];snprintf(b,sizeof b,"blk.%d.%s",il,s);return get(b);} };
struct HostExperts{std::vector<std::vector<uint8_t>>gate,up,down;std::vector<size_t>gstride,ustride,dstride;std::vector<ggml_type>gtype,utype,dtype;};
static bool is_exp(const char*n){return strstr(n,"_exps")!=nullptr;}
static void load_model(Model&M,HostExperts&H,const char*path,ggml_backend_t backend){
    M.backend=backend;H.gate.resize(n_layer);H.up.resize(n_layer);H.down.resize(n_layer);
    H.gstride.resize(n_layer);H.ustride.resize(n_layer);H.dstride.resize(n_layer);H.gtype.resize(n_layer);H.utype.resize(n_layer);H.dtype.resize(n_layer);
    ggml_context*all=nullptr;gguf_context*gguf=gguf_init_from_file(path,{true,&all});
    ggml_context*nectx=ggml_init({(size_t)(n_layer*12+16)*ggml_tensor_overhead(),nullptr,true});std::map<std::string,ggml_tensor*>nesrc;
    for(ggml_tensor*t=ggml_get_first_tensor(all);t;t=ggml_get_next_tensor(all,t)){if(is_exp(t->name))continue;ggml_tensor*c=ggml_new_tensor(nectx,t->type,GGML_MAX_DIMS,t->ne);ggml_set_name(c,t->name);nesrc[t->name]=c;}
    M.wbuf=ggml_backend_alloc_ctx_tensors(nectx,backend);
    FILE*f=fopen(path,"rb");size_t doff=gguf_get_data_offset(gguf);std::vector<uint8_t>buf;
    for(ggml_tensor*t=ggml_get_first_tensor(all);t;t=ggml_get_next_tensor(all,t)){
        int64_t ti=gguf_find_tensor(gguf,t->name);size_t off=gguf_get_tensor_offset(gguf,ti),nb=ggml_nbytes(t);buf.resize(nb);_fseeki64(f,(int64_t)(doff+off),SEEK_SET);fread(buf.data(),1,nb,f);
        if(is_exp(t->name)){int il=-1;char w[16]={0};sscanf(t->name,"blk.%d.ffn_%15[^_]_exps",&il,w);
            if(!strcmp(w,"gate")){H.gate[il]=buf;H.gstride[il]=nb/n_expert;H.gtype[il]=t->type;}else if(!strcmp(w,"up")){H.up[il]=buf;H.ustride[il]=nb/n_expert;H.utype[il]=t->type;}else if(!strcmp(w,"down")){H.down[il]=buf;H.dstride[il]=nb/n_expert;H.dtype[il]=t->type;}
        }else{ggml_backend_tensor_set(nesrc[t->name],buf.data(),0,nb);M.ten[t->name]=nesrc[t->name];}
    }
    fclose(f);M.ctx=nectx;
}
struct KV{std::vector<ggml_tensor*>k,v;ggml_context*ctx=nullptr;ggml_backend_buffer_t buf=nullptr;};
struct Slots{int K;std::vector<std::vector<ggml_tensor*>>gate,up,down;ggml_context*ctx=nullptr;ggml_backend_buffer_t buf=nullptr;
    std::vector<std::vector<int>>slot_expert;std::vector<std::map<int,int>>expert_slot;std::vector<std::list<int>>lru;};

static Model M; static HostExperts H; static KV kv; static Slots S; static ggml_gallocr_t galloc;
static long g_copies=0,g_hits=0,g_reqs=0;

static int ensure(int il,int e){ // returns slot holding expert e (LRU load on miss)
    g_reqs++; auto it=S.expert_slot[il].find(e);
    if(it!=S.expert_slot[il].end()){g_hits++;S.lru[il].remove(it->second);S.lru[il].push_front(it->second);return it->second;}
    int slot=S.lru[il].back();S.lru[il].pop_back();int old=S.slot_expert[il][slot];if(old>=0)S.expert_slot[il].erase(old);
    ggml_backend_tensor_set(S.gate[il][slot],H.gate[il].data()+(size_t)e*H.gstride[il],0,H.gstride[il]);
    ggml_backend_tensor_set(S.up[il][slot],H.up[il].data()+(size_t)e*H.ustride[il],0,H.ustride[il]);
    ggml_backend_tensor_set(S.down[il][slot],H.down[il].data()+(size_t)e*H.dstride[il],0,H.dstride[il]);
    g_copies++;S.slot_expert[il][slot]=e;S.expert_slot[il][e]=slot;S.lru[il].push_front(slot);return slot;
}

// One decode forward. n_use experts/layer. resident_only => pick top n_use among CACHED experts
// (draft: zero copies). store => write K/V to cache (target only). Returns logits[n_vocab].
static std::vector<float> forward(int32_t tok,int p,int n_use,bool resident_only,bool store){
    std::vector<float> x(n_embd);
    { ggml_context*g=ggml_init({(size_t)8*1024*1024,nullptr,true});ggml_cgraph*gf=ggml_new_graph(g);
      ggml_tensor*it=ggml_new_tensor_1d(g,GGML_TYPE_I32,1);ggml_set_input(it);
      ggml_tensor*e=ggml_get_rows(g,M.get("token_embd.weight"),it);ggml_set_output(e);ggml_build_forward_expand(gf,e);ggml_gallocr_alloc_graph(galloc,gf);
      ggml_backend_tensor_set(it,&tok,0,4);ggml_backend_graph_compute(M.backend,gf);ggml_backend_tensor_get(e,x.data(),0,(size_t)n_embd*4);ggml_free(g);}
    for(int il=0;il<n_layer;++il){
        std::vector<float> ffnx(n_embd),ffninp(n_embd),probs(n_expert);
        { ggml_context*g=ggml_init({(size_t)128*1024*1024,nullptr,true});ggml_cgraph*gf=ggml_new_graph_custom(g,4096,false);
          ggml_tensor*ix=ggml_new_tensor_2d(g,GGML_TYPE_F32,n_embd,1);ggml_set_input(ix);
          ggml_tensor*ip=ggml_new_tensor_1d(g,GGML_TYPE_I32,1);ggml_set_input(ip);
          ggml_tensor*cur=ggml_mul(g,ggml_rms_norm(g,ix,eps),M.blk(il,"attn_norm.weight"));
          ggml_tensor*Qc=ggml_mul_mat(g,M.blk(il,"attn_q.weight"),cur),*Kc=ggml_mul_mat(g,M.blk(il,"attn_k.weight"),cur),*Vc=ggml_mul_mat(g,M.blk(il,"attn_v.weight"),cur);
          Qc=ggml_mul(g,ggml_rms_norm(g,Qc,eps),M.blk(il,"attn_q_norm.weight"));Kc=ggml_mul(g,ggml_rms_norm(g,Kc,eps),M.blk(il,"attn_k_norm.weight"));
          Qc=ggml_reshape_3d(g,Qc,head_dim,n_head,1);Kc=ggml_reshape_3d(g,Kc,head_dim,n_head,1);Vc=ggml_reshape_3d(g,Vc,head_dim,n_head,1);
          Qc=ggml_rope_ext(g,Qc,ip,nullptr,head_dim,GGML_ROPE_TYPE_NEOX,n_ctx_orig,freq_base,1,0,1,32,1);
          Kc=ggml_rope_ext(g,Kc,ip,nullptr,head_dim,GGML_ROPE_TYPE_NEOX,n_ctx_orig,freq_base,1,0,1,32,1);
          ggml_tensor*kc=kv.k[il],*vc=kv.v[il];
          std::vector<ggml_tensor*> stores;
          if(store){ggml_tensor*dk=ggml_view_3d(g,kc,head_dim,n_head,1,kc->nb[1],kc->nb[2],(size_t)p*kc->nb[2]),*dv=ggml_view_3d(g,vc,head_dim,n_head,1,vc->nb[1],vc->nb[2],(size_t)p*vc->nb[2]);
            stores.push_back(ggml_cpy(g,Kc,dk));stores.push_back(ggml_cpy(g,Vc,dv));}
          ggml_tensor*Kall,*Vall;
          if(p==0){Kall=Kc;Vall=Vc;}else{Kall=ggml_concat(g,ggml_view_3d(g,kc,head_dim,n_head,p,kc->nb[1],kc->nb[2],0),Kc,2);Vall=ggml_concat(g,ggml_view_3d(g,vc,head_dim,n_head,p,vc->nb[1],vc->nb[2],0),Vc,2);}
          ggml_tensor*q=ggml_permute(g,Qc,0,2,1,3),*k=ggml_permute(g,Kall,0,2,1,3);
          ggml_tensor*kq=ggml_soft_max_ext(g,ggml_mul_mat(g,k,q),nullptr,1.0f/sqrtf((float)head_dim),0.0f);
          ggml_tensor*vv=ggml_cont(g,ggml_permute(g,Vall,1,2,0,3));
          ggml_tensor*kqv=ggml_permute(g,ggml_mul_mat(g,vv,kq),0,2,1,3);
          cur=ggml_mul_mat(g,M.blk(il,"attn_output.weight"),ggml_cont_2d(g,kqv,n_embd,1));
          ggml_tensor*ffn_inp=ggml_add(g,cur,ix);
          ggml_tensor*fx=ggml_mul(g,ggml_rms_norm(g,ffn_inp,eps),M.blk(il,"ffn_norm.weight"));
          ggml_tensor*pr=ggml_soft_max(g,ggml_mul_mat(g,M.blk(il,"ffn_gate_inp.weight"),fx));
          ggml_set_output(fx);ggml_set_output(ffn_inp);ggml_set_output(pr);
          ggml_build_forward_expand(gf,fx);ggml_build_forward_expand(gf,ffn_inp);ggml_build_forward_expand(gf,pr);
          for(auto s:stores)ggml_build_forward_expand(gf,s);
          ggml_gallocr_alloc_graph(galloc,gf);
          ggml_backend_tensor_set(ix,x.data(),0,(size_t)n_embd*4);int32_t pp=p;ggml_backend_tensor_set(ip,&pp,0,4);
          ggml_backend_graph_compute(M.backend,gf);
          ggml_backend_tensor_get(fx,ffnx.data(),0,(size_t)n_embd*4);ggml_backend_tensor_get(ffn_inp,ffninp.data(),0,(size_t)n_embd*4);ggml_backend_tensor_get(pr,probs.data(),0,(size_t)n_expert*4);
          ggml_free(g);}
        // select experts
        std::vector<int> sel; std::vector<float> wsel;
        if(resident_only){ // top n_use among experts currently resident in this layer
            std::vector<std::pair<float,int>> cand;
            for(int s=0;s<S.K;++s){int e=S.slot_expert[il][s];if(e>=0)cand.push_back({probs[e],e});}
            std::sort(cand.begin(),cand.end(),[](auto&a,auto&b){return a.first>b.first;});
            for(int j=0;j<n_use && j<(int)cand.size();++j){sel.push_back(cand[j].second);wsel.push_back(cand[j].first);g_reqs++;g_hits++;}
        } else {
            std::vector<std::pair<float,int>> cand(n_expert);for(int e=0;e<n_expert;++e)cand[e]={probs[e],e};
            std::partial_sort(cand.begin(),cand.begin()+n_use,cand.end(),[](auto&a,auto&b){return a.first>b.first;});
            for(int j=0;j<n_use;++j){sel.push_back(cand[j].second);wsel.push_back(cand[j].first);}
        }
        int ns=(int)sel.size();
        std::vector<int> slotidx(ns);
        for(int j=0;j<ns;++j) slotidx[j]=resident_only?S.expert_slot[il][sel[j]]:ensure(il,sel[j]);
        // expert FFN
        std::vector<float> eout((size_t)n_embd*ns);
        { ggml_context*g=ggml_init({(size_t)64*1024*1024,nullptr,true});ggml_cgraph*gf=ggml_new_graph_custom(g,2048,false);
          ggml_tensor*ifx=ggml_new_tensor_2d(g,GGML_TYPE_F32,n_embd,1);ggml_set_input(ifx);
          std::vector<ggml_tensor*>ys;
          for(int j=0;j<ns;++j){int s=slotidx[j];ggml_tensor*gj=ggml_mul_mat(g,S.gate[il][s],ifx),*uj=ggml_mul_mat(g,S.up[il][s],ifx);ys.push_back(ggml_mul_mat(g,S.down[il][s],ggml_mul(g,ggml_silu(g,gj),uj)));}
          ggml_tensor*st=ys[0];for(int j=1;j<ns;++j)st=ggml_concat(g,st,ys[j],1);ggml_set_output(st);ggml_build_forward_expand(gf,st);ggml_gallocr_alloc_graph(galloc,gf);
          ggml_backend_tensor_set(ifx,ffnx.data(),0,(size_t)n_embd*4);ggml_backend_graph_compute(M.backend,gf);ggml_backend_tensor_get(st,eout.data(),0,(size_t)n_embd*ns*4);ggml_free(g);}
        for(int d=0;d<n_embd;++d){float acc=0;for(int j=0;j<ns;++j)acc+=wsel[j]*eout[(size_t)j*n_embd+d];x[d]=acc+ffninp[d];}
    }
    std::vector<float> logits(n_vocab);
    { ggml_context*g=ggml_init({(size_t)64*1024*1024,nullptr,true});ggml_cgraph*gf=ggml_new_graph(g);
      ggml_tensor*ix=ggml_new_tensor_2d(g,GGML_TYPE_F32,n_embd,1);ggml_set_input(ix);
      ggml_tensor*lg=ggml_mul_mat(g,M.get("output.weight"),ggml_mul(g,ggml_rms_norm(g,ix,eps),M.get("output_norm.weight")));ggml_set_output(lg);
      ggml_build_forward_expand(gf,lg);ggml_gallocr_alloc_graph(galloc,gf);ggml_backend_tensor_set(ix,x.data(),0,(size_t)n_embd*4);ggml_backend_graph_compute(M.backend,gf);ggml_backend_tensor_get(lg,logits.data(),0,(size_t)n_vocab*4);ggml_free(g);}
    return logits;
}
static int argmax(const std::vector<float>&l){int bi=0;float bv=l[0];for(int v=1;v<(int)l.size();++v)if(l[v]>bv){bv=l[v];bi=v;}return bi;}

int main(int argc,char**argv){
    if(argc<5){fprintf(stderr,"usage: %s model ids_csv ngen K\n",argv[0]);return 2;}
    int ngen=atoi(argv[3]),K=atoi(argv[4]);
    std::vector<int32_t> seq;{char*s=strdup(argv[2]);for(char*p=strtok(s,",");p;p=strtok(nullptr,","))seq.push_back(atoi(p));free(s);}
    int T=(int)seq.size();
    ggml_backend_load_all();ggml_backend_dev_t vk0=nullptr;
    for(size_t i=0;i<ggml_backend_dev_count();++i){ggml_backend_dev_t d=ggml_backend_dev_get(i);if(!strcmp(ggml_backend_dev_name(d),"Vulkan0")){vk0=d;break;}}
    ggml_backend_t backend=ggml_backend_dev_init(vk0,nullptr);load_model(M,H,argv[1],backend);
    int max_kv=T+ngen+2;kv.ctx=ggml_init({(size_t)2*n_layer*ggml_tensor_overhead()+1024,nullptr,true});
    for(int il=0;il<n_layer;++il){kv.k.push_back(ggml_new_tensor_3d(kv.ctx,GGML_TYPE_F32,head_dim,n_head,max_kv));kv.v.push_back(ggml_new_tensor_3d(kv.ctx,GGML_TYPE_F32,head_dim,n_head,max_kv));}
    kv.buf=ggml_backend_alloc_ctx_tensors(kv.ctx,backend);
    S.K=K;S.ctx=ggml_init({(size_t)(n_layer*K*3+16)*ggml_tensor_overhead(),nullptr,true});S.gate.resize(n_layer);S.up.resize(n_layer);S.down.resize(n_layer);
    S.slot_expert.assign(n_layer,std::vector<int>(K,-1));S.expert_slot.resize(n_layer);S.lru.resize(n_layer);
    for(int il=0;il<n_layer;++il)for(int s=0;s<K;++s){S.gate[il].push_back(ggml_new_tensor_2d(S.ctx,H.gtype[il],n_embd,n_ff));S.up[il].push_back(ggml_new_tensor_2d(S.ctx,H.utype[il],n_embd,n_ff));S.down[il].push_back(ggml_new_tensor_2d(S.ctx,H.dtype[il],n_ff,n_embd));S.lru[il].push_back(s);}
    S.buf=ggml_backend_alloc_ctx_tensors(S.ctx,backend);
    galloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));

    long accept=0,total=0; double t_target=0,t_draft=0; int decode_steps=0;
    struct timespec a,b;
    for(int p=0;;++p){
        clock_gettime(CLOCK_MONOTONIC,&a);
        std::vector<float> tl=forward(seq[p],p,8,false,true);   // target: top-8, stream, store KV
        clock_gettime(CLOCK_MONOTONIC,&b); if(p>=T) t_target+=(b.tv_sec-a.tv_sec)+(b.tv_nsec-a.tv_nsec)/1e9;
        int tgt=argmax(tl);
        if(p>=T-1){
            // draft prediction at same position (top-1 among resident experts, no copies, no store)
            clock_gettime(CLOCK_MONOTONIC,&a);
            std::vector<float> dl=forward(seq[p],p,1,true,false);
            clock_gettime(CLOCK_MONOTONIC,&b); t_draft+=(b.tv_sec-a.tv_sec)+(b.tv_nsec-a.tv_nsec)/1e9;
            int drf=argmax(dl);
            total++; if(drf==tgt) accept++; decode_steps++;
            if((int)total>=ngen||tgt==50279) break;
            seq.push_back(tgt);
        }
    }
    double alpha=(double)accept/total;
    double tgt_s=t_target/decode_steps, drf_s=t_draft/decode_steps;
    // greedy spec-decode projection: draft proposes 1 token, target verifies.
    // expected target steps per accepted token: 1/(1+alpha) (1 verify yields 1 or 2 tokens).
    // joint time/token ~ (drf_s + tgt_s) / (1+alpha); speedup vs target-alone = tgt_s / that.
    double joint_per_tok=(drf_s+tgt_s)/(1.0+alpha);
    double speedup=tgt_s/joint_per_tok;
    printf("M4 residency-aware speculation (K=%d):\n",K);
    printf("  acceptance rate (resident top-1 draft vs top-8 target) = %.1f%%  (%ld/%ld)\n",100*alpha,accept,total);
    printf("  target tok/s = %.2f   draft tok/s = %.2f   (draft is %.1fx faster, %ld copies during draft)\n",1/tgt_s,1/drf_s,tgt_s/drf_s,(long)0);
    printf("  projected joint tok/s = %.2f  => %.2fx vs target-alone\n",1/joint_per_tok,speedup);
    return 0;
}
