// M4 VALIDATION: integrated greedy speculative decoding (batched verify + rollback), measured.
// Draft = top-1 routing among RESIDENT experts (0 copies). Verify = ONE batched forward over k
// draft tokens through the top-8 streaming target (expert loads amortized over the k tokens via
// mul_mat_id over a per-layer slot pool). Accept longest greedy-matching prefix + 1 correction.
// Reports MEASURED joint tok/s vs target-alone tok/s, real acceptance + avg accepted length.
//   run_specloop <model> <ids_csv> <ngen> <K> <k_draft>
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
static Model M; static HostExperts H;
static void load_model(const char*path,ggml_backend_t backend){
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
// KV cache
static std::vector<ggml_tensor*> KCACHE,VCACHE; static ggml_backend_buffer_t kvbuf;
// per-layer slot pool as 3D tensors [in,out,K]; residency maps
static int POOLK; static std::vector<ggml_tensor*> GPOOL,UPOOL,DPOOL; static ggml_backend_buffer_t poolbuf;
static std::vector<std::vector<int>> slot_expert; static std::vector<std::map<int,int>> expert_slot; static std::vector<std::list<int>> lru;
static ggml_gallocr_t galloc;
static long g_copies=0;

static int ensure(int il,int e){ // expert e -> slot index (load into pool slice on miss)
    auto it=expert_slot[il].find(e);
    if(it!=expert_slot[il].end()){lru[il].remove(it->second);lru[il].push_front(it->second);return it->second;}
    int s=lru[il].back();lru[il].pop_back();int old=slot_expert[il][s];if(old>=0)expert_slot[il].erase(old);
    ggml_backend_tensor_set(GPOOL[il],H.gate[il].data()+(size_t)e*H.gstride[il],(size_t)s*H.gstride[il],H.gstride[il]);
    ggml_backend_tensor_set(UPOOL[il],H.up[il].data()+(size_t)e*H.ustride[il],(size_t)s*H.ustride[il],H.ustride[il]);
    ggml_backend_tensor_set(DPOOL[il],H.down[il].data()+(size_t)e*H.dstride[il],(size_t)s*H.dstride[il],H.dstride[il]);
    g_copies++;slot_expert[il][s]=e;expert_slot[il][e]=s;lru[il].push_front(s);return s;
}

// forward over m tokens at positions base..base+m-1. n_use experts/token. resident_only: draft.
// store: write K/V to cache. Returns logits [n_vocab * m] (row-major: position-major).
static std::vector<float> forward(const std::vector<int32_t>&toks,int base,int n_use,bool resident_only,bool store){
    int m=(int)toks.size();
    std::vector<float> X((size_t)n_embd*m);
    { ggml_context*g=ggml_init({(size_t)16*1024*1024,nullptr,true});ggml_cgraph*gf=ggml_new_graph(g);
      ggml_tensor*it=ggml_new_tensor_1d(g,GGML_TYPE_I32,m);ggml_set_input(it);
      ggml_tensor*e=ggml_get_rows(g,M.get("token_embd.weight"),it);ggml_set_output(e);ggml_build_forward_expand(gf,e);ggml_gallocr_alloc_graph(galloc,gf);
      ggml_backend_tensor_set(it,toks.data(),0,(size_t)m*4);ggml_backend_graph_compute(M.backend,gf);ggml_backend_tensor_get(e,X.data(),0,(size_t)n_embd*m*4);ggml_free(g);}
    int nkv=base+m;
    for(int il=0;il<n_layer;++il){
        std::vector<float> ffnx((size_t)n_embd*m),ffninp((size_t)n_embd*m),probs((size_t)n_expert*m);
        { ggml_context*g=ggml_init({(size_t)160*1024*1024,nullptr,true});ggml_cgraph*gf=ggml_new_graph_custom(g,4096,false);
          ggml_tensor*ix=ggml_new_tensor_2d(g,GGML_TYPE_F32,n_embd,m);ggml_set_input(ix);
          ggml_tensor*ip=ggml_new_tensor_1d(g,GGML_TYPE_I32,m);ggml_set_input(ip);
          ggml_tensor*mask=nullptr; if(m>1){mask=ggml_new_tensor_2d(g,GGML_TYPE_F32,nkv,m);ggml_set_input(mask);}
          ggml_tensor*cur=ggml_mul(g,ggml_rms_norm(g,ix,eps),M.blk(il,"attn_norm.weight"));
          ggml_tensor*Qc=ggml_mul_mat(g,M.blk(il,"attn_q.weight"),cur),*Kc=ggml_mul_mat(g,M.blk(il,"attn_k.weight"),cur),*Vc=ggml_mul_mat(g,M.blk(il,"attn_v.weight"),cur);
          Qc=ggml_mul(g,ggml_rms_norm(g,Qc,eps),M.blk(il,"attn_q_norm.weight"));Kc=ggml_mul(g,ggml_rms_norm(g,Kc,eps),M.blk(il,"attn_k_norm.weight"));
          Qc=ggml_reshape_3d(g,Qc,head_dim,n_head,m);Kc=ggml_reshape_3d(g,Kc,head_dim,n_head,m);Vc=ggml_reshape_3d(g,Vc,head_dim,n_head,m);
          Qc=ggml_rope_ext(g,Qc,ip,nullptr,head_dim,GGML_ROPE_TYPE_NEOX,n_ctx_orig,freq_base,1,0,1,32,1);
          Kc=ggml_rope_ext(g,Kc,ip,nullptr,head_dim,GGML_ROPE_TYPE_NEOX,n_ctx_orig,freq_base,1,0,1,32,1);
          ggml_tensor*kc=KCACHE[il],*vc=VCACHE[il];
          std::vector<ggml_tensor*> stores;
          if(store){ggml_tensor*dk=ggml_view_3d(g,kc,head_dim,n_head,m,kc->nb[1],kc->nb[2],(size_t)base*kc->nb[2]),*dv=ggml_view_3d(g,vc,head_dim,n_head,m,vc->nb[1],vc->nb[2],(size_t)base*vc->nb[2]);
            stores.push_back(ggml_cpy(g,Kc,dk));stores.push_back(ggml_cpy(g,Vc,dv));}
          ggml_tensor*Kall,*Vall;
          if(base==0){Kall=Kc;Vall=Vc;}else{Kall=ggml_concat(g,ggml_view_3d(g,kc,head_dim,n_head,base,kc->nb[1],kc->nb[2],0),Kc,2);Vall=ggml_concat(g,ggml_view_3d(g,vc,head_dim,n_head,base,vc->nb[1],vc->nb[2],0),Vc,2);}
          ggml_tensor*q=ggml_permute(g,Qc,0,2,1,3),*k=ggml_permute(g,Kall,0,2,1,3);
          ggml_tensor*kq=ggml_mul_mat(g,k,q); // [nkv,m,heads]
          kq=ggml_soft_max_ext(g,kq,mask,1.0f/sqrtf((float)head_dim),0.0f);
          ggml_tensor*vv=ggml_cont(g,ggml_permute(g,Vall,1,2,0,3));
          ggml_tensor*kqv=ggml_permute(g,ggml_mul_mat(g,vv,kq),0,2,1,3);
          cur=ggml_mul_mat(g,M.blk(il,"attn_output.weight"),ggml_cont_2d(g,kqv,n_embd,m));
          ggml_tensor*ffn_inp=ggml_add(g,cur,ix);
          ggml_tensor*fx=ggml_mul(g,ggml_rms_norm(g,ffn_inp,eps),M.blk(il,"ffn_norm.weight"));
          ggml_tensor*pr=ggml_soft_max(g,ggml_mul_mat(g,M.blk(il,"ffn_gate_inp.weight"),fx));
          ggml_set_output(fx);ggml_set_output(ffn_inp);ggml_set_output(pr);
          ggml_build_forward_expand(gf,fx);ggml_build_forward_expand(gf,ffn_inp);ggml_build_forward_expand(gf,pr);for(auto s:stores)ggml_build_forward_expand(gf,s);
          ggml_gallocr_alloc_graph(galloc,gf);
          ggml_backend_tensor_set(ix,X.data(),0,(size_t)n_embd*m*4);
          std::vector<int32_t>pos(m);for(int i=0;i<m;++i)pos[i]=base+i;ggml_backend_tensor_set(ip,pos.data(),0,(size_t)m*4);
          if(m>1){std::vector<float>mk((size_t)nkv*m);for(int qi=0;qi<m;++qi)for(int kvi=0;kvi<nkv;++kvi)mk[(size_t)qi*nkv+kvi]=(kvi<base||kvi-base<=qi)?0.0f:-INFINITY;ggml_backend_tensor_set(mask,mk.data(),0,(size_t)nkv*m*4);}
          ggml_backend_graph_compute(M.backend,gf);
          ggml_backend_tensor_get(fx,ffnx.data(),0,(size_t)n_embd*m*4);ggml_backend_tensor_get(ffn_inp,ffninp.data(),0,(size_t)n_embd*m*4);ggml_backend_tensor_get(pr,probs.data(),0,(size_t)n_expert*m*4);
          ggml_free(g);}
        // select experts per token + ensure resident in pool; build ids/weights [n_use, m]
        std::vector<int32_t> ids((size_t)n_use*m); std::vector<float> wgt((size_t)n_use*m);
        for(int t=0;t<m;++t){
            std::vector<std::pair<float,int>> cand;
            if(resident_only){for(int s=0;s<POOLK;++s){int e=slot_expert[il][s];if(e>=0)cand.push_back({probs[(size_t)t*n_expert+e],e});}}
            else{cand.resize(n_expert);for(int e=0;e<n_expert;++e)cand[e]={probs[(size_t)t*n_expert+e],e};}
            int nu=std::min((int)cand.size(),n_use);
            std::partial_sort(cand.begin(),cand.begin()+nu,cand.end(),[](auto&a,auto&b){return a.first>b.first;});
            for(int j=0;j<n_use;++j){ if(j<nu){int e=cand[j].second;int s=resident_only?expert_slot[il][e]:ensure(il,e);ids[(size_t)j*m+t]=s;wgt[(size_t)j*m+t]=cand[j].first;} else {ids[(size_t)j*m+t]=ids[t];wgt[(size_t)j*m+t]=0;} }
        }
        // expert FFN via mul_mat_id over pool, weighted sum -> moe[n_embd,m]
        std::vector<float> moe((size_t)n_embd*m);
        { ggml_context*g=ggml_init({(size_t)160*1024*1024,nullptr,true});ggml_cgraph*gf=ggml_new_graph_custom(g,2048,false);
          ggml_tensor*ifx=ggml_new_tensor_2d(g,GGML_TYPE_F32,n_embd,m);ggml_set_input(ifx);
          ggml_tensor*idT=ggml_new_tensor_2d(g,GGML_TYPE_I32,n_use,m);ggml_set_input(idT);
          ggml_tensor*wT=ggml_new_tensor_3d(g,GGML_TYPE_F32,1,n_use,m);ggml_set_input(wT);
          ggml_tensor*cx=ggml_reshape_3d(g,ifx,n_embd,1,m);
          ggml_tensor*up=ggml_mul_mat_id(g,UPOOL[il],cx,idT);
          ggml_tensor*ga=ggml_silu(g,ggml_mul_mat_id(g,GPOOL[il],cx,idT));
          ggml_tensor*ex=ggml_mul_mat_id(g,DPOOL[il],ggml_mul(g,up,ga),idT); // [n_embd,n_use,m]
          ex=ggml_mul(g,ex,wT);
          ggml_tensor*mo=nullptr;for(int j=0;j<n_use;++j){ggml_tensor*s=ggml_view_2d(g,ex,n_embd,m,ex->nb[2],(size_t)j*ex->nb[1]);mo=j==0?s:ggml_add(g,mo,s);}
          ggml_set_output(mo);ggml_build_forward_expand(gf,mo);ggml_gallocr_alloc_graph(galloc,gf);
          ggml_backend_tensor_set(ifx,ffnx.data(),0,(size_t)n_embd*m*4);ggml_backend_tensor_set(idT,ids.data(),0,(size_t)n_use*m*4);ggml_backend_tensor_set(wT,wgt.data(),0,(size_t)n_use*m*4);
          ggml_backend_graph_compute(M.backend,gf);ggml_backend_tensor_get(mo,moe.data(),0,(size_t)n_embd*m*4);ggml_free(g);}
        for(size_t d=0;d<(size_t)n_embd*m;++d)X[d]=moe[d]+ffninp[d];
    }
    std::vector<float> logits((size_t)n_vocab*m);
    { ggml_context*g=ggml_init({(size_t)160*1024*1024,nullptr,true});ggml_cgraph*gf=ggml_new_graph(g);
      ggml_tensor*ix=ggml_new_tensor_2d(g,GGML_TYPE_F32,n_embd,m);ggml_set_input(ix);
      ggml_tensor*lg=ggml_mul_mat(g,M.get("output.weight"),ggml_mul(g,ggml_rms_norm(g,ix,eps),M.get("output_norm.weight")));ggml_set_output(lg);
      ggml_build_forward_expand(gf,lg);ggml_gallocr_alloc_graph(galloc,gf);ggml_backend_tensor_set(ix,X.data(),0,(size_t)n_embd*m*4);ggml_backend_graph_compute(M.backend,gf);ggml_backend_tensor_get(lg,logits.data(),0,(size_t)n_vocab*m*4);ggml_free(g);}
    return logits;
}
static int amax(const float*l){int bi=0;float bv=l[0];for(int v=1;v<n_vocab;++v)if(l[v]>bv){bv=l[v];bi=v;}return bi;}

int main(int argc,char**argv){
    if(argc<6){fprintf(stderr,"usage: %s model ids_csv ngen K k_draft\n",argv[0]);return 2;}
    int ngen=atoi(argv[3]),K=atoi(argv[4]),kdr=atoi(argv[5]);
    std::vector<int32_t> prompt;{char*s=strdup(argv[2]);for(char*p=strtok(s,",");p;p=strtok(nullptr,","))prompt.push_back(atoi(p));free(s);}
    int T=(int)prompt.size();
    ggml_backend_load_all();ggml_backend_dev_t vk0=nullptr;for(size_t i=0;i<ggml_backend_dev_count();++i){ggml_backend_dev_t d=ggml_backend_dev_get(i);if(!strcmp(ggml_backend_dev_name(d),"Vulkan0")){vk0=d;break;}}
    ggml_backend_t backend=ggml_backend_dev_init(vk0,nullptr);load_model(argv[1],backend);
    int max_kv=T+ngen+kdr+4;
    ggml_context*kvc=ggml_init({(size_t)2*n_layer*ggml_tensor_overhead()+1024,nullptr,true});
    for(int il=0;il<n_layer;++il){KCACHE.push_back(ggml_new_tensor_3d(kvc,GGML_TYPE_F32,head_dim,n_head,max_kv));VCACHE.push_back(ggml_new_tensor_3d(kvc,GGML_TYPE_F32,head_dim,n_head,max_kv));}
    kvbuf=ggml_backend_alloc_ctx_tensors(kvc,backend);
    POOLK=K;ggml_context*pc=ggml_init({(size_t)(n_layer*3+8)*ggml_tensor_overhead(),nullptr,true});
    for(int il=0;il<n_layer;++il){GPOOL.push_back(ggml_new_tensor_3d(pc,H.gtype[il],n_embd,n_ff,K));UPOOL.push_back(ggml_new_tensor_3d(pc,H.utype[il],n_embd,n_ff,K));DPOOL.push_back(ggml_new_tensor_3d(pc,H.dtype[il],n_ff,n_embd,K));}
    poolbuf=ggml_backend_alloc_ctx_tensors(pc,backend);
    slot_expert.assign(n_layer,std::vector<int>(K,-1));expert_slot.resize(n_layer);lru.resize(n_layer);
    for(int il=0;il<n_layer;++il)for(int s=0;s<K;++s)lru[il].push_back(s);
    galloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));

    // ---- prefill prompt (token by token, top-8, store) ----
    std::vector<float> pred;
    for(int i=0;i<T;++i){std::vector<float> lg=forward({prompt[i]},i,8,false,true);pred.assign(lg.begin(),lg.begin()+n_vocab);}
    int cur=T;
    auto reset=[&](){for(int il=0;il<n_layer;++il){slot_expert[il].assign(K,-1);expert_slot[il].clear();lru[il].clear();for(int s=0;s<K;++s)lru[il].push_back(s);}};

    struct timespec a,b;
    // ===== baseline: target-alone =====
    std::vector<int32_t> base_seq;
    { std::vector<float> pr=pred; int c=cur; clock_gettime(CLOCK_MONOTONIC,&a); int made=0;
      while(made<ngen){int nx=amax(pr.data());base_seq.push_back(nx);std::vector<float> lg=forward({nx},c,8,false,true);pr.assign(lg.begin(),lg.begin()+n_vocab);c++;made++;if(nx==50279)break;}
      clock_gettime(CLOCK_MONOTONIC,&b);double s=(b.tv_sec-a.tv_sec)+(b.tv_nsec-a.tv_nsec)/1e9;
      printf("target-alone:  %d tok in %.3fs = %.2f tok/s\n",made,s,made/s);
    }
    // restore cache state to end-of-prompt for the spec run (re-prefill cheaply by resetting cur+slots)
    reset();
    // re-run prefill to refill cache K/V for positions 0..T-1 (target-alone advanced cache past T)
    for(int i=0;i<T;++i){std::vector<float> lg=forward({prompt[i]},i,8,false,true);pred.assign(lg.begin(),lg.begin()+n_vocab);}
    cur=T;

    // ===== speculative decode (residency-biased draft + batched verify) =====
    long accepted_total=0, proposed_total=0, rounds=0, made=0;
    std::vector<int32_t> spec_seq;
    clock_gettime(CLOCK_MONOTONIC,&a);
    while(made<ngen){
        // DRAFT k tokens (top-1 resident, store top-1 K/V at cur..cur+k-1)
        std::vector<int32_t> draft; std::vector<float> p=pred; int dpos=cur;
        for(int j=0;j<kdr;++j){int d=amax(p.data());draft.push_back(d);std::vector<float> lg=forward({d},dpos,1,true,true);p.assign(lg.begin(),lg.begin()+n_vocab);dpos++;}
        // VERIFY: batched forward over draft (top-8 stream, store top-8 K/V over cur..cur+k-1)
        std::vector<float> V=forward(draft,cur,8,false,true);
        // accept: check draft[0] vs pred; draft[j] vs V at position j-1
        int n=0; std::vector<float> last=pred;
        for(int j=0;j<kdr;++j){ int tgt=amax(j==0?pred.data():&V[(size_t)(j-1)*n_vocab]); if(draft[j]==tgt){n++; if(j==kdr-1) last.assign(&V[(size_t)(kdr-1)*n_vocab],&V[(size_t)(kdr-1)*n_vocab]+n_vocab);} else { last.assign(j==0?pred.data():&V[(size_t)(j-1)*n_vocab], (j==0?pred.data():&V[(size_t)(j-1)*n_vocab])+n_vocab); break; } }
        int corrected=amax(last.data());
        proposed_total+=kdr; accepted_total+=n; rounds++;
        for(int j=0;j<n;++j){spec_seq.push_back(draft[j]);} // accepted drafts
        cur+=n; made+=n; if(made>=ngen)break;
        // process corrected token (single forward top-8, store) -> pred for next round + its K/V
        spec_seq.push_back(corrected);
        std::vector<float> lg=forward({corrected},cur,8,false,true);pred.assign(lg.begin(),lg.begin()+n_vocab);cur++;made++;
        if(corrected==50279||made>=ngen)break;
    }
    clock_gettime(CLOCK_MONOTONIC,&b);double s=(b.tv_sec-a.tv_sec)+(b.tv_nsec-a.tv_nsec)/1e9;
    printf("speculative:   %ld tok in %.3fs = %.2f tok/s   (k=%d, K=%d)\n",made,s,made/s,kdr,K);
    printf("  acceptance = %.1f%% (%ld/%ld)  avg accepted/round = %.2f  rounds=%ld\n",100.0*accepted_total/proposed_total,accepted_total,proposed_total,(double)(made)/rounds,rounds);
    int mlen=0,mn=std::min(base_seq.size(),spec_seq.size());while(mlen<mn&&base_seq[mlen]==spec_seq[mlen])mlen++;
    printf("  lossless check: spec matches target-alone for %d/%d tokens\n",mlen,(int)mn);
    return 0;
}
