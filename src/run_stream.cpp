// M2a: single-token DECODE with a KV cache (experts still resident).
// Validates the KV-cache + decode loop against the M1c reference before adding streaming.
// KV cache uses the concat trick: K_all = concat(cache[0..p-1], K_cur) so there's no
// write-then-read hazard; a separate cpy persists K_cur into the cache for the next step.
//   run_stream <model.gguf> <prompt_ids_csv> <ngen>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <map>

#ifndef GGML_ROPE_TYPE_NEOX
#define GGML_ROPE_TYPE_NEOX 2
#endif

struct Model {
    gguf_context* gguf=nullptr; ggml_context* ctx=nullptr;
    ggml_backend_t backend=nullptr; ggml_backend_buffer_t wbuf=nullptr;
    std::map<std::string, ggml_tensor*> ten;
    ggml_tensor* get(const std::string& n){ auto it=ten.find(n); if(it==ten.end()){fprintf(stderr,"MISSING %s\n",n.c_str());exit(1);} return it->second; }
    ggml_tensor* blk(int il,const char* s){ char b[80]; snprintf(b,sizeof b,"blk.%d.%s",il,s); return get(b); }
};
static void load_model(Model& M, const char* path, ggml_backend_t backend) {
    M.backend=backend; gguf_init_params p={true,&M.ctx};
    M.gguf=gguf_init_from_file(path,p); if(!M.gguf){fprintf(stderr,"gguf fail\n");exit(1);}
    M.wbuf=ggml_backend_alloc_ctx_tensors(M.ctx,backend);
    FILE* f=fopen(path,"rb"); size_t doff=gguf_get_data_offset(M.gguf); std::vector<uint8_t> buf;
    for(ggml_tensor* t=ggml_get_first_tensor(M.ctx); t; t=ggml_get_next_tensor(M.ctx,t)){
        int64_t ti=gguf_find_tensor(M.gguf,t->name); size_t off=gguf_get_tensor_offset(M.gguf,ti), nb=ggml_nbytes(t);
        buf.resize(nb); _fseeki64(f,(int64_t)(doff+off),SEEK_SET);
        if(fread(buf.data(),1,nb,f)!=nb){fprintf(stderr,"read fail %s\n",t->name);exit(1);}
        ggml_backend_tensor_set(t,buf.data(),0,nb); M.ten[t->name]=t;
    }
    fclose(f);
}

static const int n_layer=16,n_embd=2048,n_head=16,head_dim=128,n_ff=1024;
static const int n_expert=64,n_used=8,n_vocab=50304,n_ctx_orig=4096;
static const float eps=1e-5f,freq_base=10000.0f;

struct KV { std::vector<ggml_tensor*> k,v; ggml_context* ctx=nullptr; ggml_backend_buffer_t buf=nullptr; };

static std::vector<float> forward_step(Model& M, ggml_gallocr_t galloc, KV& kv, int32_t tok, int p) {
    ggml_context* g = ggml_init({ (size_t)256*1024*1024, nullptr, true });
    ggml_cgraph* gf = ggml_new_graph_custom(g, 8192, false);

    ggml_tensor* inp_tok = ggml_new_tensor_1d(g, GGML_TYPE_I32, 1); ggml_set_input(inp_tok);
    ggml_tensor* inp_pos = ggml_new_tensor_1d(g, GGML_TYPE_I32, 1); ggml_set_input(inp_pos);
    ggml_tensor* x = ggml_get_rows(g, M.get("token_embd.weight"), inp_tok);
    std::vector<ggml_tensor*> stores;

    for (int il=0; il<n_layer; ++il) {
        ggml_tensor* cur = ggml_mul(g, ggml_rms_norm(g,x,eps), M.blk(il,"attn_norm.weight"));
        ggml_tensor* Qc = ggml_mul_mat(g, M.blk(il,"attn_q.weight"), cur);
        ggml_tensor* Kc = ggml_mul_mat(g, M.blk(il,"attn_k.weight"), cur);
        ggml_tensor* Vc = ggml_mul_mat(g, M.blk(il,"attn_v.weight"), cur);
        Qc = ggml_mul(g, ggml_rms_norm(g,Qc,eps), M.blk(il,"attn_q_norm.weight"));
        Kc = ggml_mul(g, ggml_rms_norm(g,Kc,eps), M.blk(il,"attn_k_norm.weight"));
        Qc = ggml_reshape_3d(g,Qc,head_dim,n_head,1);
        Kc = ggml_reshape_3d(g,Kc,head_dim,n_head,1);
        Vc = ggml_reshape_3d(g,Vc,head_dim,n_head,1);
        Qc = ggml_rope_ext(g,Qc,inp_pos,nullptr,head_dim,GGML_ROPE_TYPE_NEOX,n_ctx_orig,freq_base,1,0,1,32,1);
        Kc = ggml_rope_ext(g,Kc,inp_pos,nullptr,head_dim,GGML_ROPE_TYPE_NEOX,n_ctx_orig,freq_base,1,0,1,32,1);

        ggml_tensor* kc=kv.k[il]; ggml_tensor* vc=kv.v[il];
        ggml_tensor* dk = ggml_view_3d(g,kc,head_dim,n_head,1, kc->nb[1],kc->nb[2], (size_t)p*kc->nb[2]);
        ggml_tensor* dv = ggml_view_3d(g,vc,head_dim,n_head,1, vc->nb[1],vc->nb[2], (size_t)p*vc->nb[2]);
        stores.push_back(ggml_cpy(g,Kc,dk));
        stores.push_back(ggml_cpy(g,Vc,dv));

        ggml_tensor *Kall,*Vall;
        if (p==0){ Kall=Kc; Vall=Vc; }
        else {
            ggml_tensor* ck=ggml_view_3d(g,kc,head_dim,n_head,p, kc->nb[1],kc->nb[2], 0);
            ggml_tensor* cv=ggml_view_3d(g,vc,head_dim,n_head,p, vc->nb[1],vc->nb[2], 0);
            Kall=ggml_concat(g,ck,Kc,2); Vall=ggml_concat(g,cv,Vc,2);
        }
        ggml_tensor* q=ggml_permute(g,Qc,0,2,1,3);
        ggml_tensor* k=ggml_permute(g,Kall,0,2,1,3);
        ggml_tensor* kq=ggml_mul_mat(g,k,q);
        kq=ggml_soft_max_ext(g,kq,nullptr,1.0f/sqrtf((float)head_dim),0.0f);
        ggml_tensor* vv=ggml_cont(g, ggml_permute(g,Vall,1,2,0,3));
        ggml_tensor* kqv=ggml_mul_mat(g,vv,kq);
        kqv=ggml_permute(g,kqv,0,2,1,3);
        cur=ggml_cont_2d(g,kqv,n_embd,1);
        cur=ggml_mul_mat(g, M.blk(il,"attn_output.weight"), cur);
        ggml_tensor* ffn_inp=ggml_add(g,cur,x);

        cur=ggml_mul(g, ggml_rms_norm(g,ffn_inp,eps), M.blk(il,"ffn_norm.weight"));
        ggml_tensor* rlog=ggml_mul_mat(g, M.blk(il,"ffn_gate_inp.weight"), cur);
        ggml_tensor* probs=ggml_soft_max(g,rlog);
        ggml_tensor* sel=ggml_top_k(g,probs,n_used);
        ggml_tensor* w=ggml_get_rows(g, ggml_reshape_3d(g,probs,1,n_expert,1), sel);
        ggml_tensor* cx=ggml_reshape_3d(g,cur,n_embd,1,1);
        ggml_tensor* up=ggml_mul_mat_id(g, M.blk(il,"ffn_up_exps.weight"),  cx,sel);
        ggml_tensor* ga=ggml_silu(g, ggml_mul_mat_id(g, M.blk(il,"ffn_gate_exps.weight"), cx,sel));
        ggml_tensor* par=ggml_mul(g,up,ga);
        ggml_tensor* ex=ggml_mul_mat_id(g, M.blk(il,"ffn_down_exps.weight"), par,sel);
        ex=ggml_mul(g,ex,w);
        ggml_tensor* moe=nullptr;
        for(int e=0;e<n_used;++e){ ggml_tensor* s=ggml_view_2d(g,ex,n_embd,1,ex->nb[2],(size_t)e*ex->nb[1]); moe=e==0?s:ggml_add(g,moe,s); }
        x=ggml_add(g,moe,ffn_inp);
    }
    ggml_tensor* rn=ggml_mul(g, ggml_rms_norm(g,x,eps), M.get("output_norm.weight"));
    ggml_tensor* logits=ggml_mul_mat(g, M.get("output.weight"), rn);
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);
    for(auto s:stores) ggml_build_forward_expand(gf, s);

    ggml_gallocr_alloc_graph(galloc, gf);
    ggml_backend_tensor_set(inp_tok,&tok,0,4);
    int32_t pp=p; ggml_backend_tensor_set(inp_pos,&pp,0,4);
    ggml_backend_graph_compute(M.backend, gf);
    std::vector<float> out(n_vocab);
    ggml_backend_tensor_get(logits,out.data(),0,(size_t)n_vocab*4);
    ggml_free(g);
    return out;
}

int main(int argc,char**argv){
    if(argc<4){fprintf(stderr,"usage: %s model ids_csv ngen\n",argv[0]);return 2;}
    int ngen=atoi(argv[3]);
    std::vector<int32_t> seq;
    { char* s=strdup(argv[2]); for(char* p=strtok(s,","); p; p=strtok(nullptr,",")) seq.push_back(atoi(p)); free(s); }
    int T=(int)seq.size();

    ggml_backend_load_all();
    ggml_backend_dev_t vk0=nullptr;
    for(size_t i=0;i<ggml_backend_dev_count();++i){ggml_backend_dev_t d=ggml_backend_dev_get(i); if(!strcmp(ggml_backend_dev_name(d),"Vulkan0")){vk0=d;break;}}
    ggml_backend_t backend=ggml_backend_dev_init(vk0,nullptr);
    Model M; load_model(M, argv[1], backend);

    const int max_kv = T+ngen+2;
    KV kv; kv.ctx=ggml_init({(size_t)2*n_layer*ggml_tensor_overhead()+1024,nullptr,true});
    for(int il=0;il<n_layer;++il){
        kv.k.push_back(ggml_new_tensor_3d(kv.ctx,GGML_TYPE_F32,head_dim,n_head,max_kv));
        kv.v.push_back(ggml_new_tensor_3d(kv.ctx,GGML_TYPE_F32,head_dim,n_head,max_kv));
    }
    kv.buf=ggml_backend_alloc_ctx_tensors(kv.ctx,backend);
    ggml_gallocr_t galloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));

    std::vector<int32_t> gen;
    for(int p=0;;++p){
        std::vector<float> logits=forward_step(M,galloc,kv,seq[p],p);
        if(p>=T-1){
            int bi=0; float bv=logits[0];
            for(int v=1;v<n_vocab;++v) if(logits[v]>bv){bv=logits[v];bi=v;}
            gen.push_back(bi);
            if((int)gen.size()>=ngen || bi==50279) break;
            seq.push_back(bi);
        }
    }
    printf("GEN:"); for(int32_t id:gen) printf(" %d",id); printf("\n");
    return 0;
}
