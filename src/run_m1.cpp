// M1c: greedy generator using the validated from-scratch OLMoE forward pass.
// No KV cache (recompute the growing prefix each step) — correctness first.
//   run_m1 <model.gguf> <prompt_ids_csv> <ngen>
// Prints: "GEN: id id id ..."  (the greedy continuation token IDs)
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
    gguf_context* gguf = nullptr;
    ggml_context* ctx  = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;
    std::map<std::string, ggml_tensor*> ten;
    ggml_tensor* get(const std::string& n){ auto it=ten.find(n); if(it==ten.end()){fprintf(stderr,"MISSING %s\n",n.c_str());exit(1);} return it->second; }
    ggml_tensor* blk(int il,const char* s){ char b[80]; snprintf(b,sizeof b,"blk.%d.%s",il,s); return get(b); }
};

static void load_model(Model& M, const char* path, ggml_backend_t backend) {
    M.backend = backend;
    gguf_init_params p = { true, &M.ctx };
    M.gguf = gguf_init_from_file(path, p);
    if (!M.gguf) { fprintf(stderr,"gguf open failed\n"); exit(1); }
    M.wbuf = ggml_backend_alloc_ctx_tensors(M.ctx, backend);
    FILE* f = fopen(path,"rb");
    size_t data_off = gguf_get_data_offset(M.gguf);
    std::vector<uint8_t> buf;
    for (ggml_tensor* t=ggml_get_first_tensor(M.ctx); t; t=ggml_get_next_tensor(M.ctx,t)) {
        int64_t ti = gguf_find_tensor(M.gguf, t->name);
        size_t off = gguf_get_tensor_offset(M.gguf, ti), nb = ggml_nbytes(t);
        buf.resize(nb);
        _fseeki64(f, (int64_t)(data_off+off), SEEK_SET);          // 64-bit seek (>2GB file)
        if (fread(buf.data(),1,nb,f)!=nb){fprintf(stderr,"read fail %s\n",t->name);exit(1);}
        ggml_backend_tensor_set(t, buf.data(), 0, nb);
        M.ten[t->name]=t;
    }
    fclose(f);
}

// hparams (confirmed)
static const int n_layer=16, n_embd=2048, n_head=16, head_dim=128, n_ff=1024;
static const int n_expert=64, n_used=8, n_vocab=50304, n_ctx_orig=4096;
static const float eps=1e-5f, freq_base=10000.0f;

// build graph for `tokens`, compute, return last-position logits [n_vocab]
static std::vector<float> forward(Model& M, ggml_gallocr_t galloc, const std::vector<int32_t>& tokens) {
    const int T = (int)tokens.size();
    ggml_context* gctx = ggml_init({ (size_t)256*1024*1024, nullptr, true });
    ggml_cgraph* gf = ggml_new_graph_custom(gctx, 16384, false);

    ggml_tensor* inp_tokens = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, T); ggml_set_input(inp_tokens);
    ggml_tensor* inp_pos    = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, T); ggml_set_input(inp_pos);
    ggml_tensor* inp_mask   = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, T, T); ggml_set_input(inp_mask);

    ggml_tensor* inpL = ggml_get_rows(gctx, M.get("token_embd.weight"), inp_tokens);
    for (int il=0; il<n_layer; ++il) {
        ggml_tensor* inpSA = inpL;
        ggml_tensor* cur = ggml_mul(gctx, ggml_rms_norm(gctx,inpL,eps), M.blk(il,"attn_norm.weight"));
        ggml_tensor* Q = ggml_mul_mat(gctx, M.blk(il,"attn_q.weight"), cur);
        ggml_tensor* K = ggml_mul_mat(gctx, M.blk(il,"attn_k.weight"), cur);
        ggml_tensor* V = ggml_mul_mat(gctx, M.blk(il,"attn_v.weight"), cur);
        Q = ggml_mul(gctx, ggml_rms_norm(gctx,Q,eps), M.blk(il,"attn_q_norm.weight"));
        K = ggml_mul(gctx, ggml_rms_norm(gctx,K,eps), M.blk(il,"attn_k_norm.weight"));
        Q = ggml_reshape_3d(gctx,Q,head_dim,n_head,T);
        K = ggml_reshape_3d(gctx,K,head_dim,n_head,T);
        V = ggml_reshape_3d(gctx,V,head_dim,n_head,T);
        Q = ggml_rope_ext(gctx,Q,inp_pos,nullptr,head_dim,GGML_ROPE_TYPE_NEOX,n_ctx_orig,freq_base,1,0,1,32,1);
        K = ggml_rope_ext(gctx,K,inp_pos,nullptr,head_dim,GGML_ROPE_TYPE_NEOX,n_ctx_orig,freq_base,1,0,1,32,1);
        ggml_tensor* q = ggml_permute(gctx,Q,0,2,1,3);
        ggml_tensor* k = ggml_permute(gctx,K,0,2,1,3);
        ggml_tensor* kq = ggml_mul_mat(gctx,k,q);
        kq = ggml_soft_max_ext(gctx, kq, inp_mask, 1.0f/sqrtf((float)head_dim), 0.0f);
        ggml_tensor* v = ggml_cont(gctx, ggml_permute(gctx,V,1,2,0,3));
        ggml_tensor* kqv = ggml_mul_mat(gctx,v,kq);
        kqv = ggml_permute(gctx,kqv,0,2,1,3);
        cur = ggml_cont_2d(gctx,kqv,n_embd,T);
        cur = ggml_mul_mat(gctx, M.blk(il,"attn_output.weight"), cur);
        ggml_tensor* ffn_inp = ggml_add(gctx, cur, inpSA);
        cur = ggml_mul(gctx, ggml_rms_norm(gctx,ffn_inp,eps), M.blk(il,"ffn_norm.weight"));
        ggml_tensor* rlog  = ggml_mul_mat(gctx, M.blk(il,"ffn_gate_inp.weight"), cur);
        ggml_tensor* probs = ggml_soft_max(gctx, rlog);
        ggml_tensor* sel   = ggml_top_k(gctx, probs, n_used);
        ggml_tensor* w     = ggml_get_rows(gctx, ggml_reshape_3d(gctx,probs,1,n_expert,T), sel);
        ggml_tensor* cx    = ggml_reshape_3d(gctx, cur, n_embd, 1, T);
        ggml_tensor* up    = ggml_mul_mat_id(gctx, M.blk(il,"ffn_up_exps.weight"),   cx, sel);
        ggml_tensor* ga    = ggml_silu(gctx, ggml_mul_mat_id(gctx, M.blk(il,"ffn_gate_exps.weight"), cx, sel));
        ggml_tensor* par   = ggml_mul(gctx, up, ga);
        ggml_tensor* ex    = ggml_mul_mat_id(gctx, M.blk(il,"ffn_down_exps.weight"), par, sel);
        ex = ggml_mul(gctx, ex, w);
        ggml_tensor* moe = nullptr;
        for (int e=0;e<n_used;++e){
            ggml_tensor* s = ggml_view_2d(gctx, ex, n_embd, T, ex->nb[2], (size_t)e*ex->nb[1]);
            moe = e==0 ? s : ggml_add(gctx, moe, s);
        }
        inpL = ggml_add(gctx, moe, ffn_inp);
    }
    ggml_tensor* rn = ggml_mul(gctx, ggml_rms_norm(gctx,inpL,eps), M.get("output_norm.weight"));
    ggml_tensor* logits = ggml_mul_mat(gctx, M.get("output.weight"), rn);
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);

    ggml_gallocr_alloc_graph(galloc, gf);
    std::vector<int32_t> pos(T); for(int i=0;i<T;++i) pos[i]=i;
    std::vector<float> mask((size_t)T*T);
    for(int qi=0;qi<T;++qi) for(int kv=0;kv<T;++kv) mask[(size_t)qi*T+kv]=(kv<=qi)?0.0f:-INFINITY;
    ggml_backend_tensor_set(inp_tokens, tokens.data(),0,(size_t)T*4);
    ggml_backend_tensor_set(inp_pos,    pos.data(),   0,(size_t)T*4);
    ggml_backend_tensor_set(inp_mask,   mask.data(),  0,(size_t)T*T*4);
    ggml_backend_graph_compute(M.backend, gf);

    std::vector<float> out(n_vocab);
    ggml_backend_tensor_get(logits, out.data(), (size_t)(T-1)*n_vocab*4, (size_t)n_vocab*4);
    ggml_free(gctx);
    return out;
}

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr,"usage: %s model.gguf prompt_ids_csv ngen\n",argv[0]); return 2; }
    int ngen = atoi(argv[3]);

    std::vector<int32_t> tokens;
    { char* s = strdup(argv[2]); for (char* p=strtok(s,","); p; p=strtok(nullptr,",")) tokens.push_back(atoi(p)); free(s); }

    ggml_backend_load_all();
    ggml_backend_dev_t vk0=nullptr;
    for (size_t i=0;i<ggml_backend_dev_count();++i){ ggml_backend_dev_t d=ggml_backend_dev_get(i); if(!strcmp(ggml_backend_dev_name(d),"Vulkan0")){vk0=d;break;} }
    ggml_backend_t backend = ggml_backend_dev_init(vk0,nullptr);
    Model M; load_model(M, argv[1], backend);
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));

    std::vector<int32_t> gen;
    for (int i=0;i<ngen;++i) {
        std::vector<float> logits = forward(M, galloc, tokens);
        long bi=-1; float bv=-INFINITY;
        for (int v=0; v<n_vocab; ++v) if (logits[v]>bv){bv=logits[v]; bi=v;}
        gen.push_back((int32_t)bi);
        if (bi==50279) break;                 // eos
        tokens.push_back((int32_t)bi);
    }
    printf("GEN:");
    for (int32_t id : gen) printf(" %d", id);
    printf("\n");
    return 0;
}
