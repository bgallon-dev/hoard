// M1c reference generator: tokenize each prompt, greedy-decode ngen tokens via llama.dll
#include <algorithm>
// (FA disabled to match our explicit-attention engine), print prompt IDs + generated IDs.
//   ref_gen <model.gguf> <ngen> "prompt1" "prompt2" ...
#include "llama.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr,"usage: %s model.gguf ngen prompt...\n",argv[0]); return 2; }
    int ngen = atoi(argv[2]);

    ggml_backend_load_all();
    ggml_backend_dev_t vk0=nullptr;
    for (size_t i=0;i<ggml_backend_dev_count();++i){ ggml_backend_dev_t d=ggml_backend_dev_get(i); if(!strcmp(ggml_backend_dev_name(d),"Vulkan0")){vk0=d;break;} }
    ggml_backend_dev_t devs[2]={vk0,nullptr};

    llama_model_params mp = llama_model_default_params();
    const char* nglenv = getenv("NGL"); int ngl = nglenv ? atoi(nglenv) : 99;
    static ggml_backend_dev_t nogpu[1] = { nullptr };   // empty list -> CPU only
    mp.n_gpu_layers = ngl;
    if (ngl > 0 && vk0) mp.devices = devs;
    else                mp.devices = nogpu;             // force CPU: no GPU devices at all
    fprintf(stderr, "[ref_gen] n_gpu_layers=%d devices=%s\n", ngl, (ngl>0&&vk0)?"Vulkan0":"CPU-only");
    llama_model* model = llama_model_load_from_file(argv[1], mp);
    if (!model){ fprintf(stderr,"load failed\n"); return 1; }
    const llama_vocab* vocab = llama_model_get_vocab(model);
    int n_vocab = llama_vocab_n_tokens(vocab);
    llama_token eos = llama_vocab_eos(vocab);

    for (int a=3; a<argc; ++a) {
        const char* text = argv[a];
        int32_t need = llama_tokenize(vocab, text, (int)strlen(text), nullptr, 0, true, false);
        std::vector<llama_token> toks(-need);
        llama_tokenize(vocab, text, (int)strlen(text), toks.data(), (int)toks.size(), true, false);

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx=512; cp.n_batch=512; cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
        llama_context* ctx = llama_init_from_model(model, cp);

        llama_batch b = llama_batch_get_one(toks.data(), (int)toks.size());
        llama_decode(ctx, b);

        std::vector<llama_token> gen;
        for (int i=0;i<ngen;++i){
            float* lg = llama_get_logits_ith(ctx, -1);
            int bi=0; float bv=lg[0];
            for (int v=1; v<n_vocab; ++v) if (lg[v]>bv){bv=lg[v]; bi=v;}
            // top-5 dump to expose near-ties (logit gap between #1 and #2)
            { std::vector<int> idx(n_vocab); for(int v=0;v<n_vocab;++v) idx[v]=v;
              std::partial_sort(idx.begin(), idx.begin()+5, idx.end(), [&](int a,int b){return lg[a]>lg[b];});
              fprintf(stderr,"  pos%d top5:", i); for(int k=0;k<5;++k) fprintf(stderr," %d=%.3f", idx[k], lg[idx[k]]); fprintf(stderr,"  (gap %.3f)\n", lg[idx[0]]-lg[idx[1]]); }
            gen.push_back(bi);
            if (bi==eos) break;
            llama_token t = bi;
            llama_batch bb = llama_batch_get_one(&t, 1);
            llama_decode(ctx, bb);
        }

        printf("=== prompt: \"%s\"\n", text);
        printf("PROMPT_IDS:"); for (auto t: toks) printf(" %d", t); printf("\n");
        printf("GEN:");        for (auto t: gen)  printf(" %d", t); printf("\n");
        std::string s; for (auto t: gen){ char buf[256]; int k=llama_token_to_piece(vocab,t,buf,256,0,false); if(k>0) s.append(buf,k);}
        printf("GEN_TEXT: %s\n", s.c_str());
        llama_free(ctx);
    }
    return 0;
}
