// M1a: dense oracle. Instrument the prebuilt llama.dll via the eval callback to dump
// every named intermediate tensor of the OLMoE graph for a fixed prompt. These golden
// activations let us build our own forward pass stage-by-stage, validating each stage
// against the reference BEFORE composing the next (per-kernel-oracle discipline at graph level).
//
// FA disabled so the reference uses the same explicit-softmax attention our engine will.
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <map>

struct Dump {
    std::string outdir;
    std::map<std::string,int> seen;   // name -> occurrence count (Qcur appears pre- and post-RoPE)
    FILE* meta = nullptr;
};
static Dump g;

// What to capture: full detail for layer 0, plus per-layer l_out and the head.
static bool want(const char* name) {
    static const char* subs[] = {
        "embd", "inp_tokens",
        "attn_norm-0", "Qcur-0", "Kcur-0", "Vcur-0", "Qcur_normed-0", "Kcur_normed-0",
        // attention internals (FA disabled): scores -> softmax -> weighted-V -> wo.
        // kqv_out-0 is the PRE-RESIDUAL attention output (unmasked by the skip connection).
        "kq-0", "kq_soft_max-0", "kqv-0", "kqv_out-0",
        "ffn_inp-0", "ffn_norm-0", "ffn_moe_out-0", "l_out-", "result_norm", "result_output",
        // per-layer split to localize the massive activation: attn residual vs moe output
        "attn_residual-", "ffn_out-", "post_moe-",
        // routed vs shared expert split
        "ffn_moe_out-", "ffn_shexp_gated-", "ffn_shexp-", "ffn_moe_logits-", "attn_post_norm-", "linear_attn_out-",
    };
    for (const char* s : subs) if (strstr(name, s)) return true;
    return false;
}

static bool eval_cb(struct ggml_tensor* t, bool ask, void* /*ud*/) {
    if (ask) return true;                 // observe every node
    if (!t->name[0] || !want(t->name)) return true;

    int occ = g.seen[t->name]++;
    size_t nb = ggml_nbytes(t);
    std::vector<uint8_t> buf(nb);
    ggml_backend_tensor_get(t, buf.data(), 0, nb);   // device -> host

    char fn[600];
    snprintf(fn, sizeof fn, "%s/%s_%d.bin", g.outdir.c_str(), t->name, occ);
    if (FILE* f = fopen(fn, "wb")) { fwrite(buf.data(), 1, nb, f); fclose(f); }

    // quick stats (intermediates are F32)
    int64_t n = ggml_nelements(t);
    double mean = 0, std = 0; float first = 0, fmin = 0, fmax = 0;
    if (t->type == GGML_TYPE_F32 && n > 0) {
        const float* d = (const float*)buf.data();
        first = fmin = fmax = d[0];
        for (int64_t i = 0; i < n; ++i) { mean += d[i]; if (d[i]<fmin) fmin=d[i]; if (d[i]>fmax) fmax=d[i]; }
        mean /= n;
        for (int64_t i = 0; i < n; ++i) { double e = d[i]-mean; std += e*e; }
        std = sqrt(std/n);
    }
    fprintf(g.meta, "%-24s occ=%d type=%-5s ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] cont=%d mean=%+.6f std=%.6f min=%+.4f max=%+.4f first=%+.6f\n",
        t->name, occ, ggml_type_name(t->type),
        (long long)t->ne[0],(long long)t->ne[1],(long long)t->ne[2],(long long)t->ne[3],
        t->nb[0],t->nb[1],t->nb[2],t->nb[3], (int)ggml_is_contiguous(t),
        mean, std, fmin, fmax, first);
    fflush(g.meta);
    return true;
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model.gguf outdir\n", argv[0]); return 2; }
    g.outdir = argv[2];
    std::string metapath = g.outdir + "/_meta.txt";
    g.meta = fopen(metapath.c_str(), "w");
    if (!g.meta) { fprintf(stderr, "cannot write %s\n", metapath.c_str()); return 1; }

    ggml_backend_load_all();

    // pin to the AMD GPU (device named "Vulkan0"); avoid the Intel iGPU (Vulkan1)
    ggml_backend_dev_t vk0 = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (strcmp(ggml_backend_dev_name(d), "Vulkan0") == 0) { vk0 = d; break; }
    }
    ggml_backend_dev_t devs[2] = { vk0, nullptr };

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;          // CPU reference (80B won't fit 8GB; match the gold CPU oracle numerics)
    (void)devs; (void)vk0;
    llama_model* model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 64; cp.n_batch = 64; cp.n_ubatch = 64;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.cb_eval = eval_cb;
    cp.cb_eval_user_data = &g;
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx init failed\n"); return 1; }

    // single token "The" (785) — matches the engine's pos0 (TFLEN=1) for per-layer l_out comparison
    llama_token toks[] = {785};
    int n = (int)(sizeof(toks)/sizeof(toks[0]));
    llama_batch batch = llama_batch_get_one(toks, n);

    int rc = llama_decode(ctx, batch);
    fprintf(stderr, "llama_decode rc=%d ; dumped to %s\n", rc, g.outdir.c_str());

    fclose(g.meta);
    return 0;
}
