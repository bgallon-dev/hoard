// Vocab-only tokenizer: load just the GGUF vocab (fast, no weights / no GPU), tokenize a prompt
// read from a file, print comma-separated token IDs ready to feed run_moe_stream.
//   tok <model.gguf> <prompt_file> [add_special=0] [parse_special=1]
#include "llama.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model.gguf prompt_file [add_special=0] [parse_special=1]\n", argv[0]); return 2; }
    bool add_special   = (argc > 3) ? atoi(argv[3]) : 0;
    bool parse_special = (argc > 4) ? atoi(argv[4]) : 1;

    // slurp prompt file
    FILE* f = fopen(argv[2], "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }
    std::string text; { char buf[4096]; size_t n; while ((n = fread(buf,1,sizeof buf,f)) > 0) text.append(buf,n); } fclose(f);

    ggml_backend_load_all();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;                              // no weights -> instant load, no Vulkan alloc
    llama_model* m = llama_model_load_from_file(argv[1], mp);
    if (!m) { fprintf(stderr, "load failed\n"); return 1; }
    const llama_vocab* v = llama_model_get_vocab(m);

    int32_t need = llama_tokenize(v, text.c_str(), (int)text.size(), nullptr, 0, add_special, parse_special);
    std::vector<llama_token> toks(need < 0 ? -need : need);
    llama_tokenize(v, text.c_str(), (int)text.size(), toks.data(), (int)toks.size(), add_special, parse_special);

    for (size_t i = 0; i < toks.size(); ++i) printf("%s%d", i ? "," : "", toks[i]);
    printf("\n");
    fprintf(stderr, "%d tokens\n", (int)toks.size());
    return 0;
}
