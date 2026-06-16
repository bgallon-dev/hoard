// Dump the full tokenizer.chat_template (and bos/eos ids) from a GGUF.
// gguf_dump.c truncates strings to 80 chars; we need the whole template verbatim.
#include "ggml.h"
#include "gguf.h"
#include <stdio.h>

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.gguf\n", argv[0]); return 2; }
    struct gguf_init_params p = { /*no_alloc=*/true, /*ctx=*/NULL };
    struct gguf_context * ctx = gguf_init_from_file(argv[1], p);
    if (!ctx) { fprintf(stderr, "failed to open %s\n", argv[1]); return 1; }

    const char * keys[] = {
        "tokenizer.chat_template",
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.add_bos_token",
    };
    for (int k = 0; k < 4; ++k) {
        int64_t i = gguf_find_key(ctx, keys[k]);
        if (i < 0) { printf("[%s] = <absent>\n", keys[k]); continue; }
        enum gguf_type t = gguf_get_kv_type(ctx, i);
        if (t == GGUF_TYPE_STRING)      printf("[%s] =\n<<<\n%s\n>>>\n", keys[k], gguf_get_val_str(ctx, i));
        else if (t == GGUF_TYPE_UINT32) printf("[%s] = %u\n", keys[k], gguf_get_val_u32(ctx, i));
        else if (t == GGUF_TYPE_BOOL)   printf("[%s] = %s\n", keys[k], gguf_get_val_bool(ctx, i) ? "true":"false");
        else                            printf("[%s] = (type %d)\n", keys[k], (int)t);
    }
    gguf_free(ctx);
    return 0;
}
