// gguf_dump: read OLMoE GGUF metadata + tensor table via ggml's gguf API.
// Purpose: (1) validate we can parse the model file ourselves, (2) confirm the
// [runtime-confirm] values in notes/olmoe_spec.md against ground truth.
#include "ggml.h"
#include "gguf.h"
#include <stdio.h>
#include <string.h>

static void print_kv(const struct gguf_context * ctx, int64_t i) {
    const char * key = gguf_get_key(ctx, i);
    enum gguf_type t = gguf_get_kv_type(ctx, i);
    printf("  %-44s ", key);
    switch (t) {
        case GGUF_TYPE_UINT8:   printf("u8   = %u\n",  gguf_get_val_u8(ctx, i)); break;
        case GGUF_TYPE_INT8:    printf("i8   = %d\n",  gguf_get_val_i8(ctx, i)); break;
        case GGUF_TYPE_UINT16:  printf("u16  = %u\n",  gguf_get_val_u16(ctx, i)); break;
        case GGUF_TYPE_INT16:   printf("i16  = %d\n",  gguf_get_val_i16(ctx, i)); break;
        case GGUF_TYPE_UINT32:  printf("u32  = %u\n",  gguf_get_val_u32(ctx, i)); break;
        case GGUF_TYPE_INT32:   printf("i32  = %d\n",  gguf_get_val_i32(ctx, i)); break;
        case GGUF_TYPE_FLOAT32: printf("f32  = %.9g\n",gguf_get_val_f32(ctx, i)); break;
        case GGUF_TYPE_UINT64:  printf("u64  = %llu\n",(unsigned long long)gguf_get_val_u64(ctx, i)); break;
        case GGUF_TYPE_INT64:   printf("i64  = %lld\n",(long long)gguf_get_val_i64(ctx, i)); break;
        case GGUF_TYPE_FLOAT64: printf("f64  = %.17g\n",gguf_get_val_f64(ctx, i)); break;
        case GGUF_TYPE_BOOL:    printf("bool = %s\n",  gguf_get_val_bool(ctx, i) ? "true":"false"); break;
        case GGUF_TYPE_STRING:  printf("str  = \"%.80s\"\n", gguf_get_val_str(ctx, i)); break;
        case GGUF_TYPE_ARRAY: {
            enum gguf_type at = gguf_get_arr_type(ctx, i);
            size_t n = gguf_get_arr_n(ctx, i);
            printf("arr<%s>[%zu]", gguf_type_name(at), n);
            if (at == GGUF_TYPE_STRING && n > 0)
                printf(" e0=\"%.24s\"", gguf_get_arr_str(ctx, i, 0));
            printf("\n");
        } break;
        default: printf("(type %d)\n", (int)t);
    }
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.gguf\n", argv[0]); return 2; }

    struct ggml_context * meta = NULL;
    struct gguf_init_params p = { /*no_alloc=*/true, /*ctx=*/&meta };
    struct gguf_context * ctx = gguf_init_from_file(argv[1], p);
    if (!ctx) { fprintf(stderr, "failed to open %s\n", argv[1]); return 1; }

    printf("== GGUF header ==\n");
    printf("  version    = %u\n", gguf_get_version(ctx));
    printf("  alignment  = %zu\n", gguf_get_alignment(ctx));
    printf("  n_kv       = %lld\n", (long long)gguf_get_n_kv(ctx));
    printf("  n_tensors  = %lld\n", (long long)gguf_get_n_tensors(ctx));
    printf("  data_off   = %zu\n", gguf_get_data_offset(ctx));

    printf("\n== metadata KV ==\n");
    for (int64_t i = 0; i < gguf_get_n_kv(ctx); ++i) print_kv(ctx, i);

    // Tensor table: iterate the ggml_context gguf populated (shapes live there).
    printf("\n== tensors (name : type [ne0,ne1,ne2,ne3] bytes) ==\n");
    int n_expert_tensors = 0;
    size_t expert_bytes = 0, total_bytes = 0;
    for (struct ggml_tensor * t = ggml_get_first_tensor(meta); t; t = ggml_get_next_tensor(meta, t)) {
        size_t nb = ggml_nbytes(t);
        total_bytes += nb;
        int is_exp = (strstr(t->name, "_exps") != NULL);
        if (is_exp) { n_expert_tensors++; expert_bytes += nb; }
        // print only layer-0 + globals to keep output readable
        if (strstr(t->name,"blk.0.") || strncmp(t->name,"blk.",4)!=0) {
            printf("  %-28s : %-8s [%lld,%lld,%lld,%lld] %zu\n",
                t->name, ggml_type_name(t->type),
                (long long)t->ne[0],(long long)t->ne[1],(long long)t->ne[2],(long long)t->ne[3], nb);
        }
    }
    printf("\n== summary ==\n");
    printf("  expert tensors        = %d\n", n_expert_tensors);
    printf("  expert bytes (stream) = %.1f MB\n", expert_bytes/1e6);
    printf("  total tensor bytes    = %.1f MB\n", total_bytes/1e6);
    printf("  non-expert (resident) = %.1f MB\n", (total_bytes-expert_bytes)/1e6);

    gguf_free(ctx);
    ggml_free(meta);
    return 0;
}
