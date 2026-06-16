// Smoke test: link our own program against the prebuilt ggml DLLs.
// Validates the whole "Option C" path: ucrt64 gcc -> Clang-built ggml C ABI.
#include "ggml.h"
#include "ggml-backend.h"
#include <stdio.h>

int main(void) {
    printf("GGML_TYPE_COUNT = %d\n", (int)GGML_TYPE_COUNT);

    struct ggml_init_params p = { 16*1024*1024, NULL, false };
    struct ggml_context * ctx = ggml_init(p);
    if (!ctx) { printf("ggml_init FAILED\n"); return 1; }

    struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    printf("tensor a: ne0=%lld nbytes=%zu\n", (long long)a->ne[0], ggml_nbytes(a));
    ggml_free(ctx);

    // With GGML_BACKEND_DL, backends live in separate ggml-*.dll next to the exe
    // and must be explicitly loaded before they register devices.
    ggml_backend_load_all();

    // Enumerate backends.
    size_t n = ggml_backend_dev_count();
    printf("backend devices: %zu\n", n);
    for (size_t i = 0; i < n; ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        printf("  dev[%zu]: %s  (%s)\n", i, ggml_backend_dev_name(d), ggml_backend_dev_description(d));
    }
    printf("OK\n");
    return 0;
}
