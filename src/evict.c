// Evict the OS file cache by allocating+touching ~N GB, forcing reclaimable file pages out.
// Frees on exit, but the model file stays cold for the next run.  usage: evict [GB=13]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc, char** argv) {
    size_t gb = argc > 1 ? (size_t)atoll(argv[1]) : 13;
    size_t chunk = 1ULL << 30;                 // 1 GiB chunks (dodges .NET/2GB limits; this is C anyway)
    char** ps = (char**)malloc(sizeof(char*) * gb);
    size_t got = 0;
    for (size_t i = 0; i < gb; ++i) {
        ps[i] = (char*)malloc(chunk);
        if (!ps[i]) { fprintf(stderr, "alloc stopped at %zu GiB\n", i); break; }
        memset(ps[i], (int)(i & 0xff), chunk);  // touch every page -> commit -> evict file cache
        got = i + 1;
    }
    fprintf(stderr, "evict: touched %zu GiB, file cache flushed\n", got);
    return 0;
}
