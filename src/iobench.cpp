// Measure realized scattered-read bandwidth vs queue depth on the actual NVMe, cold.
// FILE_FLAG_NO_BUFFERING => device-direct (bypasses OS page cache). QD = #threads, each
// issuing synchronous random aligned reads => N threads = N outstanding I/Os = QD N.
//   iobench <file> <block_bytes> <total_reads>
#include <windows.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <thread>
#include <atomic>

static const char* g_file;
static uint64_t     g_filesize;
static size_t       g_block;
static std::atomic<long long> g_bytes{0};

static void worker(int seed, int nreads) {
    HANDLE h = CreateFileA(g_file, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
                           FILE_FLAG_NO_BUFFERING | FILE_FLAG_RANDOM_ACCESS, NULL);
    if (h == INVALID_HANDLE_VALUE) { fprintf(stderr, "open fail %lu\n", GetLastError()); return; }
    char* buf = (char*)VirtualAlloc(NULL, g_block, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE); // page-aligned
    unsigned s = (unsigned)seed * 2654435761u + 1u;
    uint64_t maxoff = g_filesize - g_block;
    for (int i = 0; i < nreads; ++i) {
        s = s * 1103515245u + 12345u;
        uint64_t off = ((uint64_t)s * 4096ull) % maxoff;
        off &= ~((uint64_t)4095);                                  // sector-align for NO_BUFFERING
        LARGE_INTEGER li; li.QuadPart = (LONGLONG)off;
        SetFilePointerEx(h, li, NULL, FILE_BEGIN);
        DWORD got = 0;
        if (ReadFile(h, buf, (DWORD)g_block, &got, NULL)) g_bytes += got;
        else { fprintf(stderr, "read fail %lu at %llu\n", GetLastError(), off); break; }
    }
    VirtualFree(buf, 0, MEM_RELEASE); CloseHandle(h);
}

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s file block_bytes total_reads\n", argv[0]); return 2; }
    g_file = argv[1]; g_block = (size_t)atoll(argv[2]); int total = atoi(argv[3]);
    HANDLE h = CreateFileA(g_file, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, 0, NULL);
    LARGE_INTEGER sz; GetFileSizeEx(h, &sz); g_filesize = (uint64_t)sz.QuadPart; CloseHandle(h);
    printf("file %.1f GB, block %.0f KB, ~%d reads/trial, cold (NO_BUFFERING)\n",
           g_filesize / 1e9, g_block / 1024.0, total);
    int qds[] = {1, 2, 4, 8, 16, 24, 32};
    double bw1 = 0;
    for (int qi = 0; qi < (int)(sizeof(qds)/sizeof(int)); ++qi) {
        int qd = qds[qi];
        g_bytes = 0;
        LARGE_INTEGER f, t0, t1; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&t0);
        std::vector<std::thread> ths; int per = total / qd;
        for (int i = 0; i < qd; ++i) ths.emplace_back(worker, i + 1, per);
        for (auto& th : ths) th.join();
        QueryPerformanceCounter(&t1);
        double secs = (double)(t1.QuadPart - t0.QuadPart) / f.QuadPart;
        double gb = g_bytes / 1e9, bw = gb / secs;
        if (qd == 1) bw1 = bw;
        printf("QD %2d: %5d reads, %6.2f GB/s, %.3f ms/read avg  (%.2fx vs QD1)\n",
               qd, per * qd, bw, secs * 1000.0 / (per * qd), bw / bw1);
    }
    return 0;
}
