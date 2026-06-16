// fbin: tiny f32 binary inspector/comparator. The workhorse of M1b stage validation.
//   fbin top  <a.bin> [k]         -> top-k (index,value) of a flat f32 array
//   fbin diff <a.bin> <b.bin>     -> element count, max/mean abs err, rms, max rel err
// Files are raw little-endian f32 dumps (as produced by ref_dump and our engine).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int cmpf(const void* x, const void* y) {
    float fx = *(const float*)x, fy = *(const float*)y;
    return (fx > fy) - (fx < fy);
}

static float * load(const char * path, long * n_out) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long bytes = ftell(f); fseek(f, 0, SEEK_SET);
    long n = bytes / 4;
    float * d = (float*)malloc(n * sizeof(float));
    if (fread(d, 4, n, f) != (size_t)n) { fprintf(stderr, "short read %s\n", path); exit(1); }
    fclose(f);
    *n_out = n;
    return d;
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: fbin top <a> [k] | diff <a> <b>\n"); return 2; }

    if (!strcmp(argv[1], "top")) {
        long n; float * a = load(argv[2], &n);
        int k = (argc > 3) ? atoi(argv[3]) : 5;
        printf("n=%ld, top-%d:\n", n, k);
        for (int r = 0; r < k; ++r) {
            long bi = -1; float bv = -INFINITY;
            for (long i = 0; i < n; ++i) if (a[i] > bv) { bv = a[i]; bi = i; }
            printf("  #%d  index=%ld  value=%.6f\n", r, bi, bv);
            a[bi] = -INFINITY; // mask for next
        }
        return 0;
    }

    if (!strcmp(argv[1], "diff")) {
        long na, nb; float * a = load(argv[2], &na); float * b = load(argv[3], &nb);
        if (na != nb) { printf("SIZE MISMATCH na=%ld nb=%ld\n", na, nb); return 1; }
        double max_abs = 0, sum_abs = 0, sum_sq = 0, max_rel = 0;
        long argmax_a = 0, argmax_b = 0; float va = -INFINITY, vb = -INFINITY;
        for (long i = 0; i < na; ++i) {
            double e = fabs((double)a[i] - b[i]);
            if (e > max_abs) max_abs = e;
            sum_abs += e; sum_sq += e*e;
            double denom = fabs(b[i]) > 1e-6 ? fabs(b[i]) : 1e-6;
            double rel = e / denom; if (rel > max_rel) max_rel = rel;
            if (a[i] > va) { va = a[i]; argmax_a = i; }
            if (b[i] > vb) { vb = b[i]; argmax_b = i; }
        }
        printf("n=%ld  max_abs=%.3e  mean_abs=%.3e  rms=%.3e  max_rel=%.3e\n",
               na, max_abs, sum_abs/na, sqrt(sum_sq/na), max_rel);
        printf("argmax: a=%ld (%.4f)  b=%ld (%.4f)  %s\n",
               argmax_a, va, argmax_b, vb, argmax_a==argmax_b ? "MATCH" : "DIFFER");
        return 0;
    }
    if (!strcmp(argv[1], "stat")) {
        long n; float * a = load(argv[2], &n);
        double mean=0,mn=a[0],mx=a[0];
        for (long i=0;i<n;++i){ mean+=a[i]; if(a[i]<mn)mn=a[i]; if(a[i]>mx)mx=a[i]; }
        mean/=n; double s=0; for(long i=0;i<n;++i){double e=a[i]-mean; s+=e*e;}
        printf("n=%ld mean=%.6f std=%.6f min=%.4f max=%.4f\n", n, mean, sqrt(s/n), mn, mx);
        return 0;
    }

    if (!strcmp(argv[1], "sdiff")) {  // sorted-multiset diff: ~0 means same values, different order
        long na, nb; float * a = load(argv[2], &na); float * b = load(argv[3], &nb);
        if (na != nb) { printf("SIZE MISMATCH\n"); return 1; }
        qsort(a,na,4,cmpf); qsort(b,nb,4,cmpf);
        double max_abs=0,sum=0; for(long i=0;i<na;++i){double e=fabs((double)a[i]-b[i]); if(e>max_abs)max_abs=e; sum+=e;}
        printf("sorted: n=%ld max_abs=%.3e mean_abs=%.3e  (~0 => same values, permuted layout)\n", na, max_abs, sum/na);
        return 0;
    }

    if (!strcmp(argv[1], "sub")) {   // write a-b to argv[4]
        long na, nb; float * a = load(argv[2], &na); float * b = load(argv[3], &nb);
        if (na != nb) { printf("SIZE MISMATCH\n"); return 1; }
        for (long i=0;i<na;++i) a[i] -= b[i];
        FILE* f = fopen(argv[4],"wb"); fwrite(a,4,na,f); fclose(f);
        printf("wrote %s (a-b, n=%ld)\n", argv[4], na);
        return 0;
    }

    fprintf(stderr, "unknown cmd %s\n", argv[1]);
    return 2;
}
