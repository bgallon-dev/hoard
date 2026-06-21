// Bisect WHICH part of the engine environment halves NVMe throughput.
// Same persistent cv-pool, but measure GB/s after each init stage:
//   stage0 = bare pool ; stage1 = +ggml_backend_load_all ; stage2 = +Vulkan0 device init.
//   pooltest2 <file> <nworkers> <total_reads>
#include "ggml.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <windows.h>
#include <time.h>

struct IOJob { uint8_t* dst; uint64_t aoff; uint32_t asize; uint32_t head; uint32_t rsize; };
struct DirectIO {
    int n=0; std::vector<HANDLE> h; uint64_t fsize=0; std::string fpath;
    std::vector<std::thread> pool;
    std::mutex m; std::condition_variable cv_work, cv_done;
    std::vector<IOJob>* cur=nullptr; int batch_total=0; uint64_t gen=0; bool stop=false;
    std::atomic<int> next_idx{0}, done_cnt{0}, ready{0};
    void init(const char* p,int nn){
        n=nn; fpath=p;
        HANDLE q=CreateFileA(p,GENERIC_READ,FILE_SHARE_READ,NULL,OPEN_EXISTING,0,NULL);
        LARGE_INTEGER li; GetFileSizeEx(q,&li); fsize=(uint64_t)li.QuadPart; CloseHandle(q);
        h.assign(n,INVALID_HANDLE_VALUE);
        for(int w=0;w<n;++w) pool.emplace_back([this,w]{ worker(w); });
        while(ready.load()<n) std::this_thread::yield();
    }
    void doread(int w,const IOJob& j){
        if(j.aoff+j.asize<=fsize){ LARGE_INTEGER li; li.QuadPart=(LONGLONG)j.aoff;
            SetFilePointerEx(h[w],li,NULL,FILE_BEGIN); DWORD got=0; ReadFile(h[w],j.dst,j.asize,&got,NULL); }
    }
    void worker(int w){
        h[w]=CreateFileA(fpath.c_str(),GENERIC_READ,FILE_SHARE_READ,NULL,OPEN_EXISTING,FILE_FLAG_NO_BUFFERING|FILE_FLAG_RANDOM_ACCESS,NULL);
        ready.fetch_add(1);
        uint64_t seen=0;
        for(;;){
            { std::unique_lock<std::mutex> lk(m); cv_work.wait(lk,[&]{return stop||gen!=seen;}); if(stop) return; seen=gen; }
            for(;;){ int j=next_idx.fetch_add(1); if(j>=batch_total) break; doread(w,(*cur)[j]); }
            if(done_cnt.fetch_add(1)+1==n){ { std::lock_guard<std::mutex> lk(m); } cv_done.notify_one(); }
        }
    }
    void run(std::vector<IOJob>& jobs){
        if(jobs.empty()) return; int total=(int)jobs.size();
        { std::lock_guard<std::mutex> lk(m); cur=&jobs; batch_total=total; next_idx.store(0); done_cnt.store(0); ++gen; }
        cv_work.notify_all();
        { std::unique_lock<std::mutex> lk(m); cv_done.wait(lk,[&]{return done_cnt.load()==n;}); }
    }
    ~DirectIO(){ { std::lock_guard<std::mutex> lk(m); stop=true; ++gen; } cv_work.notify_all(); for(auto&t:pool) if(t.joinable()) t.join(); }
};

int main(int argc,char**argv){
    if(argc<4){ fprintf(stderr,"usage: %s file nworkers total_reads\n",argv[0]); return 2; }
    DirectIO io; io.init(argv[1],atoi(argv[2]));
    int M=atoi(argv[3]); const uint32_t BS=524288; uint64_t maxoff=io.fsize-BS-4096;
    std::vector<uint8_t*> bufs(M); std::vector<IOJob> all(M);
    for(int i=0;i<M;++i) bufs[i]=(uint8_t*)VirtualAlloc(NULL,BS,MEM_COMMIT|MEM_RESERVE,PAGE_READWRITE);
    auto bw=[&](unsigned seed)->double{
        for(int i=0;i<M;++i){ seed=seed*1103515245u+12345u; uint64_t off=(((uint64_t)seed*4096ull)%maxoff)&~(uint64_t)4095; all[i]=IOJob{bufs[i],off,BS,0,BS}; }
        struct timespec a,c; clock_gettime(CLOCK_MONOTONIC,&a); io.run(all);
        clock_gettime(CLOCK_MONOTONIC,&c); double sec=(c.tv_sec-a.tv_sec)+(c.tv_nsec-a.tv_nsec)/1e9; return (double)M*BS/1e9/sec; };
    printf("stage0  bare pool                : %.2f GB/s\n", bw(11u));
    ggml_backend_load_all();
    printf("stage1  +ggml_backend_load_all   : %.2f GB/s\n", bw(13u));
    ggml_backend_dev_t vk0=nullptr;
    for(size_t i=0;i<ggml_backend_dev_count();++i){ ggml_backend_dev_t d=ggml_backend_dev_get(i); if(!strcmp(ggml_backend_dev_name(d),"Vulkan0")){vk0=d;break;} }
    if(vk0){ ggml_backend_t b=ggml_backend_dev_init(vk0,nullptr);
        printf("stage2  +Vulkan0 device init     : %.2f GB/s\n", bw(17u));
        // stage3: accumulate VRAM allocations to ~6GB (1GB chunks), mimicking a model resident on the 8GB card
        ggml_backend_buffer_type_t bt=ggml_backend_get_default_buffer_type(b);
        std::vector<ggml_backend_buffer_t> held;
        for(int gb=1; gb<=6; ++gb){
            ggml_backend_buffer_t vbuf=ggml_backend_buft_alloc_buffer(bt,(size_t)1024*1024*1024);
            if(!vbuf){ printf("stage3  alloc failed at %dGB cumulative (VRAM full)\n",gb); break; }
            held.push_back(vbuf);
            printf("stage3  %dGB VRAM resident       : %.2f GB/s\n", gb, bw((unsigned)(23+gb)));
        }
        for(auto v:held) ggml_backend_buffer_free(v);
        // stage4: fill the OS page cache with ~6GB of the file via a buffered handle (exactly what load_model's H.f does)
        { FILE* f=fopen(argv[1],"rb"); std::vector<char> tmp(16*1024*1024); size_t tot=0,cap=(size_t)6*1024*1024*1024;
          while(tot<cap){ size_t r=fread(tmp.data(),1,tmp.size(),f); if(!r)break; tot+=r; }
          printf("stage4  +%.1fGB file in page cache: %.2f GB/s  (buffered handle OPEN)\n", tot/1e9, bw(31u));
          if(f) fclose(f);
          printf("stage4  same, buffered handle CLOSED: %.2f GB/s\n", bw(33u)); }
    }
    else printf("stage2  Vulkan0 not found\n");
    return 0;
}
