// Fail-fast smoke test for batched prefill: does Vulkan ggml_gated_delta_net / ggml_ssm_conv
// with n_tokens=T produce the SAME result as T sequential single-token (n_tokens=1) calls?
// The streaming engine has only ever called these ops with n_tokens=1 (single-step recurrence),
// so the batched kernel path is unverified on this GPU. If batched != sequential, the single-op
// batched-GDN plan collapses and we'd need a hand-rolled scan. No model needed; synthetic tensors.
//   build:  scripts\build.ps1 src\batchgdn_test.cpp batchgdn_test
//   run from build\:  .\batchgdn_test.exe
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>

static ggml_backend_t backend=nullptr;
static long long now_ns(){ return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// Time the fused ggml_gated_delta_net op at n_tokens=T (single call). Diagnostic: is per-token cost FLAT
// (serial per-token scan -> a chunked-matmul form could beat it) or DECREASING with T (already parallel -> no win)?
static void time_gdn(int Sv,int Hv,int Hk,int T,int reps){
    std::mt19937 rng(7); std::normal_distribution<float> N(0.f,1.f); std::uniform_real_distribution<float> U(0.f,1.f);
    std::vector<float> Q((size_t)Sv*Hk*T),K((size_t)Sv*Hk*T),V((size_t)Sv*Hv*T),G((size_t)Hv*T),B((size_t)Hv*T),S0((size_t)Sv*Sv*Hv);
    for(auto&x:Q)x=0.1f*N(rng); for(auto&x:K)x=0.1f*N(rng); for(auto&x:V)x=0.3f*N(rng);
    for(auto&x:G)x=-0.2f*U(rng)-0.01f; for(auto&x:B)x=U(rng); for(auto&x:S0)x=0.05f*N(rng);
    ggml_context* g=ggml_init({(size_t)256*1024*1024,nullptr,true});
    ggml_tensor* q=ggml_new_tensor_4d(g,GGML_TYPE_F32,Sv,Hk,T,1); ggml_set_input(q);
    ggml_tensor* k=ggml_new_tensor_4d(g,GGML_TYPE_F32,Sv,Hk,T,1); ggml_set_input(k);
    ggml_tensor* v=ggml_new_tensor_4d(g,GGML_TYPE_F32,Sv,Hv,T,1); ggml_set_input(v);
    ggml_tensor* gg=ggml_new_tensor_4d(g,GGML_TYPE_F32,1,Hv,T,1); ggml_set_input(gg);
    ggml_tensor* bb=ggml_new_tensor_4d(g,GGML_TYPE_F32,1,Hv,T,1); ggml_set_input(bb);
    ggml_tensor* s0=ggml_new_tensor_4d(g,GGML_TYPE_F32,Sv,Sv,Hv,1); ggml_set_input(s0);
    ggml_tensor* r=ggml_gated_delta_net(g,q,k,v,gg,bb,s0,1); ggml_set_output(r);
    ggml_cgraph* gf=ggml_new_graph(g); ggml_build_forward_expand(gf,r);
    ggml_gallocr_t alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)); ggml_gallocr_alloc_graph(alloc,gf);
    ggml_backend_tensor_set(q,Q.data(),0,Q.size()*4); ggml_backend_tensor_set(k,K.data(),0,K.size()*4); ggml_backend_tensor_set(v,V.data(),0,V.size()*4);
    ggml_backend_tensor_set(gg,G.data(),0,G.size()*4); ggml_backend_tensor_set(bb,B.data(),0,B.size()*4); ggml_backend_tensor_set(s0,S0.data(),0,S0.size()*4);
    ggml_backend_graph_compute(backend,gf); ggml_backend_synchronize(backend);   // warmup (shader compile)
    long long t0=now_ns(); for(int i=0;i<reps;++i) ggml_backend_graph_compute(backend,gf); ggml_backend_synchronize(backend); long long t1=now_ns();
    double ms=(t1-t0)/1e6/reps;
    printf("  [GDN-time] T=%5d : %8.3f ms/call  %.5f ms/token  (Sv=%d Hv=%d Hk=%d)\n", T, ms, ms/T, Sv,Hv,Hk);
    ggml_gallocr_free(alloc); ggml_free(g);
}

// run a single graph that builds `result` from the given inputs, return result as host floats.
// inputs: list of (tensor, host data ptr, n_floats). The graph context owns all tensors.
struct In { ggml_tensor* t; const float* d; size_t n; };
static std::vector<float> run_graph(ggml_context* g, ggml_tensor* result, const std::vector<In>& ins){
    ggml_cgraph* gf=ggml_new_graph(g);
    ggml_build_forward_expand(gf,result);
    ggml_gallocr_t alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(alloc,gf);
    for(auto& in:ins) ggml_backend_tensor_set(in.t,in.d,0,in.n*sizeof(float));
    ggml_backend_graph_compute(backend,gf);
    std::vector<float> out((size_t)ggml_nelements(result));
    ggml_backend_tensor_get(result,out.data(),0,out.size()*sizeof(float));
    ggml_gallocr_free(alloc);
    return out;
}

static double maxabs_diff(const float* a,const float* b,size_t n,double* mref=nullptr){
    double m=0,mr=0; for(size_t i=0;i<n;++i){ double d=fabs((double)a[i]-(double)b[i]); if(d>m)m=d; double r=fabs((double)a[i]); if(r>mr)mr=r; }
    if(mref)*mref=mr; return m;
}

// ---- GDN: ggml_gated_delta_net batched(T) vs sequential T x (1) ----
static int test_gdn(int Sv,int Hv,int Hk,int T){
    printf("\n[GDN] Sv=%d Hv=%d Hk=%d T=%d\n",Sv,Hv,Hk,T);
    std::mt19937 rng(1234); std::normal_distribution<float> N(0.f,1.f); std::uniform_real_distribution<float> U(0.f,1.f);
    // synthetic per-token inputs (token-major t outer). q/k l2-ish, g negative (decay<1), beta in (0,1).
    auto L2=[&](std::vector<float>& v,int dim,int heads,int t){ // l2-normalize each [dim] head vector for token t
        for(int h=0;h<heads;++h){ double s=0; size_t base=((size_t)t*heads+h)*dim; for(int i=0;i<dim;++i)s+=(double)v[base+i]*v[base+i]; s=sqrt(s)+1e-6; for(int i=0;i<dim;++i)v[base+i]/=(float)s; } };
    std::vector<float> Q((size_t)Sv*Hk*T),K((size_t)Sv*Hk*T),V((size_t)Sv*Hv*T),G((size_t)Hv*T),B((size_t)Hv*T),S0((size_t)Sv*Sv*Hv);
    for(int t=0;t<T;++t){ for(size_t i=0;i<(size_t)Sv*Hk;++i){Q[(size_t)t*Sv*Hk+i]=N(rng);K[(size_t)t*Sv*Hk+i]=N(rng);} L2(Q,Sv,Hk,t); L2(K,Sv,Hk,t);
        for(size_t i=0;i<(size_t)Sv*Hv;++i)V[(size_t)t*Sv*Hv+i]=0.3f*N(rng);
        for(int h=0;h<Hv;++h){ G[(size_t)t*Hv+h]=-0.2f*U(rng)-0.01f; B[(size_t)t*Hv+h]=U(rng); } }
    for(size_t i=0;i<S0.size();++i)S0[i]=0.05f*N(rng);

    // BATCHED: one call, n_tokens=T
    std::vector<float> batched;
    { ggml_context* g=ggml_init({(size_t)64*1024*1024,nullptr,true});
      ggml_tensor* q=ggml_new_tensor_4d(g,GGML_TYPE_F32,Sv,Hk,T,1); ggml_set_input(q);
      ggml_tensor* k=ggml_new_tensor_4d(g,GGML_TYPE_F32,Sv,Hk,T,1); ggml_set_input(k);
      ggml_tensor* v=ggml_new_tensor_4d(g,GGML_TYPE_F32,Sv,Hv,T,1); ggml_set_input(v);
      ggml_tensor* gg=ggml_new_tensor_4d(g,GGML_TYPE_F32,1,Hv,T,1); ggml_set_input(gg);
      ggml_tensor* bb=ggml_new_tensor_4d(g,GGML_TYPE_F32,1,Hv,T,1); ggml_set_input(bb);
      ggml_tensor* s0=ggml_new_tensor_4d(g,GGML_TYPE_F32,Sv,Sv,Hv,1); ggml_set_input(s0);
      ggml_tensor* r=ggml_gated_delta_net(g,q,k,v,gg,bb,s0,1); ggml_set_output(r);
      batched=run_graph(g,r,{{q,Q.data(),Q.size()},{k,K.data(),K.size()},{v,V.data(),V.size()},{gg,G.data(),G.size()},{bb,B.data(),B.size()},{s0,S0.data(),S0.size()}});
      ggml_free(g); }
    // batched result layout [Sv*Hv, T + Sv]: rows 0..T-1 = per-token attn; rows T.. = final state
    const float* b_attn = batched.data();                          // [Sv*Hv * T]
    const float* b_state= batched.data()+(size_t)Sv*Hv*T;          // [Sv*Hv * Sv] = [Sv,Sv,Hv]

    // SEQUENTIAL: T calls n_tokens=1, threading the state
    std::vector<float> seq_attn((size_t)Sv*Hv*T); std::vector<float> state=S0;
    for(int t=0;t<T;++t){
        ggml_context* g=ggml_init({(size_t)32*1024*1024,nullptr,true});
        ggml_tensor* q=ggml_new_tensor_4d(g,GGML_TYPE_F32,Sv,Hk,1,1); ggml_set_input(q);
        ggml_tensor* k=ggml_new_tensor_4d(g,GGML_TYPE_F32,Sv,Hk,1,1); ggml_set_input(k);
        ggml_tensor* v=ggml_new_tensor_4d(g,GGML_TYPE_F32,Sv,Hv,1,1); ggml_set_input(v);
        ggml_tensor* gg=ggml_new_tensor_4d(g,GGML_TYPE_F32,1,Hv,1,1); ggml_set_input(gg);
        ggml_tensor* bb=ggml_new_tensor_4d(g,GGML_TYPE_F32,1,Hv,1,1); ggml_set_input(bb);
        ggml_tensor* s0=ggml_new_tensor_4d(g,GGML_TYPE_F32,Sv,Sv,Hv,1); ggml_set_input(s0);
        ggml_tensor* r=ggml_gated_delta_net(g,q,k,v,gg,bb,s0,1); ggml_set_output(r);
        std::vector<float> out=run_graph(g,r,{
            {q,Q.data()+(size_t)t*Sv*Hk,(size_t)Sv*Hk},{k,K.data()+(size_t)t*Sv*Hk,(size_t)Sv*Hk},
            {v,V.data()+(size_t)t*Sv*Hv,(size_t)Sv*Hv},{gg,G.data()+(size_t)t*Hv,(size_t)Hv},
            {bb,B.data()+(size_t)t*Hv,(size_t)Hv},{s0,state.data(),state.size()}});
        memcpy(seq_attn.data()+(size_t)t*Sv*Hv, out.data(), (size_t)Sv*Hv*sizeof(float)); // token attn
        memcpy(state.data(), out.data()+(size_t)Sv*Hv, (size_t)Sv*Sv*Hv*sizeof(float));   // carry state
        ggml_free(g);
    }
    double mr1,mr2;
    double da=maxabs_diff(b_attn,seq_attn.data(),(size_t)Sv*Hv*T,&mr1);
    double ds=maxabs_diff(b_state,state.data(),(size_t)Sv*Sv*Hv,&mr2);
    printf("  attn : max|d|=%.3e (rel %.2e of max|%.3e|)\n",da,da/(mr1+1e-30),mr1);
    printf("  state: max|d|=%.3e (rel %.2e of max|%.3e|)\n",ds,ds/(mr2+1e-30),mr2);
    bool ok = da < 1e-3*(mr1+1e-6) && ds < 1e-3*(mr2+1e-6);
    printf("  => %s\n", ok?"MATCH (batched==sequential)":"*** MISMATCH ***");
    return ok?0:1;
}

// ---- ssm_conv: batched(T) vs sequential rolling window ----
static int test_conv(int cd,int dconv,int T){
    printf("\n[CONV] conv_dim=%d d_conv=%d T=%d\n",cd,dconv,T);
    std::mt19937 rng(99); std::normal_distribution<float> N(0.f,1.f);
    // ggml is CHANNEL-MAJOR (ne[0] fastest). kernel c[d_conv,cd] -> c[ch*dconv+i];
    // sx[d_conv-1+T,cd] -> sx[ch*(dconv-1+T)+time]; state[d_conv-1,cd] -> st[ch*(dconv-1)+j]; tok[T,cd] -> tok[ch*T+t].
    int W2=dconv-1+T;
    std::vector<float> W((size_t)dconv*cd); for(int ch=0;ch<cd;++ch)for(int i=0;i<dconv;++i)W[(size_t)ch*dconv+i]=0.2f*N(rng);
    std::vector<float> st0((size_t)(dconv-1)*cd); for(int ch=0;ch<cd;++ch)for(int j=0;j<dconv-1;++j)st0[(size_t)ch*(dconv-1)+j]=0.1f*N(rng);
    std::vector<float> tok((size_t)T*cd); for(int ch=0;ch<cd;++ch)for(int t=0;t<T;++t)tok[(size_t)ch*T+t]=N(rng);

    // BATCHED: sx=[d_conv-1+T, cd, 1], channel-major
    std::vector<float> sx((size_t)W2*cd);
    for(int ch=0;ch<cd;++ch){ for(int j=0;j<dconv-1;++j) sx[(size_t)ch*W2+j]=st0[(size_t)ch*(dconv-1)+j];
        for(int t=0;t<T;++t) sx[(size_t)ch*W2+(dconv-1)+t]=tok[(size_t)ch*T+t]; }
    std::vector<float> batched;
    { ggml_context* g=ggml_init({(size_t)32*1024*1024,nullptr,true});
      ggml_tensor* sxt=ggml_new_tensor_3d(g,GGML_TYPE_F32,W2,cd,1); ggml_set_input(sxt);
      ggml_tensor* c=ggml_new_tensor_2d(g,GGML_TYPE_F32,dconv,cd); ggml_set_input(c);
      ggml_tensor* r=ggml_ssm_conv(g,sxt,c); ggml_set_output(r);               // [cd, T, 1] -> result[t*cd+ch]
      batched=run_graph(g,r,{{sxt,sx.data(),sx.size()},{c,W.data(),W.size()}});
      ggml_free(g); }

    // SEQUENTIAL: roll a [d_conv-1] window per token (channel-major)
    std::vector<float> seq((size_t)cd*T); std::vector<float> win=st0; // [d_conv-1, cd]
    for(int t=0;t<T;++t){
        std::vector<float> sx1((size_t)dconv*cd);
        for(int ch=0;ch<cd;++ch){ for(int j=0;j<dconv-1;++j) sx1[(size_t)ch*dconv+j]=win[(size_t)ch*(dconv-1)+j];
            sx1[(size_t)ch*dconv+(dconv-1)]=tok[(size_t)ch*T+t]; }                                 // append token t
        ggml_context* g=ggml_init({(size_t)16*1024*1024,nullptr,true});
        ggml_tensor* sxt=ggml_new_tensor_3d(g,GGML_TYPE_F32,dconv,cd,1); ggml_set_input(sxt);
        ggml_tensor* c=ggml_new_tensor_2d(g,GGML_TYPE_F32,dconv,cd); ggml_set_input(c);
        ggml_tensor* r=ggml_ssm_conv(g,sxt,c); ggml_set_output(r);                          // [cd,1,1]
        std::vector<float> out=run_graph(g,r,{{sxt,sx1.data(),sx1.size()},{c,W.data(),W.size()}});
        memcpy(seq.data()+(size_t)t*cd, out.data(), (size_t)cd*sizeof(float));
        // roll window: keep last d_conv-1 of [win ++ token_t] = drop oldest column
        for(int ch=0;ch<cd;++ch){ for(int j=0;j<dconv-2;++j) win[(size_t)ch*(dconv-1)+j]=win[(size_t)ch*(dconv-1)+j+1];
            win[(size_t)ch*(dconv-1)+(dconv-2)]=tok[(size_t)ch*T+t]; }
        ggml_free(g);
    }
    double mr; double d=maxabs_diff(batched.data(),seq.data(),(size_t)cd*T,&mr);
    printf("  conv : max|d|=%.3e (rel %.2e of max|%.3e|)\n",d,d/(mr+1e-30),mr);
    bool ok = d < 1e-3*(mr+1e-6);
    printf("  => %s\n", ok?"MATCH (batched==sequential rolling)":"*** MISMATCH ***");
    return ok?0:1;
}

int main(){
    ggml_backend_load_all();
    ggml_backend_dev_t vk0=nullptr;
    for(size_t i=0;i<ggml_backend_dev_count();++i){ ggml_backend_dev_t d=ggml_backend_dev_get(i); if(!strcmp(ggml_backend_dev_name(d),"Vulkan0")){vk0=d;break;} }
    if(!vk0){ printf("Vulkan0 not found\n"); return 2; }
    backend=ggml_backend_dev_init(vk0,nullptr);
    printf("backend = %s\n", ggml_backend_name(backend));
    int fail=0;
    fail += test_gdn(128,32,16,1);   // T=1 sanity (must match the per-position path trivially)
    fail += test_gdn(128,32,16,2);
    fail += test_gdn(128,32,16,4);
    fail += test_gdn(128,32,16,16);
    fail += test_conv(256,4,1);
    fail += test_conv(256,4,5);
    fail += test_conv(256,4,17);
    printf("\n[GDN scaling — flat ms/token = serial per-token scan; decreasing = parallel/chunked]\n");
    time_gdn(128,32,16,64,50); time_gdn(128,32,16,256,30); time_gdn(128,32,16,1024,10);
    time_gdn(128,32,16,2048,6); time_gdn(128,32,16,4096,4);
    printf("\n==== %s (%d failing) ====\n", fail?"FAIL":"ALL PASS", fail);
    return fail?1:0;
}
