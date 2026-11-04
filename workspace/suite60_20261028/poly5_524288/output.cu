
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>
#include <cstdint>
#define CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess) { fprintf(stderr,"%s\n",cudaGetErrorString(e)); exit(2); } } while(0)
uint32_t mix(uint32_t x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; return x ^ (x >> 16); }
__global__ void work(float *C, const float *A, const float *B, int N) {
int i = blockIdx.x * blockDim.x + threadIdx.x;
 if (i < N) { float value=0.125f; for(int k=0;k<5;k++) {
 value = value * A[i] + B[i]; } C[i]=value; }
}
int main(int argc,char **argv) {
    int N=524288, seed=0;
    const char *output=nullptr;
    for(int arg=1;arg<argc;++arg) {
        if(!strcmp(argv[arg],"--output") && arg+1<argc) output=argv[++arg];
        else if(!strcmp(argv[arg],"--size") && arg+1<argc) N=atoi(argv[++arg]);
        else if(!strcmp(argv[arg],"--seed") && arg+1<argc) seed=atoi(argv[++arg]);
    }
    size_t inputs=N, outputs=N;
    std::vector<float> A(inputs),B(inputs),C(outputs);
    for(size_t j=0;j<inputs;++j) {
        A[j]=float(int(mix(uint32_t(j)+uint32_t(seed)*1000003u+17u)%2048u)-1024)/1024.0f;
        B[j]=float(int(mix(uint32_t(j)+uint32_t(seed)*2000003u+991u)%2048u)-1024)/1024.0f;
    }
    float *a,*b,*c;
    CHECK(cudaMalloc(&a,inputs*sizeof(float))); CHECK(cudaMalloc(&b,inputs*sizeof(float)));
    CHECK(cudaMalloc(&c,outputs*sizeof(float)));
    CHECK(cudaMemcpy(a,A.data(),inputs*sizeof(float),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(b,B.data(),inputs*sizeof(float),cudaMemcpyHostToDevice));
    CHECK(cudaMemset(c,255,outputs*sizeof(float)));
    dim3 block(128), grid((N+127)/128);
    for(int w=0;w<5;++w) work<<<grid,block>>>(c,a,b,N);
    CHECK(cudaDeviceSynchronize());
    cudaEvent_t start,stop;
    CHECK(cudaEventCreate(&start)); CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    for(int repeat=0;repeat<30;++repeat) work<<<grid,block>>>(c,a,b,N);
    CHECK(cudaEventRecord(stop)); CHECK(cudaEventSynchronize(stop)); CHECK(cudaGetLastError());
    float elapsed; CHECK(cudaEventElapsedTime(&elapsed,start,stop));
    CHECK(cudaMemcpy(C.data(),c,outputs*sizeof(float),cudaMemcpyDeviceToHost));
    if(output) {
        FILE *f=fopen(output,"wb"); if(!f) return 3;
        size_t written=fwrite(C.data(),sizeof(float),outputs,f); fclose(f);
        if(written!=outputs) return 4;
    }
    CHECK(cudaFree(a)); CHECK(cudaFree(b)); CHECK(cudaFree(c));
    CHECK(cudaEventDestroy(start)); CHECK(cudaEventDestroy(stop));
    printf("%.9f\n",elapsed/30.0f);
    return 0;
}
