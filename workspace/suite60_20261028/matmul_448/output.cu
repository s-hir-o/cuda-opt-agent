
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
    __shared__ float A_s[16][16];
    __shared__ float B_s[16][16];

    int row = blockIdx.y * 16 + threadIdx.y;
    int col = blockIdx.x * 16 + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < N; t += 16) {
        if (row < N && (t + threadIdx.x) < N)
            A_s[threadIdx.y][threadIdx.x] = A[row * N + t + threadIdx.x];
        else
            A_s[threadIdx.y][threadIdx.x] = 0.0f;

        if ((t + threadIdx.y) < N && col < N)
            B_s[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * N + col];
        else
            B_s[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < 16; k++) {
            sum += A_s[threadIdx.y][k] * B_s[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;

}
int main(int argc,char **argv) {
    int N=448, seed=0;
    const char *output=nullptr;
    for(int arg=1;arg<argc;++arg) {
        if(!strcmp(argv[arg],"--output") && arg+1<argc) output=argv[++arg];
        else if(!strcmp(argv[arg],"--size") && arg+1<argc) N=atoi(argv[++arg]);
        else if(!strcmp(argv[arg],"--seed") && arg+1<argc) seed=atoi(argv[++arg]);
    }
    size_t inputs=size_t(N)*N, outputs=size_t(N)*N;
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
    dim3 block(16,16), grid((N+15)/16,(N+15)/16);
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
