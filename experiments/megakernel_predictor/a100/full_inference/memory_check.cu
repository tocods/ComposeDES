// Standalone CUDA memory check: no PyTorch, FlashInfer, weights, or CUDA Graphs.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <thread>
#include <chrono>
#define CHECK(x) do {auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));return 2;}} while(0)
__global__ void fill(unsigned* p,size_t n,unsigned salt){
  for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x)p[i]=unsigned(i)^salt;
}
__global__ void verify(const unsigned* p,size_t n,unsigned salt,unsigned long long* bad){
  for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x)
    if(p[i]!=(unsigned(i)^salt)){atomicAdd(bad,1ULL);atomicMin(bad+1,(unsigned long long)i);}
}
int main(){
  const size_t bytes=256ULL<<20,n=bytes/sizeof(unsigned);const int chunks=80;
  std::vector<unsigned*> buffers(chunks);unsigned long long* bad;
  CHECK(cudaMalloc(&bad,2*sizeof(*bad)));
  for(int k=0;k<chunks;k++){
    CHECK(cudaMalloc(&buffers[k],bytes));fill<<<1024,256>>>(buffers[k],n,0xa5a50000u+k);
  }
  CHECK(cudaDeviceSynchronize());bool failed=false;
  for(int repeat=0;repeat<4;repeat++){
    std::this_thread::sleep_for(std::chrono::seconds(1));unsigned long long total=0;
    for(int k=0;k<chunks;k++){
      CHECK(cudaMemset(bad,0,sizeof(*bad)));CHECK(cudaMemset(bad+1,255,sizeof(*bad)));
      verify<<<1024,256>>>(buffers[k],n,0xa5a50000u+k,bad);
      unsigned long long stats[2];CHECK(cudaMemcpy(stats,bad,sizeof(stats),cudaMemcpyDeviceToHost));
      if(stats[0]){
        unsigned actual;CHECK(cudaMemcpy(&actual,buffers[k]+stats[1],sizeof(actual),cudaMemcpyDeviceToHost));
        printf("round=%d chunk=%d address=%p mismatched_words=%llu first_index=%llu expected=0x%x cpu_read=0x%x\n",
               repeat,k,(void*)buffers[k],stats[0],stats[1],unsigned(stats[1])^(0xa5a50000u+k),actual);
      }
      total+=stats[0];
    }
    printf("round=%d verified_GiB=20 total_mismatched_words=%llu\n",repeat,total);fflush(stdout);
    failed|=total!=0;
  }
  for(auto p:buffers)CHECK(cudaFree(p));CHECK(cudaFree(bad));return failed?1:0;
}
