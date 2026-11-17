// Independent check: CPU generates and verifies every word; no custom GPU kernel.
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <thread>
#include <chrono>
#define CHECK(x) do {auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));return 2;}} while(0)
int main(){
  const size_t bytes=256ULL<<20,n=bytes/sizeof(unsigned);const int chunks=80;
  unsigned* host;CHECK(cudaMallocHost(&host,bytes));std::vector<unsigned*> buffers(chunks);
  for(int k=0;k<chunks;k++){
    CHECK(cudaMalloc(&buffers[k],bytes));
    for(size_t i=0;i<n;i++)host[i]=unsigned(i)^(0xa5a50000u+k);
    CHECK(cudaMemcpy(buffers[k],host,bytes,cudaMemcpyHostToDevice));
  }
  std::this_thread::sleep_for(std::chrono::seconds(1));bool failed=false;
  for(int round=0;round<2;round++){
    unsigned long long total=0;
    for(int k=0;k<chunks;k++){
      CHECK(cudaMemcpy(host,buffers[k],bytes,cudaMemcpyDeviceToHost));
      size_t first=n;unsigned long long count=0;
      for(size_t i=0;i<n;i++)if(host[i]!=(unsigned(i)^(0xa5a50000u+k))){if(first==n)first=i;count++;}
      if(count)printf("round=%d chunk=%d mismatched_words=%llu first_index=%zu expected=0x%x cpu_read=0x%x\n",
                      round,k,count,first,unsigned(first)^(0xa5a50000u+k),host[first]);
      total+=count;
    }
    printf("round=%d CPU_verified_GiB=20 total_mismatched_words=%llu\n",round,total);fflush(stdout);failed|=total!=0;
  }
  for(auto p:buffers)CHECK(cudaFree(p));CHECK(cudaFreeHost(host));return failed?1:0;
}
