// CUDA Graph edge-insertion experiment. No timing nodes inside measured graphs.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

static void ck(cudaError_t e) { if(e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
__global__ void propagate(const float* input, const float* a, const float* b,
                          float* output, int n, int rounds) {
  for (int i=blockIdx.x*blockDim.x+threadIdx.x; i<n; i+=blockDim.x*gridDim.x) {
    float x=input[i]+fmaxf(a ? a[i] : 0.f, b ? b[i] : 0.f);
    for(int j=0;j<rounds;++j) x=fmaf(x,0.9999f,0.0001f);
    output[i]=x;
  }
}
struct Data {
  int gates,n;
  std::vector<int> p1,p2,rounds,blocks;
  std::vector<float> expected;
  float *host=nullptr,*input=nullptr,*output=nullptr;
  cudaStream_t stream=nullptr;
  ~Data(){if(stream)cudaStreamDestroy(stream); if(host)cudaFreeHost(host); if(input)cudaFree(input); if(output)cudaFree(output);}
};
struct Plan {
  cudaGraph_t graph=nullptr; cudaGraphExec_t exec=nullptr;
  cudaEvent_t begin=nullptr,end=nullptr; Data* data;
  ~Plan(){if(exec)cudaGraphExecDestroy(exec);if(graph)cudaGraphDestroy(graph);if(begin)cudaEventDestroy(begin);if(end)cudaEventDestroy(end);}
};
extern "C" void* data_create(int gates,int n,const int* p1,const int* p2,const int* rounds,const int* blocks) {
  try {
    auto d=std::make_unique<Data>();d->gates=gates;d->n=n;
    d->p1.assign(p1,p1+gates);d->p2.assign(p2,p2+gates);
    d->rounds.assign(rounds,rounds+gates);d->blocks.assign(blocks,blocks+gates);
    size_t bytes=size_t(gates)*n*sizeof(float);
    ck(cudaMallocHost(&d->host,bytes));ck(cudaMalloc(&d->input,bytes));ck(cudaMalloc(&d->output,bytes));
    ck(cudaStreamCreateWithFlags(&d->stream,cudaStreamNonBlocking));
    for(int g=0;g<gates;g++)for(int i=0;i<n;i++)d->host[size_t(g)*n+i]=float((i+g*13)%101)/1024.f;
    return d.release();
  }catch(const std::exception& e){fprintf(stderr,"data: %s\n",e.what());return nullptr;}
}
extern "C" void data_destroy(void* p){delete static_cast<Data*>(p);}
extern "C" void* graph_create(void* raw,int edge_count,const int* sources,const int* targets,const int* order,int only) {
  try {
    auto* d=static_cast<Data*>(raw);auto p=std::make_unique<Plan>();p->data=d;
    ck(cudaGraphCreate(&p->graph,0));std::vector<cudaGraphNode_t> nodes(2*d->gates);
    for(int k=0;k<2*d->gates;k++){
      int node=order[k];if(only>=0 && node!=only)continue;
      int g=node/2;size_t off=size_t(g)*d->n;
      if(node%2==0){
        ck(cudaGraphAddMemcpyNode1D(&nodes[node],p->graph,nullptr,0,d->input+off,d->host+off,size_t(d->n)*4,cudaMemcpyHostToDevice));
      }else{
        const float* in=d->input+off;
        const float* a=d->p1[g]<0?nullptr:d->output+size_t(d->p1[g])*d->n;
        const float* b=d->p2[g]<0?nullptr:d->output+size_t(d->p2[g])*d->n;
        float* out=d->output+off;int n=d->n,rounds=d->rounds[g];
        void* args[]={&in,&a,&b,&out,&n,&rounds};
        cudaKernelNodeParams params{};params.func=(void*)propagate;
        params.gridDim=dim3(d->blocks[g]);params.blockDim=dim3(256);params.kernelParams=args;
        ck(cudaGraphAddKernelNode(&nodes[node],p->graph,nullptr,0,&params));
      }
    }
    if(only<0)for(int i=0;i<edge_count;i++){
      ck(cudaGraphAddDependencies(p->graph,&nodes[sources[i]],&nodes[targets[i]],nullptr,1));
    }
    ck(cudaGraphInstantiateWithFlags(&p->exec,p->graph,0));
    ck(cudaEventCreate(&p->begin));ck(cudaEventCreate(&p->end));
    ck(cudaGraphUpload(p->exec,d->stream));ck(cudaStreamSynchronize(d->stream));return p.release();
  }catch(const std::exception& e){fprintf(stderr,"graph: %s\n",e.what());return nullptr;}
}
extern "C" void graph_destroy(void* p){delete static_cast<Plan*>(p);}
extern "C" double graph_run(void* raw,int repeat){
  try{auto& p=*static_cast<Plan*>(raw);auto s=p.data->stream;
    ck(cudaEventRecord(p.begin,s));for(int i=0;i<repeat;i++)ck(cudaGraphLaunch(p.exec,s));
    ck(cudaEventRecord(p.end,s));ck(cudaEventSynchronize(p.end));float ms;ck(cudaEventElapsedTime(&ms,p.begin,p.end));return ms/repeat;
  }catch(const std::exception& e){fprintf(stderr,"run: %s\n",e.what());return -1;}
}
extern "C" double graph_check(void* raw){
  try{auto& p=*static_cast<Plan*>(raw);auto& d=*p.data;size_t size=size_t(d.gates)*d.n;
    ck(cudaMemsetAsync(d.input,0xff,size*4,d.stream));ck(cudaMemsetAsync(d.output,0xff,size*4,d.stream));
    ck(cudaGraphLaunch(p.exec,d.stream));ck(cudaStreamSynchronize(d.stream));
    std::vector<float> observed(size);ck(cudaMemcpy(observed.data(),d.output,size*4,cudaMemcpyDeviceToHost));
    if(d.expected.empty()){
      d.expected.resize(size);
      for(int g=0;g<d.gates;g++)for(int i=0;i<d.n;i++){
      float a=d.p1[g]<0?0.f:d.expected[size_t(d.p1[g])*d.n+i];
      float b=d.p2[g]<0?0.f:d.expected[size_t(d.p2[g])*d.n+i];
      float x=d.host[size_t(g)*d.n+i]+fmaxf(a,b);
      for(int j=0;j<d.rounds[g];j++)x=fmaf(x,0.9999f,0.0001f);
      d.expected[size_t(g)*d.n+i]=x;
    }}
    double worst=0;
    for(size_t i=0;i<size;i++){
      float y=observed[i];if(!std::isfinite(y))return -2;
      worst=std::max(worst,double(fabsf(y-d.expected[i])));
    }return worst;
  }catch(const std::exception& e){fprintf(stderr,"check: %s\n",e.what());return -1;}
}
extern "C" int sm_count(){cudaDeviceProp p;ck(cudaGetDeviceProperties(&p,0));return p.multiProcessorCount;}
