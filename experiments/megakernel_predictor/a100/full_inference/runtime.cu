// Reuse the audited kernel/memcpy transplant and graph lifetime implementation.
#include "../research_probes/explicit_graph.cu"
#include <unordered_set>

extern "C" void* full_create(int count,const uintptr_t* bodies,const int* order,
                             const int* offsets,const int* prereqs){
  try {
    auto p=std::make_unique<Plan>();check(cudaGraphCreate(&p->graph,0));
    check(cudaEventCreate(&p->begin));check(cudaEventCreate(&p->end));
    std::vector<cudaGraphNode_t> ends(count);
    std::vector<std::vector<cudaGraphNode_t>> roots(count);
    std::unordered_set<cudaGraphNode_t> known;
    // Creation order is independent of auxiliary edges: add all edges afterward.
    for(int k=0;k<count;k++){
      int i=order[k];ends[i]=append(p->graph,reinterpret_cast<cudaGraph_t>(bodies[i]),{});
      size_t n=0;check(cudaGraphGetRootNodes(p->graph,nullptr,&n));
      std::vector<cudaGraphNode_t> all(n);check(cudaGraphGetRootNodes(p->graph,all.data(),&n));
      for(auto root:all)if(known.insert(root).second)roots[i].push_back(root);
    }
    for(int i=0;i<count;i++)for(int j=offsets[i];j<offsets[i+1];j++)for(auto root:roots[i])
      check(cudaGraphAddDependencies(p->graph,&ends[prereqs[j]],&root,nullptr,1));
    check(cudaGraphInstantiateWithFlags(&p->exec,p->graph,0));
    return p.release();
  }catch(const std::exception& e){std::fprintf(stderr,"full_create: %s\n",e.what());return nullptr;}
}
extern "C" int full_launch(void* raw,uintptr_t stream){
  try{check(cudaGraphLaunch(static_cast<Plan*>(raw)->exec,reinterpret_cast<cudaStream_t>(stream)));return 0;}
  catch(const std::exception& e){std::fprintf(stderr,"full_launch: %s\n",e.what());return 1;}
}
