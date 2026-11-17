#include "../research_probes/explicit_graph.cu"
#include <fstream>
#include <sstream>
#include <iomanip>

std::vector<cudaGraphNode_t> getnodes(cudaGraph_t g){
  size_t n=0;check(cudaGraphGetNodes(g,nullptr,&n));std::vector<cudaGraphNode_t> v(n);
  check(cudaGraphGetNodes(g,v.data(),&n));return v;
}
std::string quote(const std::string& s){
  std::ostringstream o;o<<'"';for(unsigned char c:s){
    if(c=='"'||c=='\\')o<<'\\'<<c;else if(c<32)o<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<int(c)<<std::dec;else o<<c;
  }o<<'"';return o.str();
}
extern "C" int kg_export(uintptr_t raw,const char* path){
  try{
    auto g=reinterpret_cast<cudaGraph_t>(raw);auto nodes=getnodes(g);std::unordered_map<cudaGraphNode_t,int> ids;
    for(int i=0;i<nodes.size();i++)ids[nodes[i]]=i;
    std::ofstream out(path);out<<"{\"nodes\":[";
    for(int i=0;i<nodes.size();i++){
      cudaGraphNodeType type;check(cudaGraphNodeGetType(nodes[i],&type));
      int threads=0,blocks=0,regs=0,shared=0,resident=0;size_t bytes=0;std::string name;
      if(type==cudaGraphNodeTypeKernel){
        CUDA_KERNEL_NODE_PARAMS p{};check_driver(cuGraphKernelNodeGetParams(nodes[i],&p));
        threads=p.blockDimX*p.blockDimY*p.blockDimZ;blocks=p.gridDimX*p.gridDimY*p.gridDimZ;shared=p.sharedMemBytes;
        const char* text=nullptr;check_driver(cuFuncGetName(&text,p.func));if(text)name=text;
        check_driver(cuFuncGetAttribute(&regs,CU_FUNC_ATTRIBUTE_NUM_REGS,p.func));
        check_driver(cuOccupancyMaxActiveBlocksPerMultiprocessor(&resident,p.func,threads,shared));
      }else if(type==cudaGraphNodeTypeMemcpy){
        cudaMemcpy3DParms p{};check(cudaGraphMemcpyNodeGetParams(nodes[i],&p));bytes=p.extent.width*std::max(size_t(1),p.extent.height)*std::max(size_t(1),p.extent.depth);name="memcpy";
      }else if(type==cudaGraphNodeTypeMemset){name="memset";}else name="control";
      if(i)out<<',';
      out<<"{\"id\":"<<i<<",\"type\":"<<int(type)<<",\"name\":"<<quote(name)<<",\"threads\":"<<threads
         <<",\"blocks\":"<<blocks<<",\"registers\":"<<regs<<",\"dynamic_shared\":"<<shared<<",\"resident_blocks_per_sm\":"<<resident<<",\"copy_bytes\":"<<bytes<<'}';
    }
    out<<"],\"edges\":[";bool first=true;
    for(int v=0;v<nodes.size();v++){
      size_t n=0;check(cudaGraphNodeGetDependencies(nodes[v],nullptr,nullptr,&n));std::vector<cudaGraphNode_t> deps(n);
      check(cudaGraphNodeGetDependencies(nodes[v],deps.data(),nullptr,&n));
      for(auto u:deps){if(!first)out<<',';first=false;out<<'['<<ids.at(u)<<','<<v<<']';}
    }out<<"]}";return 0;
  }catch(const std::exception& e){fprintf(stderr,"export: %s\n",e.what());return 1;}
}
extern "C" void* kg_create(uintptr_t raw,int num,const int* from,const int* to){
  try{
    auto src=reinterpret_cast<cudaGraph_t>(raw);auto originals=getnodes(src);auto p=std::make_unique<Plan>();
    check(cudaGraphClone(&p->graph,src));check(cudaEventCreate(&p->begin));check(cudaEventCreate(&p->end));
    for(int i=0;i<num;i++){
      cudaGraphNode_t u,v;check(cudaGraphNodeFindInClone(&u,originals.at(from[i]),p->graph));
      check(cudaGraphNodeFindInClone(&v,originals.at(to[i]),p->graph));check(cudaGraphAddDependencies(p->graph,&u,&v,nullptr,1));
    }
    check(cudaGraphInstantiateWithFlags(&p->exec,p->graph,0));return p.release();
  }catch(const std::exception& e){fprintf(stderr,"clone: %s\n",e.what());return nullptr;}
}
extern "C" int kg_time(void* raw,uintptr_t stream,int replays,float* elapsed){
  try{
    auto p=static_cast<Plan*>(raw);auto s=reinterpret_cast<cudaStream_t>(stream);
    check(cudaGraphLaunch(p->exec,s));check(cudaEventRecord(p->begin,s));
    for(int i=0;i<replays;i++)check(cudaGraphLaunch(p->exec,s));
    check(cudaEventRecord(p->end,s));check(cudaEventSynchronize(p->end));check(cudaEventElapsedTime(elapsed,p->begin,p->end));
    *elapsed/=replays;return 0;
  }catch(const std::exception& e){fprintf(stderr,"time: %s\n",e.what());return 1;}
}
extern "C" int kg_launch(void* raw,uintptr_t stream){return int(cudaGraphLaunch(static_cast<Plan*>(raw)->exec,reinterpret_cast<cudaStream_t>(stream)));}
extern "C" int kg_profile(uintptr_t raw,uintptr_t stream,float* times){
  try{
    auto s=reinterpret_cast<cudaStream_t>(stream);auto nodes=getnodes(reinterpret_cast<cudaGraph_t>(raw));
    for(int i=0;i<nodes.size();i++){
      cudaGraphNodeType t;check(cudaGraphNodeGetType(nodes[i],&t));times[i]=0;
      if(t!=cudaGraphNodeTypeKernel&&t!=cudaGraphNodeTypeMemcpy&&t!=cudaGraphNodeTypeMemset)continue;
      Plan p;check(cudaGraphCreate(&p.graph,0));transplant(p.graph,nodes[i],{});
      check(cudaGraphInstantiateWithFlags(&p.exec,p.graph,0));check(cudaEventCreate(&p.begin));check(cudaEventCreate(&p.end));
      if(kg_time(&p,stream,5,times+i))throw std::runtime_error("node profiling failed");
    }return 0;
  }catch(const std::exception& e){fprintf(stderr,"profile: %s\n",e.what());return 1;}
}
