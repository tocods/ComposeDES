// Compiled discrete-event resource model used by the ComposeDES predictor.
#include <vector>
#include <algorithm>
#include <cmath>
#include <queue>
extern "C" double admission(int n,int m,const int* from,const int* to,const int* kind,
 const double* duration,const double* demand,double fraction,double alpha,double overhead,double* ends){
  std::vector<std::vector<int>> children(n);std::vector<int> indegree(n);
  for(int i=0;i<m;i++){children[from[i]].push_back(to[i]);indegree[to[i]]++;}
  std::vector<int> ready,active;std::vector<double> left(n),need(n);double now=0,used=0;int completed=0,copies=0;
  for(int i=0;i<n;i++){if(!indegree[i])ready.push_back(i);need[i]=std::min(1.,demand[i]*fraction);}
  while(completed<n){
    bool admitted=true;
    while(admitted){
      admitted=false;std::sort(ready.begin(),ready.end());
      for(size_t pos=0;pos<ready.size();pos++){
        int u=ready[pos];bool copy=kind[u]==1;
        if(copy?copies!=0:used+need[u]>1.00000001)continue;
        ready.erase(ready.begin()+pos);left[u]=std::max(.00001,duration[u]-(copy?0:overhead));
        if(copy)copies++;else used+=need[u];active.push_back(u);admitted=true;break;
      }
    }
    if(active.empty())return -1.;
    double slowdown=1+alpha*std::max(0,int(active.size())-1),dt=1e100;
    for(int u:active)dt=std::min(dt,left[u]*slowdown);
    now+=dt;std::vector<int> keep;
    for(int u:active){
      left[u]-=dt/slowdown;
      if(left[u]>1e-10){keep.push_back(u);continue;}
      if(ends)ends[u]=now;completed++;if(kind[u]==1)copies--;else used-=need[u];
      for(int v:children[u])if(!--indegree[v])ready.push_back(v);
    }
    active.swap(keep);
  }return now;
}
