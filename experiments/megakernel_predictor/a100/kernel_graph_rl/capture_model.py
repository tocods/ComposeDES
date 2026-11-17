"""Capture the full Qwen forward, retaining Q/K/V and gate/up independence."""
import sys
from pathlib import Path
_model_path=str(Path(__file__).resolve().parent.parent/'full_inference')
sys.path.insert(0,_model_path)
import torch
from model import Model,Request,attention
from transformers.models.qwen3.modeling_qwen3 import rotate_half
from runtime import capture
sys.path.remove(_model_path)

class Forward:
    def __init__(self,req,parallel=True,readback=False):
        self.req=req;self.parallel=parallel;self.readback=readback
        self.branches=[torch.cuda.Stream() for _ in range(3)]
        self.io=torch.cuda.Stream()
        self.forks=[torch.cuda.Event() for _ in range(72)]
        self.joins=[[torch.cuda.Event() for _ in range(3)] for _ in range(72)]
        self.io_start=torch.cuda.Event();self.ready=[torch.cuda.Event() for _ in range(36)]

    def fork(self,index,functions):
        if not self.parallel:return [f() for f in functions]
        current=torch.cuda.current_stream();self.forks[index].record(current);outputs=[]
        for j,fn in enumerate(functions):
            stream=self.branches[j]
            with torch.cuda.stream(stream):
                stream.wait_event(self.forks[index]);outputs.append(fn());self.joins[index][j].record(stream)
        for j in range(len(functions)):current.wait_event(self.joins[index][j])
        return outputs

    def __call__(self):
        r=self.req;m=r.model;n=r.suffix;end=r.prefix+n
        if self.readback:
            self.io_start.record()
            with torch.cuda.stream(self.io):
                self.io.wait_event(self.io_start)
                for i in range(m.layers):r.copy(i);self.ready[i].record()
        x=m.core.embed_tokens(r.ids_gpu[:,r.prefix:]);cos,sin=r.rope
        cos=cos.unsqueeze(1);sin=sin.unsqueeze(1)
        def rotary(t):return t*cos+rotate_half(t)*sin
        for i,layer in enumerate(m.core.layers):
            a=layer.self_attn;z=layer.input_layernorm(x)
            def query():
                q=a.q_norm(a.q_proj(z).view(1,n,m.heads,m.dim)).transpose(1,2)
                return rotary(q)[0].transpose(0,1).contiguous()
            def key():
                k=a.k_norm(a.k_proj(z).view(1,n,m.kv_heads,m.dim)).transpose(1,2)
                r.arena[i][0][r.prefix:end].copy_(rotary(k)[0].transpose(0,1))
            def value():r.arena[i][1][r.prefix:end].copy_(a.v_proj(z).view(n,m.kv_heads,m.dim))
            q,_,_=self.fork(2*i,[query,key,value])
            if self.readback:torch.cuda.current_stream().wait_event(self.ready[i])
            k,v=r.arena[i]
            out=attention(q,k[:end],v[:end],a.scaling)
            x=x+a.o_proj(out.reshape(1,n,m.heads*m.dim))
            z=layer.post_attention_layernorm(x)
            gate,up=self.fork(2*i+1,[lambda:layer.mlp.act_fn(layer.mlp.gate_proj(z)),lambda:layer.mlp.up_proj(z)])
            x=x+layer.mlp.down_proj(gate*up)
        r.logits.copy_(m.net.lm_head(m.core.norm(x[:,-1:])))
        r.next_token.copy_(r.logits.argmax(-1))
        if self.readback:torch.cuda.current_stream().wait_event(self.ready[-1])

    def capture(self):
        self.req.reset(resident=True)
        graph,stream=capture(self)
        self.stream=stream;self.graph=graph
        return graph
