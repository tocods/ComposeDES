"""All 36 Qwen3-8B layers, real weights, complete logits and autoregressive decode."""
import hashlib
from pathlib import Path
import random
import torch
from flashinfer.prefill import single_prefill_with_kv_cache
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import ALL_ATTENTION_FUNCTIONS,apply_rotary_pos_emb
from runtime import capture


def attention(q,k,v,scale=None):
    return single_prefill_with_kv_cache(q,k,v,causal=True,backend='fa2',sm_scale=scale)


def framework_attention(module,q,k,v,mask,**kwargs):
    assert q.shape[0]==1
    out=attention(q[0].transpose(0,1).contiguous(),k[0].transpose(0,1).contiguous(),
                  v[0].transpose(0,1).contiguous(),module.scaling)
    return out.unsqueeze(0),None


TOPICS=[
 'GPU execution combines computation, memory access, and synchronization. ',
 'A directed acyclic graph expresses the dependencies between operations. ',
 'A cache stores reusable data and avoids repeated computation. ',
 'Measurements must distinguish changes in the workload from scheduling effects. ',
 'An experiment compares identical inputs and validates numerical outputs. ',
 'A language model predicts the next token using the preceding context. ',
]


class Model:
    def __init__(self,path):
        ALL_ATTENTION_FUNCTIONS.register('full_model_fa2',framework_attention)
        self.net=AutoModelForCausalLM.from_pretrained(path,dtype=torch.bfloat16,
                    attn_implementation='full_model_fa2').cuda().eval()
        self.tokenizer=AutoTokenizer.from_pretrained(path)
        self.core=self.net.model;cfg=self.net.config
        self.layers=cfg.num_hidden_layers;self.heads=cfg.num_attention_heads
        self.kv_heads=cfg.num_key_value_heads;self.dim=cfg.head_dim;self.hidden=cfg.hidden_size
        assert self.layers==36 and len(self.core.layers)==36
        self.params=sum(p.numel() for p in self.net.parameters())

    def prompt(self,prefix,suffix,seed):
        rng=random.Random(seed)
        text='Research notes. '+''.join(rng.choices(TOPICS,k=(prefix+suffix)//8+100))
        ids=self.tokenizer.apply_chat_template([dict(role='user',content=text+
            '\nExplain the relationship between cache reuse and GPU scheduling, and give two concrete examples. Answer in detail.')],
            tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False)
        n=prefix+suffix
        tail=min(64,suffix)
        assert len(ids)>=n
        return ids[:n-tail]+ids[-tail:]

    def dense_forward(self,ids,arena,start,before_attention=None):
        count=ids.numel();end=start+count
        x=self.core.embed_tokens(ids.view(1,-1))
        pos=torch.arange(start,end,device='cuda').unsqueeze(0)
        rope=self.core.rotary_emb(x,pos)
        for i,(layer,(k,v)) in enumerate(zip(self.core.layers,arena)):
            a=layer.self_attn;z=layer.input_layernorm(x)
            q=a.q_norm(a.q_proj(z).view(1,count,self.heads,self.dim)).transpose(1,2)
            keys=a.k_norm(a.k_proj(z).view(1,count,self.kv_heads,self.dim)).transpose(1,2)
            q,keys=apply_rotary_pos_emb(q,keys,*rope)
            k[start:end].copy_(keys[0].transpose(0,1))
            v[start:end].copy_(a.v_proj(z).view(count,self.kv_heads,self.dim))
            if before_attention is not None:before_attention(i)
            out=attention(q[0].transpose(0,1).contiguous(),k[:end],v[:end],a.scaling)
            x=x+a.o_proj(out.reshape(1,count,self.heads*self.dim))
            x=x+layer.mlp(layer.post_attention_layernorm(x))
        return self.net.lm_head(self.core.norm(x[:,-1:]))


class Request:
    def __init__(self,model,prefix,suffix,seed,output_tokens=32):
        self.model=model;self.prefix=prefix;self.suffix=suffix;self.seed=seed;self.output_tokens=output_tokens
        self.ids=model.prompt(prefix,suffix,seed);self.ids_gpu=torch.tensor([self.ids],device='cuda')
        self.prompt_sha256=hashlib.sha256(bytes(str(self.ids),'utf8')).hexdigest()
        # Native Transformers executes every layer and provides complete-reference logits.
        cached=model.net(input_ids=self.ids_gpu[:,:prefix],use_cache=True,logits_to_keep=1).past_key_values
        self.host=[]
        for layer in cached.layers:
            pair=[]
            for src in (layer.keys,layer.values):
                x=src[0].transpose(0,1).contiguous()
                cpu=torch.empty_like(x,device='cpu',pin_memory=True);cpu.copy_(x);pair.append(cpu)
            self.host.append(pair)
        self.reference=[];self.reference_tokens=[]
        result=model.net(input_ids=self.ids_gpu[:,prefix:],past_key_values=cached,use_cache=True,logits_to_keep=1)
        for step in range(output_tokens):
            self.reference.append(result.logits.detach().cpu())
            token=result.logits.argmax(-1);self.reference_tokens.append(int(token.item()))
            if step+1<output_tokens:
                result=model.net(input_ids=token,past_key_values=result.past_key_values,use_cache=True,logits_to_keep=1)
        del cached,result
        length=prefix+suffix+output_tokens
        self.arena=[tuple(torch.empty((length,model.kv_heads,model.dim),device='cuda',dtype=torch.bfloat16)
                          for _ in range(2)) for _ in range(model.layers)]
        self.states=[torch.empty((1,suffix,model.hidden),device='cuda',dtype=torch.bfloat16)
                     for _ in range(model.layers+1)]
        self.q=[torch.empty((suffix,model.heads,model.dim),device='cuda',dtype=torch.bfloat16) for _ in range(model.layers)]
        self.a=[torch.empty_like(x) for x in self.q]
        self.positions=torch.arange(prefix,prefix+suffix,device='cuda').unsqueeze(0)
        self.rope=model.core.rotary_emb(self.states[0],self.positions)
        self.logits=torch.empty((1,1,model.net.config.vocab_size),device='cuda',dtype=torch.bfloat16)
        self.next_token=torch.zeros((1,1),device='cuda',dtype=torch.long)
        self.mailbox=torch.empty((1,1),device='cpu',dtype=torch.long,pin_memory=True)
        self.timers=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
        for event in self.timers:event.record()
        self.decode_graphs=[];self.capture_streams=[]
        self.bytes=sum(x.numel()*x.element_size() for pair in self.host for x in pair)

    def copy(self,i):
        for dst,src in zip(self.arena[i],self.host[i]):dst[:self.prefix].copy_(src,non_blocking=True)

    def embed(self):self.states[0].copy_(self.model.core.embed_tokens(self.ids_gpu[:,self.prefix:]))

    def project(self,i):
        m=self.model;layer=m.core.layers[i];a=layer.self_attn
        z=layer.input_layernorm(self.states[i])
        q=a.q_norm(a.q_proj(z).view(1,self.suffix,m.heads,m.dim)).transpose(1,2)
        k=a.k_norm(a.k_proj(z).view(1,self.suffix,m.kv_heads,m.dim)).transpose(1,2)
        q,k=apply_rotary_pos_emb(q,k,*self.rope)
        self.q[i].copy_(q[0].transpose(0,1))
        self.arena[i][0][self.prefix:self.prefix+self.suffix].copy_(k[0].transpose(0,1))
        self.arena[i][1][self.prefix:self.prefix+self.suffix].copy_(a.v_proj(z).view(self.suffix,m.kv_heads,m.dim))

    def attend(self,i):
        k,v=self.arena[i];end=self.prefix+self.suffix
        self.a[i].copy_(attention(self.q[i],k[:end],v[:end],self.model.core.layers[i].self_attn.scaling))

    def post(self,i):
        layer=self.model.core.layers[i]
        x=self.states[i]+layer.self_attn.o_proj(self.a[i].reshape(1,self.suffix,-1))
        self.states[i+1].copy_(x+layer.mlp(layer.post_attention_layernorm(x)))

    def head(self):
        self.logits.copy_(self.model.net.lm_head(self.model.core.norm(self.states[-1][:,-1:])))
        self.next_token.copy_(self.logits.argmax(-1))

    def forward(self):
        self.embed()
        for i in range(self.model.layers):self.project(i);self.attend(i);self.post(i)
        self.head()

    def reset(self,resident=False):
        for pair in self.arena:
            for x in pair:x.fill_(float('nan'))
        for x in self.states:x.fill_(float('nan'))
        if resident:
            for i in range(self.model.layers):self.copy(i)
        torch.cuda.synchronize()

    def check_logits(self,step):
        expected=self.reference[step].cuda().float();actual=self.logits.float()
        if not torch.isfinite(actual).all():raise AssertionError('nonfinite full-model logits')
        rel=float((actual-expected).norm()/expected.norm());absolute=float((actual-expected).abs().max())
        token=int(self.next_token.item())
        if rel>.005 or token!=self.reference_tokens[step]:
            raise AssertionError(f'full-model output mismatch step={step}, rel={rel}, max={absolute}, token={token}')
        return dict(relative_l2=rel,max_absolute=absolute,token=token)

    def capture_decode(self):
        pool=torch.cuda.graph_pool_handle()
        for step in range(self.output_tokens-1):
            def fn(step=step):
                self.logits.copy_(self.model.dense_forward(self.next_token,self.arena,self.prefix+self.suffix+step))
                self.next_token.copy_(self.logits.argmax(-1))
            g,s=capture(fn,pool=pool);self.decode_graphs.append(g);self.capture_streams.append(s)

    def token(self):
        self.mailbox.copy_(self.next_token,non_blocking=True)
        torch.cuda.current_stream().synchronize()
        return int(self.mailbox.item())
