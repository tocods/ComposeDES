"""Numerical audit; not used for performance measurements."""
import torch
import model
from transformers.cache_utils import DynamicCache

@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    m=model.Model('/root/autodl-tmp/models/Qwen3-8B')
    ids=torch.tensor([m.prompt(512,64,11)],device='cuda')
    prefix=m.net(input_ids=ids[:,:512],use_cache=True,logits_to_keep=1).past_key_values
    saved=[(x.keys.clone(),x.values.clone()) for x in prefix.layers]
    samples=[]
    def wrapper(module,q,k,v,mask,**kwargs):
        out=model.framework_attention(module,q,k,v,mask,**kwargs)
        if module.layer_idx==22:
            samples.append([x.detach().clone() for x in (q,k,v,out[0])])
        return out
    model.ALL_ATTENTION_FUNCTIONS.register('full_model_fa2',wrapper)
    logits=[];records=[];current={}
    def hook(name):
        def fn(module,inputs,output):
            if isinstance(output,torch.Tensor):current[name]=(inputs[0].clone(),output.clone())
        return fn
    handles=[module.register_forward_hook(hook(name)) for name,module in m.core.named_modules()
             if any(name.startswith(f'layers.{i}.') for i in range(22,25)) and not list(module.children())]
    for j in range(4):
        current={}
        cache=DynamicCache(config=m.net.config)
        for i,(k,v) in enumerate(saved):cache.update(k.clone(),v.clone(),i)
        logits.append(m.net(input_ids=ids[:,512:],past_key_values=cache,use_cache=True,logits_to_keep=1).logits.clone())
        print('native repeat',j, float((logits[-1].float()-logits[0].float()).norm()/logits[0].float().norm()),flush=True)
        records.append(current)
    for name in records[0]:
        for j in range(1,4):
            inp,out=records[j][name];inp0,out0=records[0][name]
            if not torch.equal(out,out0):
                print('module difference',j,name,'input',float((inp.float()-inp0.float()).abs().max()),'output',float((out.float()-out0.float()).abs().max()),flush=True)
    for h in handles:h.remove()
    linear=m.core.layers[22].mlp.up_proj
    x=records[0]['layers.22.mlp.up_proj'][0]
    torch.save(dict(x=x.cpu(),w=linear.weight.detach().cpu()),'/root/autodl-tmp/linear-audit.pt')
    first=linear(x).clone()
    copied_weight=linear.weight.clone();copied_x=x.clone()
    reference=torch.nn.functional.linear(copied_x,copied_weight)
    for j in range(16):
        y=linear(x).clone()
        ycopy=torch.nn.functional.linear(copied_x,copied_weight)
        print('fixed linear',j,float((y.float()-first.float()).abs().max()),int((y!=first).sum()),
              'copydiff',int((ycopy!=reference).sum()),'weights changed',int((linear.weight!=copied_weight).sum()),
              'input changed',int((x!=copied_x).sum()),flush=True)
    q,k,v,out=samples[0]
    for j,s in enumerate(samples):
        print('layer22 inputs/output',j,[(float((a.float()-b.float()).abs().max()),int((a!=b).sum())) for a,b in zip(s,samples[0])],flush=True)
    q=q[0].transpose(0,1).contiguous();k=k[0].transpose(0,1).contiguous();v=v[0].transpose(0,1).contiguous()
    outs=[]
    for j in range(8):
        outs.append(model.attention(q,k,v,128**-.5).clone())
        print('fixed attention',j,float((outs[-1].float()-outs[0].float()).abs().max()),int((outs[-1]!=outs[0]).sum()),flush=True)
    torch.save(dict(q=q.cpu(),k=k.cpu(),v=v.cpu(),outputs=[x.cpu() for x in outs]),'/root/autodl-tmp/attention-audit.pt')

if __name__=='__main__':main()
