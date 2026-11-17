"""Check immutable model weights without running attention or constructing graphs."""
import time
import torch
from model import Model

@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    m=Model('/root/autodl-tmp/models/Qwen3-8B')
    w=m.core.layers[22].mlp.up_proj.weight
    baseline=w.cpu().clone();copy=w.clone();torch.cuda.synchronize()
    print('weight',w.shape,w.stride(),w.data_ptr(),'clone',copy.data_ptr(),flush=True)
    for i in range(10):
        time.sleep(.2)
        now=w.cpu();idx=(now!=baseline).nonzero()
        print('idle weight differences',i,len(idx),[(p.tolist(),float(baseline[tuple(p)]),float(now[tuple(p)])) for p in idx[:8]],flush=True)
    x=torch.randn((1,64,w.shape[1]),device='cuda',dtype=w.dtype)
    for i in range(10):
        y=torch.nn.functional.linear(x,w);torch.cuda.synchronize()
        now=w.cpu();idx=(now!=baseline).nonzero()
        print('gemm weight differences',i,len(idx),[(p.tolist(),float(baseline[tuple(p)]),float(now[tuple(p)])) for p in idx[:8]],flush=True)
    print('clone changed',int((copy.cpu()!=baseline).sum()),flush=True)

if __name__=='__main__':main()
