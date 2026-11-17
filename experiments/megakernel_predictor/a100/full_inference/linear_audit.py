"""Reproduce the native GEMM repeatability issue independently of graph execution."""
import torch
torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
data=torch.load('/root/autodl-tmp/linear-audit.pt',weights_only=True)
with torch.inference_mode():
    for dtype in (torch.bfloat16,torch.float16,torch.float32):
        x=data['x'].cuda().to(dtype);w=data['w'].cuda().to(dtype)
        for library in ('cublas','cublaslt'):
            torch.backends.cuda.preferred_blas_library(library)
            y0=torch.nn.functional.linear(x,w)
            changes=[]
            for i in range(10):
                y=torch.nn.functional.linear(x,w)
                changes.append((float((y.float()-y0.float()).abs().max()),int((y!=y0).sum())))
            print(dtype,library,'finite',bool(torch.isfinite(y0).all()),changes,flush=True)
