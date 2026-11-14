# 在新机器上搭建 H20 实验

本目录保存 H20 实际使用的 llama.cpp 修改、CUDA 记录器的依赖源文件、bench 和
capture 源码、MonoMoE 特化脚本及固定 decode token。原版对照使用同一个基准
快照的未改动构建。快照仓库为 unixsysdev/llama-turboquant，固定提交为
`51e5c9afbbccb353cefd698036ab9ed00a3612ab`。原实验未启用其 DualDeadline。
[基准提交](https://github.com/unixsysdev/llama-turboquant/commit/51e5c9afbbccb353cefd698036ab9ed00a3612ab)

建置需要 Linux、CUDA、支持 CUDA 的 C++ 编译器、CMake、Ninja、Git 和 Python。
GPU 实验使用 H20 78 SM 和 CUDA 13.0.88。可先传 `--prepare-only` 来检验固定提交
和补丁，不触发编译。以下命令从 ComposeDES 仓库根目录执行：

```sh
p="$PWD/experiments/megakernel_predictor"
data="$p/replay-data"
run="$HOME/compose-h20-runtime"
bash "$p/h20/runtime/setup.sh" "$run"
cd "$run"
python3 train_prompts.py train-prompts.txt
```

将 [Unsloth GGUF](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/tree/bc014a17be43adabd7066b7a86075ff935c6a4e2)
的 `Qwen3.5-35B-A3B-UD-Q2_K_XL.gguf` 放到运行目录，用 `sha256sum -c model.sha256`
校验。32/64 行 tile、512 线程、三种分区和输入 token 已由脚本固定。
训练提示从 WikiText 公共快照重建，脚本校验文本 SHA-256，也可接收本地
datasets-server first-rows JSON。

## GGUF 原版和研究候选

下面以第二批留出请求为例。先生成参考 logits，再执行已冻结的候选。只跑 GGUF
完整 forward 对照不需要重做 378 MB 的 tile 记录。

```sh
cp "$data/b2-unseen-forecast."{json,sha256} compose-h20/
export LD_LIBRARY_PATH="$PWD/build/bin:${CUDA_HOME:-/usr/local/cuda}/lib64"
export QWEN_RESIDENT_EMBEDDING=1 QWEN_RESIDENT_FIXED_ROUTE=1
model=Qwen3.5-35B-A3B-UD-Q2_K_XL.gguf
QWEN_RESIDENT_CUT=4 QWEN_RECORD_REFERENCE=1 ./bench "$model" \
  compose-h20/unseen-prompts.txt 2 16 24 compose-h20/unseen-b2.tokens
bash compose-h20/run_b2.sh unseen
mkdir -p compose-h20/unmodified
QWEN_SKIP_REFERENCE=1 LD_LIBRARY_PATH="$PWD/pristine/build/bin:${CUDA_HOME:-/usr/local/cuda}/lib64" \
  ./bench-stock "$model" compose-h20/unseen-prompts.txt 2 16 24 compose-h20/unseen-b2.tokens \
  > compose-h20/unmodified/0-upstream.csv
```

若需重新测 tile 服务时间，由 `capture` 采集真实路由，再运行
`compose-h20/tile_trace_v2 MODEL CAPTURE 8`。训练用的捕获还可交给
`fp8_components.py`。固化的统计数据在 `../../replay-data/`，原始 GPU
时间戳和完整 logits 由运行生成。

## FP8 MonoMoE

将 [Qwen 官方 FP8 checkpoint](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8/tree/main)
下载到 `compose-h20/model/`。实验使用 Python 3.12、PyTorch 2.12.1 CUDA 13.0、
Transformers 5.9.0、tokenizers 0.22.2、safetensors、accelerate、FlashInfer 0.7.4
及其支持 CUDA 13 的 JIT 依赖，安装在独立环境中。使用与 PyTorch 匹配的 Triton。
运行时使用 fla-core 和 flash-linear-attention 0.6.0。代码的 64 CTA 特化通过
FlashInfer 的 JIT 生成。可使用已激活环境，或者设置 `PYTHONPATH` 指向已有依赖。

```sh
cp "$data/fp8-forecast-eager."{json,sha256} compose-h20/
bash compose-h20/run_fp8.sh heldout eager
```

全模型 CUDA Graph 代码和探索性预测保留用于继续排查；该后端在留出请求的数值检查
中失败。正式结果由 eager 的三轮对照生成。
