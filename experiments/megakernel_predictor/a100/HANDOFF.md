# A100 实验交接

最新工作是 ComposeDES 的准入模型、在线参数更新和主动选点。结果见 [实验报告](research_probes/ACTIVE_ADMISSION.md) 与 [紧凑汇总](reference/active-admission.json)。联合构图与搬运配置的实验见 [联合选择报告](research_probes/JOINT_ADMISSION.md)。早期的 MPK、vLLM 和 HiCache 复现代码也在本目录。

## 新服务器准备

```bash
git clone --branch codex/a100-admission-handoff --recurse-submodules https://github.com/tocods/ComposeDES.git
cd ComposeDES/experiments/megakernel_predictor/a100
```

新分支包含旧的 H20 复现提交。运行 ComposeDES 预测器至少需要 `fncs` 子模块。若之前克隆时跳过了子模块，从仓库根目录运行 `git submodule update --init fncs`。

最近一次 A100 测试的版本见 [环境记录](reference/a100-environment.txt)。GPU 实验用 Linux、CUDA 13 Toolkit 的 `nvcc`、可用的 NVIDIA 驱动以及带 CUDA 的 PyTorch。主机侧评估脚本需要 NumPy，绘图另需 Matplotlib。依照环境记录准备 GPU 虚拟环境并激活。GPU 或驱动变化后，先复跑正确性检查，再生成新机器的独立阶段计时与标定。

在新服务器准备 Hugging Face 模型 `Qwen/Qwen3-8B`，目录应含 `config.json`、`model.safetensors.index.json` 及五个权重分片。如果已安装 Hugging Face CLI：

```bash
export QWEN_MODEL_DIR=/path/to/Qwen3-8B
hf download Qwen/Qwen3-8B --local-dir "$QWEN_MODEL_DIR"
a100_dir="$PWD"
(cd "$QWEN_MODEL_DIR" && sha256sum -c "$a100_dir/qwen3-8b.sha256")
```

## 最近一轮的复跑入口

在 `a100` 目录运行下列命令。第一轮检查只运行一组 Qwen 子图，完整留出随后运行八组。设定资源预算前，先看 [报告中的候选与标定开销](research_probes/ACTIVE_ADMISSION.md)。

```bash
cd research_probes
export PATH="/usr/local/cuda/bin:$PATH"
export OMP_NUM_THREADS=4 MAX_JOBS=6
nvcc -std=c++17 -shared -Xcompiler -fPIC explicit_graph.cu -lcuda -o libexplicit_graph.so

run_dir=../../runs/mpk-a100/active-admission
mkdir -p "$run_dir"
python admission_framework.py --model "$QWEN_MODEL_DIR" --session A \
  --smoke --repeat 2 --output "$run_dir/smoke.jsonl"

python admission_calibrate.py --repeat 12 --output "$run_dir/active-cal.jsonl"
python active_evaluate.py "$run_dir" freeze
for session in A B; do
  python admission_framework.py --model "$QWEN_MODEL_DIR" --session "$session" \
    --repeat 16 --output "$run_dir/active-qwen-$session.jsonl"
done

python active_evaluate.py "$run_dir" test
python audit_active.py "$run_dir"
python ablate_active.py "$run_dir"
python plot_active.py "$run_dir"
```

训练与留出输入由脚本按固定规则产生。重新标定时按新机器生成 `frozen.json`，然后读取该机器的留出数据。分支内的历史紧凑汇总用于核对结论。旧的原始测量与冻结文件单独归档。若需重算历史数字，将归档解压到 `experiments/megakernel_predictor/runs/mpk-a100/`，然后运行对应分析命令。

## MPK 与 HiCache 旧路径

- [整模型 MPK 与 vLLM 对照](README.md)使用 Mirage `d2189e50063b54ed410cf7b8390b2c784114c7a2`。按该文档选择最初的 `mpk-stream-join.patch`。
- [HiCache 子实验](hicache/README.md)和[读回关键路径](hicache/readback/CAUSAL.md)保留了研究期的脚本、测试设置及 SGLang 补丁。
- [关机前的 Mirage 工作树补丁](mirage-server-final.patch)记录了同一 Mirage 基线上的最终三文件改动，适用于恢复关机时的 Mirage 源码。将其直接应用到干净的 Mirage 基线。它与前述阶段补丁有重叠。
- 最新准入模型实验直接回放 CUDA Graph，使用原生 SGLang 搬运 kernel 与框架 Qwen3 层。此路径无需给 Mirage 打补丁。

后续若继续研究，先冻结两条新基线并将它们放在未见过的图上比较。一条先选择搬运方式再试构图顺序，另一条按冻结预测分数依次实测。近期八组留出上，在线准入参数更新的收益仍未得到支持。
