"""Render the full-model performance measurements as a reproducible Markdown report."""
import argparse,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('summary',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    data=json.loads(a.summary.read_text());tests=data['tests'];fixed=data['fixed_from_calibration']
    lines=['# 完整 Qwen3-8B 的 CUDA Graph 调度性能实验','',
      '本报告按用户要求关闭数值一致性检查的阻断，只评价实际执行时间。GPU 已发现间歇性数据错误；这些数据是该实例上的性能测量，不是通过正确性验证的推理加速结果。模型权重、36 层计算和数据搬运均保留，未用合成 kernel 替代。','',
      '## 1. 工作负载与测量范围','',
      '- A100 PCIe 40GB；完整 Qwen3-8B，8,190,735,360 个参数，BF16，全权重驻留 GPU。',
      '- CPU 命中的前缀 KV 逐层读回 GPU；新增输入经过全部 36 层、最终归一化和 lm_head，之后自回归生成，总计固定生成 32 个 token。',
      '- TTFT 包括读回、完整模型计算及首 token 返回 CPU；不包括模型加载、分词、建立前缀缓存、构图、网络或排队。',
      '- TPOT 为后续 31 次完整 decode 的逐 token 延迟。所有候选共用相同 decode 路径；本轮只修改首 token 执行图，不能将 TPOT 的小幅波动算作调度收益。',
      '- 单请求独立执行器，使用 PyTorch/cuBLAS 与 FlashInfer FA2；不是原生 SGLang/vLLM 服务吞吐测试。',
      '- 2 个标定形状与 3 个留出形状；每个留出形状使用 A/B 两个独立进程和不同 prompt。每进程 20 个方案，各 24 次 TTFT 与 4 次完整生成；两组标定和六组验证合计 4,480 条正式测量记录，其中 640 次完整生成，共 20,480 个输出 token。预热和流程检查不计入此数。','',
      '## 2. 对照与选择协议','',
      '显式原图保留逐层读回、QKV、attention、输出投影及 MLP 的依赖。候选改变节点创建顺序，或添加计算到后续读回的辅助依赖。17 个显式图包括一个 A/A 重复图；另外设置普通完整前向 CUDA Graph、同样分阶段的普通 CUDA Graph，以及 KV 全驻留 GPU 对照。所有方案使用同一套权重、输入长度、缓存地址与读回字节数。','',
      '普通完整前向是主要性能基线。相对显式原图加速，并不等于优于这个基线。KV 全驻留方案不包含读回，不能作为相同输入位置下的可直接替换方案。','',
      f'最佳固定方案只用两个标定形状确定，为 `{fixed}`。ComposeDES 共享参数也在留出测试前冻结；每个新形状仍需要独立阶段计时。预测只读取功能依赖、创建顺序和独立阶段耗时，不读取该目标图的完整计时。仿真枚举耗时单独报告，属于离线选择，不包含在请求 TTFT 内。','',
      '实测搜索用进程 A 选择，再在进程 B 评价，并允许保留普通完整前向。事后最优只表示已测候选中最小的中位值，包含选择偏差，不是理论上限。本轮未在完整模型上训练 GNN+DQN，不能称为原论文强化学习算法的完整复现。','',
      '## 3. 进程 B 的首 token 延迟','',
      '| CPU 前缀 / 新增输入 | 普通完整前向 | 显式原图 | 标定最佳固定 | ComposeDES 选中 | 其 TTFT | A 搜索 → B 评价 |',
      '|---|---:|---:|---:|---|---:|---:|']
    for name,t in tests.items():
      if t['session']!='B':continue
      s=t['stats'];chosen=t['prediction']['label']
      lines.append(f"| {t['prefix']} / {t['suffix']} | {s['ordinary-full']['ttft_ms']:.3f} | {s['explicit-original']['ttft_ms']:.3f} | {s[fixed]['ttft_ms']:.3f} | {chosen} | {s[chosen]['ttft_ms']:.3f} | {t['search_A_evaluate_B']['ttft_ms']:.3f} |")
    lines+=['','单位为 ms，各为 24 次测量的中位数。','',
      '## 4. 完整生成的性能','',
      '| CPU 前缀 / 新增输入 | 方案 | TPOT（ms） | 生成 32 token 总耗时（ms） | 输出 token/s |',
      '|---|---|---:|---:|---:|']
    for name,t in tests.items():
      if t['session']!='B':continue
      for label in ['ordinary-full',t['prediction']['label']]:
        s=t['stats'][label]
        lines.append(f"| {t['prefix']} / {t['suffix']} | {label} | {s['tpot_ms']:.3f} | {s['total_ms']:.3f} | {s['output_tokens_per_second']:.2f} |")
    lines+=['','每方案 4 次完整生成；这些重复数不支持 P99 结论。输出 token/s 包括首 token 等待，并非并发服务吞吐。','',
      '## 5. ComposeDES 的排序与选择成本','',
      '| 留出进程 | 选中方案 | 相对普通前向配对收益（%） | 95% 区间 | 相对已测最佳显式候选的差距（%） | 排序相关系数 | 仿真枚举耗时（s） |',
      '|---|---|---:|---|---:|---:|---:|']
    for name,t in tests.items():
      p=t['prediction'];s=t['stats'][p['label']];ci=s['ci95_pct']
      rho=f"{p['spearman']:.3f}" if p['spearman'] is not None else '未定义'
      lines.append(f"| {name} | {p['label']} | {s['paired_gain_pct']:+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] | {t['selected_regret_pct']:.3f} | {rho} | {p['selection_seconds']:.3f} |")
    lines+=['','正收益表示更快。区间通过同轮配对收益的 bootstrap 得到，只反映单进程内测量波动，不能覆盖不同机器、显存错误或构图实例的全部不确定性。阶段计时和共享参数标定成本不包含在最后一列中；当前方法并非零标定或在线低开销。','',
      '## 6. A/A 对照与内存','',
      '| 留出进程 | 相同图 A/A 配对差值（%） | 95% 区间 | GPU 设备用量（GiB） | 前缀读回量（MiB） |',
      '|---|---:|---|---:|---:|']
    for name,t in tests.items():
      aa=t['aa_vs_original'];ci=aa['ci95_pct']
      lines.append(f"| {name} | {aa['paired_gain_pct']:+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] | {t['memory_bytes']['driver_used_bytes']/2**30:.2f} | {t['h2d_bytes']/2**20:.0f} |")
    lines+=['','内存数值是在所有候选图均已构建、共享权重和缓存时记录的设备用量，不是某个生产框架单独运行的峰值。A/A 差异不能算算法收益。','',
      '## 7. 全部留出候选的 TTFT','',
      '| 方案 | '+' | '.join(tests)+' |',
      '|---|'+'---:|'*len(tests)]
    labels=next(iter(tests.values()))['stats']
    for label in labels:
      lines.append('| '+label+' | '+' | '.join(f"{t['stats'][label]['ttft_ms']:.3f}" for t in tests.values())+' |')
    lines+=['','全部值单位为 ms。统计汇总、冻结参数和输入文件摘要保存于 `reference/performance-summary.json`、`reference/frozen-performance.json` 与 `reference/performance-protocol.json`。原始逐次测量位于本地 `experiments/megakernel_predictor/runs/full-qwen-performance/`，也已随分支保存为 [原始记录归档](reference/performance-raw.tar.gz)，校验值见 [归档清单](reference/performance-artifacts.json)。归档不包含模型权重。','',
      '## 8. 独立阶段数据','',
      '| 留出进程 | 独立计算阶段耗时之和（ms） | 独立读回阶段耗时之和（ms） | 普通完整前向 TTFT（ms） |',
      '|---|---:|---:|---:|']
    for name,t in tests.items():
      q=t['independent_stage_sums_ms']
      lines.append(f"| {name} | {q['compute']:.3f} | {q['io']:.3f} | {t['stats']['ordinary-full']['ttft_ms']:.3f} |")
    lines+=['','独立阶段耗时与完整图的执行状态不同，不能直接作为严格下界。它们用于判断计算与读回的相对规模；不能据此宣称已经达到 PCIe 硬件上限，也不能直接从总时间差确定具体争用来源。','',
      '## 9. 复跑','',
      '```bash','PERFORMANCE_ONLY=1 OUT=../../runs/full-qwen-performance bash run.sh',
      'python analyze.py ../../runs/full-qwen-performance --out reference/performance-summary.json',
      'python write_report.py reference/performance-summary.json PERFORMANCE.md','```','',
      '不设置 `PERFORMANCE_ONLY=1` 时，仍要求显存检查与数值校验通过。性能模式保留结构检查、实际计算与传输，遇到 CUDA 执行错误仍会终止。']
    chosen_labels=sorted({t['prediction']['label'] for t in tests.values()})
    original_gains=[100*(1-t['stats'][t['prediction']['label']]['ttft_ms']/t['stats']['explicit-original']['ttft_ms']) for t in tests.values()]
    native_gains=[100*(1-t['stats'][t['prediction']['label']]['ttft_ms']/t['stats']['ordinary-full']['ttft_ms']) for t in tests.values()]
    overview=['## 主要观测','',
      f"- 六组留出测量中，ComposeDES 的选择为：{', '.join('`'+x+'`' for x in chosen_labels)}。固定方案由标定数据选定为 `{fixed}`。",
      f'- 选择结果相对显式原图的 TTFT 变化为加速 {min(original_gains):.3f}%–{max(original_gains):.3f}%；相对普通完整前向的变化范围为 {min(native_gains):+.3f}%–{max(native_gains):+.3f}%（正值为加速）。']
    if len(chosen_labels)==1:
      overview.append('- 预测器没有表现出跨形状改变选择的行为；不能仅凭恢复显式原图的开销证明自适应选择优于固定构图规则。')
    if max(abs(x) for x in native_gains)<.1:
      overview.append('- 相对普通完整前向，所选方案的中位时间差均小于 0.1%；本轮没有显示有实际意义的推理加速。')
    overview+=['- 这里可以观察到创建顺序影响完整模型执行时间，但不能据此宣称已经复现原论文通过强化学习加边获得完整模型加速的结论。','']
    lines[4:4]=overview
    a.output.write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
