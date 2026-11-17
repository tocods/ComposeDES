"""Produce tables and an exportable budget curve from verified experiment data."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES={'random':'随机搜索','dqn':'GCN+DQN','des':'ComposeDES'}

def main():
    p=argparse.ArgumentParser();p.add_argument('summary',type=Path);a=p.parse_args()
    data=json.loads(a.summary.read_text());out=a.summary.parent
    d=data['discovery'];lines=[
        '# 完整 Qwen3-8B 的 CUDA Graph 辅助边优化实验结果',
        '',
        '本报告依据冻结方案后的独立测量生成。GPU 数值正确性按要求不作为实验前置条件；以下结果只描述该设备、软件版本和执行器上的性能，不证明模型输出正确，也不代表原生推理服务端性能。',
        '',
        '## 1. 实际执行图与随机探索',
        '',
        f"完整模型包含 36 层、8,190,735,360 个参数。探索输入为 512-token GPU 驻留前缀和 1-token 前向。实际图含 {d['nodes']} 个节点，其中 {d['kernels']} 个 CUDA kernel，原始依赖 {d['edges']} 条。Q/K/V 与 gate/up 的独立分支保留，候选仅添加辅助依赖。",
        '',
        f"共测量 {d['samples']:,} 个不同加边图，并额外测量 {d['control_queries']} 个穿插的原图对照。搜索中单次均值的最小值为 {d['sampled_min_ms']:.6f} ms，最大值为 {d['sampled_max_ms']:.6f} ms。它们是采样极值，不是理论上下界，也不是可复现收益。",
        '',
        '| 冻结方案 | 独立复测中位时间（ms） | 配对收益（%） | 95% bootstrap 区间（%） |',
        '|---|---:|---:|---|']
    for label in ['original','AA','best-0','serial']:
        x=d['confirmation'][label];ci=x['ci95_pct']
        lines.append(f"| {label} | {x['ms']:.6f} | {x['gain_pct']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] |")
    lines += ['', '`best-0` 是搜索阶段选出的第一名；确认阶段不重新挑选主结果。串行对照用于衡量显式分支并行的作用，不将这部分收益归于 GCN+DQN 或辅助边。', '',
              '## 2. 相同完整图实测预算', '',
              '训练形状为 768/1，各方法 3,200 次查询；每个留出形状各 512 次查询。一次查询含 1 次预热、3 次计时回放。每方法训练共 12,800 次图执行，每个目标形状共 2,048 次。原图对照查询计入预算。独立确认测量不回流到选择器。', '',
              '| 方法 | 训练查询 | 搜索与选择耗时（s） | 公共准备（s） | 其中独立节点标定（s） |',
              '|---|---:|---:|---:|---:|']
    for method,x in data['training'].items():
        lines.append(f"| {NAMES[method]} | {x['budget']} | {x['search_seconds']:.2f} | {x['setup_seconds']:.2f} | {x['profiling_seconds']:.3f} |")
    lines += ['', '公共准备包括模型装载、参考路径执行、图捕获、独立节点标定和预热。各方法均执行公共准备；只有 ComposeDES 使用节点计时来预测。该成本另列，不隐藏为免费信息。搜索耗时包含模型计算、图构建、CPU 决策、训练和事件回放核对，不等于 GPU 算法推理开销。', '',
              '## 3. 新形状和新进程上的冻结方案复测', '',
              '训练后先固定策略与参数，再处理两个留出形状。所有最终候选在重建模型的新进程中、使用新 prompt 随机交错测量 32 轮，并核对源图节点、参数特征和依赖指纹相同。正收益表示快于同进程的原始分支图。', '',
              '| 前缀/新增 token | 方法与目标查询数 | 中位时间（ms） | 配对收益（%） | 95% bootstrap 区间（%） |',
              '|---|---|---:|---:|---|']
    for case in data['heldout'].values():
        for label in ['original','AA','serial','random-512','dqn-512','des-512','dqn-zero','des-zero']:
            x=case['timings'][label];ci=x['ci95_pct']
            lines.append(f"| {case['prefix']}/{case['query_tokens']} | {label} | {x['ms']:.6f} | {x['gain_pct']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] |")
    lines += ['', '`zero` 表示无目标完整图计时反馈的冻结策略。ComposeDES 仍使用目标独立 kernel 标定，不能称为完全零标定。GCN+DQN 的目标权重不更新，预算内以 epsilon=0.05 探索；ComposeDES 按相同预算更新共享参数权重。', '',
              '![Budget versus independently confirmed gain](budget-gain.png)', '',
              '预算曲线按 8/32/128/512 次目标查询的冻结选择绘制；误差条是同一确认进程内配对中位收益的 bootstrap 区间，不包含多随机训练种子或多 GPU 的不确定性。', '',
              '## 4. 完整 32-token 生成', '',
              '以下测量执行首个完整前向和后续 31 个自回归前向。仅首个前向使用选中图，后续 decode 路径一致。TTFT 从提交首个前向到 CPU 取得首 token，包含同步和取回 token；不含 tokenizer、前缀构建、图捕获或服务队列。TPOT 为同一请求中后续 token 间隔的中位数，再对 8 次生成取中位数。', '',
              '| 前缀/新增 token | 方案 | TTFT（ms） | TPOT（ms） | 总时间（ms） | 输出吞吐（token/s） |',
              '|---|---|---:|---:|---:|---:|']
    for case in data['heldout'].values():
        for label,x in case['generation'].items():
            lines.append(f"| {case['prefix']}/{case['query_tokens']} | {label} | {x['ttft_ms']:.3f} | {x['tpot_ms']:.3f} | {x['total_ms']:.3f} | {x['output_tokens_per_second']:.2f} |")
    lines += ['', '## 5. 结论适用范围', '',
              '本轮复现的是论文公开的节点层级动作、两跳 GCN、DQN 训练机制及已给出的超参数；应用从 SSTA 换为完整 Qwen3-8B。论文未完整给出的实现细节和本轮的预算扩展列于 README 与 PROTOCOL.json，不能表述为作者原代码的逐项复现。', '',
              '该实验只有一个训练随机种子、两个留出输入形状和一台存在已知显存异常的 A100；固定 kernel 图的性能测量不能证明其他负载、并发批次、GPU 或原生服务框架也有相同结果。两个留出形状仍来自相同 Transformer 结构，不构成跨应用依赖拓扑泛化证据。没有进行数值正确性筛选。若性能差异与 A/A 波动相近或区间包含零，应报告未观察到稳定收益。', '',
              'ComposeDES 在本轮使用独立节点计时和有限共享参数的资源模型。编译后的事件循环负责大批候选预测，ComposeDES WorkflowController 核对所选图的依赖回放。资源模型只是近似，不应将它的结果推广为所有 ComposeDES 模型的能力上限。', '']
    (out.parent/'RESULTS.md').write_text('\n'.join(lines).replace('(budget-gain.png)','(reference/budget-gain.png)'))
    fig,axes=plt.subplots(1,len(data['heldout']),figsize=(10,4),squeeze=False)
    for ax,(name,case) in zip(axes.flat,data['heldout'].items()):
        for method,label in [('random','Random'),('dqn','GCN+DQN'),('des','ComposeDES')]:
            xs=[8,32,128,512];vals=[case['timings'][f'{method}-{n}'] for n in xs];ys=[v['gain_pct'] for v in vals]
            err=[[max(0,v['gain_pct']-v['ci95_pct'][0]) for v in vals],[max(0,v['ci95_pct'][1]-v['gain_pct']) for v in vals]]
            ax.errorbar(xs,ys,yerr=err,marker='o',capsize=2,label=label)
        ax.axhline(0,color='black',linewidth=.7);ax.set_xscale('log',base=2);ax.set_xticks([8,32,128,512],['8','32','128','512'])
        ax.set_title(f"{name}: prefix {case['prefix']}, query {case['query_tokens']}")
        ax.set_xlabel('Target full-graph timing queries');ax.set_ylabel('Paired gain over source graph (%)');ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(out/'budget-gain.png',dpi=180);fig.savefig(out/'budget-gain.pdf');plt.close(fig)

if __name__=='__main__':main()
