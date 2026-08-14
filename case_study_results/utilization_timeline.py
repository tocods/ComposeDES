#!/usr/bin/env python3
"""
GPU 利用率时间线分析

从 co-sim 的 jobRun.xml 和 jobs.json 绘制 LLaMA 7B 不同并行策略下的 GPU 时间线。
计算任务使用 NeuSight CSV 转换后的 per-task 延迟和利用率；网络等待来自 GPUSim NetworkRecord，
并按 wall-clock 区间并集统计，避免同步 rank 的等待被重复相加。
"""

import pandas as pd
import ast
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
import json
import xml.etree.ElementTree as ET

# 尝试加载中文字体
for _fname in ['PingFang SC', 'Heiti SC', 'STHeiti', 'Arial Unicode MS', 'SimHei']:
    if any(_fname in f.name for f in fm.fontManager.ttflist):
        plt.rcParams['font.sans-serif'] = [_fname, 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        break

# H100 参数
PEAK_FLOPS = 66908e9  # GFLOPS → FLOPS (from device config SingleFLOPs)
MEM_BW = 3430e9       # GB/s → B/s

CSV_DIR = Path("/Users/taco/Desktop/cosim/NeuSight/scripts/asplos/results/prediction/NVIDIA_H100_80GB_HBM3/neusight")


def compute_flops_from_ops(ops_list):
    """从 FwOps 列表计算总 FLOPs"""
    total_flops = 0
    total_bytes = 0
    for op in ops_list:
        if isinstance(op, (list, tuple)) and len(op) == 2:
            opname, args = op
            if opname == 'Linear' and len(args) == 3:
                M, N, K = args
                total_flops += 2 * M * N * K
            elif opname == 'BMM' and len(args) == 4:
                B, M, N, K = args
                total_flops += 2 * B * M * N * K
            elif opname.startswith('VEC'):
                if len(args) == 2:
                    B, H = args
                    # 向量操作是内存受限的，用字节数衡量
                    total_bytes += B * H * 2  # fp16
                elif len(args) == 1:
                    total_bytes += args[0] * 2
            elif opname == 'MEM':
                if isinstance(args, (list, tuple)):
                    for shape in args:
                        if isinstance(shape, (list, tuple)):
                            n = 1
                            for d in shape:
                                n *= d
                            total_bytes += n * 2  # fp16
            elif opname in ('ALLREDUCE', 'ALLREDUCE_ASYNC', 'SENDRECV'):
                pass  # 通信操作
    return total_flops, total_bytes


def analyze_csv(csv_path, n_layers=24, label="unknown"):
    """分析一个 NeuSight CSV，返回利用率时间线"""
    df = pd.read_csv(csv_path, converters={
        'FwOps': ast.literal_eval,
        'BwOps': ast.literal_eval,
        'AccOps': ast.literal_eval,
        'OpName': str,
    })

    # 判断是否需要层复制（base profile 单层 → 复制到 n_layers 层）
    has_layer_prefix = any('transformer_h_1_' in str(name) for name in df['Name'])
    replicate = not has_layer_prefix

    kernels = []
    for _, row in df.iterrows():
        name = str(row['Name'])
        opname = str(row['OpName'])
        fw_latency = float(row['fw_latency'])
        bw_latency = float(row['bw_latency'])
        acc_latency = float(row['acc_latency'])
        e2e_latency = float(row['e2e_latency'])

        if e2e_latency <= 0:
            continue

        fw_ops = row['FwOps'] if isinstance(row['FwOps'], list) else []
        bw_ops = row['BwOps'] if isinstance(row['BwOps'], list) else []

        # 仅用 forward FLOPs 和 forward latency 计算利用率
        fw_flops, fw_bytes = compute_flops_from_ops(fw_ops)

        # 判断操作类型
        is_comm = opname.lower() in ('allreduce', 'sendrecv')
        is_compute = opname in ('Linear', 'BMM')
        is_vec = opname.startswith('VEC')
        is_mem = opname in ('MEM', 'dropout', 'misc')

        # 计算利用率
        if is_comm:
            util = 0.0  # 通信期间 GPU 计算单元空闲
        elif is_compute and fw_latency > 0 and fw_flops > 0:
            achieved = fw_flops / (fw_latency * 1e-3)  # FLOPS
            util = min(achieved / PEAK_FLOPS, 1.0)
        elif is_vec and fw_latency > 0 and fw_bytes > 0:
            # 向量操作：用内存带宽利用率
            achieved_bw = fw_bytes / (fw_latency * 1e-3)  # B/s
            util = min(achieved_bw / MEM_BW, 1.0)
        else:
            util = 0.0

        kernels.append({
            'name': name,
            'opname': opname,
            'fw_latency': fw_latency,
            'e2e_latency': e2e_latency,
            'util': util,
            'flops': fw_flops,
            'is_comm': is_comm,
            'is_compute': is_compute,
        })

    # 展开层（base profile 需要复制）
    if replicate:
        expanded = []
        for layer_idx in range(n_layers):
            for k in kernels:
                expanded.append({
                    **k,
                    'name': f'layer_{layer_idx}/{k["name"]}',
                })
        kernels = expanded

    return kernels


def build_timeline(kernels):
    """从 kernel 列表构建时间线 (time_points, util_points)"""
    times = [0.0]
    utils = []
    current_time = 0.0

    for k in kernels:
        dt = k['e2e_latency']
        if dt > 0:
            # 阶跃函数：在 [t, t+dt) 区间利用率为 util
            times.append(current_time)
            utils.append(k['util'])
            times.append(current_time + dt)
            utils.append(k['util'])
            current_time += dt

    return np.array(times), np.array(utils), current_time


def smooth_timeline(times, utils, window_ms=1.0):
    """对利用率时间线做滑动平均平滑"""
    if len(times) < 2:
        return times, utils

    total_time = times[-1]
    n_points = max(100, int(total_time / (window_ms / 2)))
    smooth_t = np.linspace(0, total_time, n_points)
    smooth_u = np.zeros(n_points)

    for i, t in enumerate(smooth_t):
        # 找到 t 时刻正在执行的 kernel 的利用率
        # 使用阶跃函数查找
        idx = np.searchsorted(times, t, side='right') - 1
        idx = max(0, min(idx, len(utils) - 1))
        smooth_u[i] = utils[idx]

    # 滑动平均
    window = max(1, int(window_ms / (total_time / n_points)))
    if window > 1:
        kernel_arr = np.ones(window) / window
        smooth_u = np.convolve(smooth_u, kernel_arr, mode='same')

    return smooth_t, smooth_u


def load_job_utilizations(jobs_path):
    """Load NeuSight-derived utilization from co-sim jobs.json."""
    if not jobs_path.exists():
        return {}

    with open(jobs_path) as jobs_file:
        jobs = json.load(jobs_file)

    job_to_utilization = {}
    for job in jobs:
        kernels = job.get('gpu_task', {}).get('kernels', [])
        if not kernels:
            continue
        weighted_utilization = 0.0
        total_duration = 0.0
        for kernel in kernels:
            duration = float(kernel.get('thread_length', 0.0) or 0.0)
            utilization = float(kernel.get('utilization', 0.0) or 0.0)
            weighted_utilization += duration * utilization
            total_duration += duration
        job_to_utilization[job.get('name', '')] = weighted_utilization / total_duration if total_duration > 0 else 0.0
    return job_to_utilization


def parse_job_run(job_run_path, job_utilizations=None):
    """Read co-simulation jobRun.xml and return host utilization intervals in milliseconds."""
    root = ET.parse(job_run_path).getroot()
    host_to_intervals = {}
    job_utilizations = job_utilizations or {}

    for job in root.findall('Job'):
        job_name = job.get('name', 'unknown')
        for running_record in job.findall('RunningRecord'):
            if running_record.get('status') != 'Success':
                continue

            host_name = running_record.get('host', 'unknown')
            kernel_records = running_record.findall('KernelRecord')
            if not kernel_records:
                start_ms = float(running_record.get('start', 0.0)) / 1000.0
                end_ms = float(running_record.get('end', 0.0)) / 1000.0
                if end_ms > start_ms:
                    utilization = job_utilizations.get(job_name, 0.0)
                    host_to_intervals.setdefault(host_name, []).append((start_ms, end_ms, job_name, utilization))
                continue

            for kernel_record in kernel_records:
                start_ms = float(kernel_record.get('start', running_record.get('start', 0.0))) / 1000.0
                end_ms = float(kernel_record.get('end', running_record.get('end', 0.0))) / 1000.0
                utilization = float(kernel_record.get('utilization', job_utilizations.get(job_name, 0.0)))
                if end_ms <= start_ms:
                    continue
                kernel_name = kernel_record.get('name', job_name)
                host_to_intervals.setdefault(host_name, []).append((start_ms, end_ms, kernel_name, utilization))

    for intervals in host_to_intervals.values():
        intervals.sort(key=lambda item: item[0])

    return dict(sorted(host_to_intervals.items(), key=lambda item: int(item[0].replace('host', ''))))


def parse_network_records(job_run_path):
    """Read explicit network wait intervals emitted by GPUSim NetworkRecord."""
    root = ET.parse(job_run_path).getroot()
    job_to_host = {}
    for job in root.findall('Job'):
        job_name = job.get('name', 'unknown')
        running_record = job.find('RunningRecord')
        if running_record is not None:
            job_to_host[job_name] = running_record.get('host', 'unknown')

    host_to_network_intervals = {}
    for network_record in root.findall('./NetworkRecords/NetworkRecord'):
        job_name = network_record.get('job', 'unknown')
        host_name = job_to_host.get(job_name, 'unknown')
        start_ms = float(network_record.get('start', 0.0)) / 1000.0
        end_ms = float(network_record.get('end', 0.0)) / 1000.0
        if end_ms <= start_ms:
            continue
        packet_count = int(network_record.get('packetCount', 0))
        total_bytes = int(network_record.get('totalBytes', 0))
        host_to_network_intervals.setdefault(host_name, []).append(
            (start_ms, end_ms, job_name, packet_count, total_bytes)
        )

    for intervals in host_to_network_intervals.values():
        intervals.sort(key=lambda item: item[0])
    return host_to_network_intervals


def merge_time_intervals(intervals):
    """Merge overlapping wall-clock intervals and return [(start, end), ...]."""
    valid_intervals = sorted((start, end) for start, end in intervals if end > start)
    if not valid_intervals:
        return []

    merged_intervals = []
    for start, end in valid_intervals:
        if not merged_intervals or start > merged_intervals[-1][1]:
            merged_intervals.append([start, end])
            continue
        merged_intervals[-1][1] = max(merged_intervals[-1][1], end)

    return [(start, end) for start, end in merged_intervals]


def summarize_timeline(host_to_intervals, host_to_network_intervals=None):
    """Compute wall-clock time, GPU busy ratio, and wall-clock network wait from co-sim intervals."""
    host_to_network_intervals = host_to_network_intervals or {}
    compute_intervals = [interval for intervals in host_to_intervals.values() for interval in intervals]
    network_intervals = [interval for intervals in host_to_network_intervals.values() for interval in intervals]
    if not compute_intervals and not network_intervals:
        return 0.0, 0.0, 0.0, 0.0

    interval_starts = [start for start, _, _, _ in compute_intervals]
    interval_ends = [end for _, end, _, _ in compute_intervals]
    interval_starts.extend(start for start, _, _, _, _ in network_intervals)
    interval_ends.extend(end for _, end, _, _, _ in network_intervals)

    start_time = min(interval_starts)
    end_time = max(interval_ends)
    wall_time = end_time - start_time
    total_busy_time = sum(end - start for intervals in host_to_intervals.values() for start, end, _, _ in intervals)
    merged_network_intervals = merge_time_intervals((start, end) for start, end, _, _, _ in network_intervals)
    total_network_wait = sum(end - start for start, end in merged_network_intervals)
    host_count = max(len(set(host_to_intervals.keys()) | set(host_to_network_intervals.keys())), 1)
    average_busy_ratio = total_busy_time / (wall_time * host_count) if wall_time > 0 else 0.0
    return wall_time, total_busy_time, average_busy_ratio, total_network_wait


def summarize_host_intervals(host_to_intervals):
    """Compute wall-clock time and GPU busy ratio from parsed co-sim intervals."""
    wall_time, total_busy_time, average_busy_ratio, _ = summarize_timeline(host_to_intervals)
    return wall_time, total_busy_time, average_busy_ratio


def build_average_utilization_curve(host_to_intervals, max_points=1600):
    """Build average GPU utilization curve from co-sim execution intervals."""
    all_intervals = [interval for intervals in host_to_intervals.values() for interval in intervals]
    if not all_intervals:
        return np.array([]), np.array([])

    start_time = min(start for start, _, _, _ in all_intervals)
    end_time = max(end for _, end, _, _ in all_intervals)
    if end_time <= start_time:
        return np.array([]), np.array([])

    sample_count = min(max_points, max(200, int((end_time - start_time) * 4)))
    time_points = np.linspace(start_time, end_time, sample_count)
    utilization_points = np.zeros(sample_count)
    host_count = max(len(host_to_intervals), 1)

    for intervals in host_to_intervals.values():
        host_utilization = np.zeros(sample_count)
        for interval_start, interval_end, _, utilization in intervals:
            mask = (time_points >= interval_start) & (time_points < interval_end)
            host_utilization[mask] = np.maximum(host_utilization[mask], utilization)
        utilization_points += host_utilization

    return time_points - start_time, utilization_points / host_count


def summarize_network_intervals(host_to_network_intervals):
    """Compute explicit network wait time from NetworkRecord intervals."""
    all_intervals = [interval for intervals in host_to_network_intervals.values() for interval in intervals]
    if not all_intervals:
        return 0.0
    return sum(end - start for start, end, _, _, _ in all_intervals)


def draw_execution_timeline(ax, host_to_intervals, host_to_network_intervals, title, color):
    """Draw a co-sim GPU Gantt timeline; network/scheduling waits remain as blank gaps."""
    host_names = list(host_to_intervals.keys())
    y_positions = np.arange(len(host_names))
    host_to_y_position = dict(zip(host_names, y_positions))

    for host_name in host_names:
        y_position = host_to_y_position[host_name]
        intervals = host_to_intervals.get(host_name, [])
        for start, end, task_name, utilization in intervals:
            duration = end - start
            alpha = 0.20 + 0.80 * max(0.0, min(1.0, utilization))
            ax.broken_barh(
                [(start, duration)],
                (y_position - 0.35, 0.55),
                facecolors=color,
                edgecolors='white',
                linewidth=0.35,
                alpha=alpha,
            )

            if duration < 20:
                continue
            short_name = task_name.replace('layer_', 'L').replace('embed', 'emb').replace('head', 'head')
            ax.text(start + duration / 2, y_position - 0.08, short_name, ha='center', va='center', fontsize=6, color='white')

    wall_time, total_busy_time, average_busy_ratio, total_network_wait = summarize_timeline(
        host_to_intervals,
        host_to_network_intervals,
    )
    network_ratio = total_network_wait / wall_time if wall_time > 0 else 0.0
    _, utilization_points = build_average_utilization_curve(host_to_intervals)
    average_utilization = float(np.mean(utilization_points)) if len(utilization_points) else 0.0
    ax.set_title(
        f'{title} | makespan={wall_time:.1f} ms, net wait={network_ratio * 100:.1f}%, '
        f'avg busy={average_busy_ratio * 100:.1f}%, avg util={average_utilization * 100:.1f}%',
        fontsize=12,
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(host_names)
    ax.set_ylabel('GPU')
    ax.set_xlim(0, wall_time * 1.01 if wall_time > 0 else 1)
    ax.grid(True, axis='x', alpha=0.25)
    ax.set_ylim(-0.7, len(host_names) - 0.3)


def main():
    base_dir = Path('/Users/taco/Desktop/cosim/case_study_results/ai_validation/inputs')
    scenarios = [
        ('Llama7B_2GPU_TP2_25G_chain', 'LLaMA 7B | TP=2 | chain 25G', '#1976D2'),
        ('Llama7B_2GPU_TP2_25G_p2p', 'LLaMA 7B | TP=2 | direct p2p 25G', '#64B5F6'),
        ('Llama7B_4GPU_TP4_25G_chain', 'LLaMA 7B | TP=4 | chain 25G', '#0288D1'),
        ('Llama7B_4GPU_TP4_25G_p2p', 'LLaMA 7B | TP=4 | direct p2p 25G', '#4DD0E1'),
        ('Llama7B_4GPU_PP4_25G_chain', 'LLaMA 7B | PP=4 | chain 25G', '#F57C00'),
        ('Llama7B_4GPU_PP4_25G_p2p', 'LLaMA 7B | PP=4 | direct p2p 25G', '#FFB74D'),
        ('Llama7B_4GPU_TP2PP2_25G_chain', 'LLaMA 7B | TP=2 + PP=2 | chain 25G', '#388E3C'),
        ('Llama7B_4GPU_TP2PP2_25G_p2p', 'LLaMA 7B | TP=2 + PP=2 | direct p2p 25G', '#81C784'),
    ]

    parsed_scenarios = []
    for directory_name, label, color in scenarios:
        scenario_dir = base_dir / directory_name
        job_run_path = scenario_dir / 'output/jobRun.xml'
        if not job_run_path.exists():
            print(f'SKIP: {job_run_path} not found')
            continue
        job_utilizations = load_job_utilizations(scenario_dir / 'jobs.json')
        host_to_intervals = parse_job_run(job_run_path, job_utilizations)
        host_to_network_intervals = parse_network_records(job_run_path)
        if not host_to_intervals:
            print(f'SKIP: {job_run_path} has no successful tasks')
            continue
        parsed_scenarios.append((directory_name, label, color, host_to_intervals, host_to_network_intervals))

    if not parsed_scenarios:
        raise RuntimeError('No co-simulation jobRun.xml files were available for plotting.')

    subplot_rows = 4
    subplot_cols = 2
    fig, axes = plt.subplots(subplot_rows, subplot_cols, figsize=(22, 18), sharex=False)
    flat_axes = axes.flatten()

    for ax, (_, label, color, host_to_intervals, host_to_network_intervals) in zip(flat_axes, parsed_scenarios):
        draw_execution_timeline(ax, host_to_intervals, host_to_network_intervals, label, color)

    for ax in flat_axes[len(parsed_scenarios):]:
        ax.axis('off')

    for ax in axes[-1, :]:
        ax.set_xlabel('Co-simulation time (ms)')

    fig.suptitle('Same AI Workload under Different Parallelization Strategies', fontsize=15, y=0.995)
    fig.text(
        0.5,
        0.01,
        'Rows compare parallel strategies; columns compare chain vs direct p2p at 25Gbps. '
        'Colored bars are GPUSim task intervals; darker bars mean higher NeuSight-derived GPU utilization. '
        'Blank gaps include scheduling, dependency, pipeline, and network waits.',
        ha='center',
        fontsize=10,
        color='#444444',
    )
    plt.tight_layout(rect=(0, 0.035, 1, 0.985))

    out_path = Path('/Users/taco/Desktop/cosim/case_study_results/ai_validation/utilization_timeline.png')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    print(f'\nSaved: {out_path}')

    print('\n' + '=' * 112)
    print(f"{'Scenario':<34s} {'GPUs':>6s} {'Makespan(ms)':>14s} {'TotalBusy(ms)':>15s} {'AvgBusy':>10s} {'AvgUtil':>10s} {'NetWait(ms)':>13s}")
    print('-' * 112)
    for directory_name, _, _, host_to_intervals, host_to_network_intervals in parsed_scenarios:
        wall_time, total_busy_time, average_busy_ratio, total_network_wait = summarize_timeline(
            host_to_intervals,
            host_to_network_intervals,
        )
        _, utilization_points = build_average_utilization_curve(host_to_intervals)
        average_utilization = float(np.mean(utilization_points)) if len(utilization_points) else 0.0
        print(f'{directory_name:<34s} {len(host_to_intervals):>6d} {wall_time:>14.2f} {total_busy_time:>15.2f} {average_busy_ratio * 100:>9.1f}% {average_utilization * 100:>9.1f}% {total_network_wait:>13.2f}')
    print('=' * 112)


if __name__ == '__main__':
    main()
