#!/usr/bin/env python3
"""
ATLAHS GOAL → 协同仿真输入转换器

将 ATLAHS 的 GOAL trace（细粒度 calc/send/recv + 依赖关系）聚合为
GPUSim 的 jobs.json + hosts.json + topology.yaml。

聚合策略：
  1. 对每个 rank，找到所有 send/recv 操作的时序（通过依赖链拓扑排序）
  2. 将操作按"通信轮次"分组：每轮通信之间的 calc 合并为一个计算任务
  3. 每轮通信的 send 映射为跨主机 packet
"""

import re
import json
import os
import sys
from collections import defaultdict


def parse_goal(goal_path):
    """解析 GOAL 文件，返回 {rank_id: {label: op_dict}}"""
    with open(goal_path) as f:
        content = f.read()

    num_ranks = int(re.search(r'num_ranks (\d+)', content).group(1))
    rank_bodies = re.split(r'rank \d+ \{', content)[1:]

    ranks = {}
    for rank_id in range(num_ranks):
        body = rank_bodies[rank_id].split('}')[0]
        ops = {}
        for line in body.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            # 操作行 — cpu/nic 在 HPC GOAL 中可能缺失，故均为可选
            m = re.match(
                r'(l\d+): (calc|send|recv) (\d+)b?\s*'
                r'(?:(?:from|to) (\d+)\s*)?'
                r'(?:tag (\S+)\s*)?'
                r'(?:cpu (\d+))?\s*'
                r'(?:nic (\d+))?',
                line
            )
            if m:
                label, op_type, value, peer, tag, cpu, nic = m.groups()
                ops[label] = {
                    'type': op_type,
                    'value': int(value),
                    'cpu': int(cpu) if cpu else 0,
                    'peer': int(peer) if peer else None,
                    'tag': tag,
                    'deps': [],
                    'label': label,
                }
                continue
            # 依赖行 (HPC GOAL 还有 irequires 形式)
            m2 = re.match(r'(l\d+) i?requires (l\d+)', line)
            if m2:
                child, parent = m2.groups()
                if child in ops:
                    ops[child]['deps'].append(parent)
        ranks[rank_id] = ops
    return num_ranks, ranks


def aggregate_rank(ops):
    """
    将一个 rank 的细粒度操作聚合为粗粒度阶段。

    策略：按拓扑排序遍历，将连续的 calc 合并，遇到 send/recv 时切分阶段。
    返回 [(phase_type, value, peer, tag), ...]:
      - ('calc', total_ns, None, None)
      - ('send', size_bytes, dst_rank, tag)
      - ('recv', size_bytes, src_rank, tag)
    """
    # 简化：不做完整拓扑排序（1.1M 操作太慢）
    # 改为：统计总 calc 时间和总 send/recv 次数，按通信轮次均分计算

    total_calc_ns = 0
    sends = []  # (size, peer, tag)
    recvs = []

    for label, op in ops.items():
        if op['type'] == 'calc':
            total_calc_ns += op['value']
        elif op['type'] == 'send':
            sends.append((op['value'], op['peer'], op['tag']))
        elif op['type'] == 'recv':
            recvs.append((op['value'], op['peer'], op['tag']))

    return total_calc_ns, sends, recvs


def estimate_critical_path_compute(ops):
    """
    估算关键路径上的计算时间。
    返回 (总关键路径时间, [各阶段时间列表])

    策略：找主计算 CPU（calc 总量最大），将其大 calc block (>10ms) 识别为独立训练步。
    """
    cpu_calc = defaultdict(int)
    cpu_big_blocks = defaultdict(list)

    for label, op in ops.items():
        if op['type'] == 'calc':
            cpu_calc[op['cpu']] += op['value']
            if op['value'] > 10_000_000:  # >10ms 的大 block
                cpu_big_blocks[op['cpu']].append(op['value'])

    if not cpu_calc:
        return 0, [0]

    # 找主计算 CPU
    main_cpu = max(cpu_calc, key=cpu_calc.get)
    total = cpu_calc[main_cpu]

    # 将大 block 作为独立阶段，剩余小 calc 均分到各阶段间
    big_blocks = sorted(cpu_big_blocks.get(main_cpu, []), reverse=True)
    if not big_blocks:
        return total, [total]

    small_calc = total - sum(big_blocks)
    # 每个大 block 前后分配一些小 calc（代表 AllReduce kernel 开销）
    n_phases = len(big_blocks)
    small_per_phase = small_calc // max(n_phases, 1)

    phases = []
    for b in big_blocks:
        phases.append(b + small_per_phase)

    return total, phases


def goal_to_cosim(goal_path, bandwidth_gbps=200, delay_us=1, output_dir='./cosim_input'):
    """将 GOAL 文件转换为协同仿真输入"""
    os.makedirs(output_dir, exist_ok=True)

    print(f'Parsing GOAL file: {goal_path}')
    num_ranks, ranks = parse_goal(goal_path)
    print(f'  Ranks: {num_ranks}')

    # 聚合每个 rank
    jobs = []
    rank_stats = []

    for rank_id in range(num_ranks):
        ops = ranks[rank_id]
        total_calc, sends, recvs = aggregate_rank(ops)
        critical_calc, phases = estimate_critical_path_compute(ops)

        rank_stats.append({
            'rank': rank_id,
            'total_calc_ns': total_calc,
            'critical_calc_ns': critical_calc,
            'phases': phases,
            'num_sends': len(sends),
            'send_bytes': sum(s[0] for s in sends),
        })

        unique_peers = set(s[1] for s in sends)
        peer_total_bytes = defaultdict(int)
        for size, peer, tag in sends:
            peer_total_bytes[peer] += size
        host_name = f'host{rank_id + 1}'

        # 限制每 rank 最多 MAX_PHASES 个阶段，超出则合并（减少并发 task 数）
        MAX_PHASES = 20
        if len(phases) > MAX_PHASES:
            merged = []
            chunk = len(phases) // MAX_PHASES
            for i in range(MAX_PHASES):
                start = i * chunk
                end = (i + 1) * chunk if i < MAX_PHASES - 1 else len(phases)
                merged.append(sum(phases[start:end]))
            phases = merged

        n_phases = len(phases)
        sends_per_phase = len(sends) // max(n_phases, 1)

        # 网络建模策略：
        # - 少量 task (≤ 55) 的应用：每 phase 1 个 64KB 包（真实网络仿真）
        # - 大量 task 的应用：不发包（纯计算调度，避免 NS-3/FNCS 同步死锁）
        PKT_SIZE = 64 * 1024  # 64 KB
        total_bytes_all = sum(peer_total_bytes.values())
        total_tasks_estimate = n_phases * num_ranks
        enable_network = False  # 禁用 NS-3 网络仿真（避免 FNCS 同步死锁）

        # 解析通信延迟：直接加到 thread_length 里
        # 模拟相邻节点通信：bytes / bandwidth + propagation
        BANDWIDTH_BPS = args.bandwidth * 1e9  # 默认 200 Gbps
        PROP_DELAY_NS = args.delay * 1000     # 默认 1 μs = 1000 ns
        if total_bytes_all > 0 and n_phases > 0:
            per_phase_bytes = total_bytes_all / n_phases
            transfer_ns = per_phase_bytes * 8 / BANDWIDTH_BPS * 1e9
            comm_delay_ns = transfer_ns + PROP_DELAY_NS
        else:
            comm_delay_ns = 0

        prev_task = None
        for phase_idx, phase_ns in enumerate(phases):
            next_rank = (rank_id + 1) if (rank_id + 1) < num_ranks else rank_id - 1
            # 根据 task 规模决定是否发包
            if enable_network and next_rank != rank_id and total_bytes_all > 0:
                n_packets_total = 1
                packet_size = PKT_SIZE
            else:
                n_packets_total = 0
                packet_size = 0

            # 单 phase 单 task
            n_substeps = 1
            sub_phase_ns = phase_ns

            pkts_remaining = n_packets_total
            for sub_idx in range(n_substeps):
                task_name = f'rank{rank_id}_step{phase_idx}' if n_substeps == 1 \
                    else f'rank{rank_id}_step{phase_idx}_s{sub_idx}'

                children = []
                # 同 rank 内顺序依赖
                if prev_task:
                    for j in jobs:
                        if j['name'] == prev_task:
                            j['children'].append({'child': task_name})
                            break

                # 本 phase 挂的 packet（固定 0 或 1）
                sub_packets = pkts_remaining
                for pkt_idx in range(sub_packets):
                    children.append({
                        'child': f'__ar_r{rank_id}_s{phase_idx}_b{sub_idx}_p{pkt_idx}__',
                        'size': packet_size,
                        'src_host': host_name,
                        'dst_host': f'host{next_rank + 1}',
                    })
                pkts_remaining -= sub_packets

                job = {
                    'name': task_name,
                    'period': '1',
                    'cpu_task': {'ram': '100', 'pes_number': 1, 'length': 10},
                    'gpu_task': {
                        'kernels': [{
                            'block_num': 1, 'thread_num': 1,
                            'thread_length': max(int((sub_phase_ns + comm_delay_ns) / 1000), 1),  # (计算+通信) ns → us
                            'hardware': 'CPU',
                            'requested_gddram_size': 0,
                            'task_input_size': 0, 'task_output_size': 0,
                            'type': '浮点', 'calcuType': 0,
                        }],
                        'requested_gddram_size': 0,
                        'task_input_size': 0, 'task_output_size': 0,
                    },
                    'host': host_name,
                    'ifManager': False, 'ifInMaster': False,
                    'deadline': '999999999',
                    'appArgs': [], 'middlewareArgs': [],
                    'children': children,
                }
                jobs.append(job)
                prev_task = task_name

    # 输出统计
    print(f'\n  Per-rank statistics:')
    for s in rank_stats:
        print(f'    rank{s["rank"]}: critical_path={s["critical_calc_ns"]/1e6:.1f}ms, '
              f'sends={s["num_sends"]}, send_vol={s["send_bytes"]/1e6:.0f}MB')

    # 写 jobs.json
    with open(os.path.join(output_dir, 'jobs.json'), 'w') as f:
        json.dump(jobs, f, indent=2)
    print(f'\n  Generated {len(jobs)} tasks in jobs.json')

    # 写 hosts.json (flops_per_core=1 → thread_length 直接等于 ns)
    hosts = []
    for i in range(num_ranks):
        hosts.append({
            'name': f'host{i + 1}',
            'video_card_infos': [{'gpu_infos': [{
                'cores': 1, 'core_per_sm': 1, 'max_block_per_sm': 1,
                'gddram': 80000, 'flops_per_core': 1,
                'int_flops_per_core': 1, 'matrix_flops_per_core': 1, 'bw': 64,
            }], 'name': 'GPU', 'pcie_bw': 64}],
            'cpu_infos': [{'cores': 1, 'mips': 1, 'int_mips': 1, 'matrix_mips': 1}],
            'ram': 1024, 'ifMaster': i == 0,
        })
    with open(os.path.join(output_dir, 'hosts.json'), 'w') as f:
        json.dump(hosts, f, indent=2)

    # 写 topology.yaml
    topo_lines = ['sim:', '  stop: 86400s', '', 'nodes:']
    if num_ranks == 2:
        topo_lines += ['  - id: host1', '    type: host', '  - id: host2', '    type: host']
        topo_lines += ['', 'links:', '  - type: p2p',
                       '    endpoints: [host1, host2]',
                       f'    dataRate: {bandwidth_gbps}Gbps',
                       f'    delay: {delay_us}us']
    else:
        # 多节点：p2p 线性链 (host1-host2-host3-..., 避免回环路由问题)
        for i in range(num_ranks):
            topo_lines += [f'  - id: host{i+1}', '    type: host']
        topo_lines += ['', 'links:']
        for i in range(num_ranks - 1):
            topo_lines += [
                f'  - type: p2p',
                f'    endpoints: [host{i+1}, host{i+2}]',
                f'    dataRate: {bandwidth_gbps}Gbps',
                f'    delay: {delay_us}us',
            ]
    topo_lines += ['', 'apps:']
    for i in range(num_ranks):
        topo_lines += [f'  - type: fncs', f'    node: host{i+1}', f'    name: host{i+1}']
    with open(os.path.join(output_dir, 'topology.yaml'), 'w') as f:
        f.write('\n'.join(topo_lines) + '\n')

    # 写 faults.json
    with open(os.path.join(output_dir, 'faults.json'), 'w') as f:
        json.dump([], f)

    return jobs, rank_stats


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='ATLAHS GOAL → Co-simulation converter')
    parser.add_argument('--goal', required=True, help='Path to GOAL file')
    parser.add_argument('--bandwidth', type=int, default=200, help='Link bandwidth (Gbps)')
    parser.add_argument('--delay', type=int, default=1, help='Propagation delay (μs)')
    parser.add_argument('--output-dir', default='./cosim_from_goal', help='Output directory')
    args = parser.parse_args()

    goal_to_cosim(args.goal, args.bandwidth, args.delay, args.output_dir)
    print('Done!')
