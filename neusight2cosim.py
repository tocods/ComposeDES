#!/usr/bin/env python3
"""
NeuSight CSV → 协同仿真输入文件转换器

将 NeuSight 预测的 per-kernel fw_latency 转换为 GPUSim + NS-3 的输入文件：
  - hosts.json: GPU 主机配置 (flops_per_core=1 使执行时间=thread_length)
  - jobs.json: 任务 DAG (计算 + 通信)
  - topology.yaml: NS-3 网络拓扑
"""

import csv
import json
import os
import argparse
import math
import ast


COMMUNICATION_OPS = {'ALLREDUCE', 'ALLREDUCE_ASYNC', 'SENDRECV'}


def reduce_mul(values):
    result = 1
    for value in values:
        result *= value
    return result


def parse_ops_field(value):
    try:
        parsed_value = ast.literal_eval(value or '[]')
    except Exception:
        return []
    return parsed_value if isinstance(parsed_value, list) else []


def row_contains_communication(row):
    op_name = str(row.get('OpName', '')).upper()
    if op_name in COMMUNICATION_OPS or op_name in {'ALLREDUCE', 'SENDRECV'}:
        return True

    for ops_field in ('FwOps', 'BwOps', 'AccOps'):
        for op in parse_ops_field(row.get(ops_field, '[]')):
            if not isinstance(op, (list, tuple)) or not op:
                continue
            if str(op[0]).upper() in COMMUNICATION_OPS:
                return True
    return False


def extract_communication_metadata_list(row, include_all=False):
    markers = []
    for ops_field in ('FwOps', 'BwOps', 'AccOps'):
        for op in parse_ops_field(row.get(ops_field, '[]')):
            if not isinstance(op, (list, tuple)) or not op:
                continue
            communication_op = str(op[0]).upper()
            if communication_op not in COMMUNICATION_OPS:
                continue
            args = op[1] if len(op) > 1 else ()
            element_count = int(args[0]) if isinstance(args, (list, tuple)) and args else 0
            markers.append((communication_op, element_count * 4))
            if not include_all:
                return markers

    if markers:
        return markers

    op_name = str(row.get('OpName', '')).upper()
    if op_name in COMMUNICATION_OPS:
        return [(op_name, 0)]
    return []


def compute_neusight_kernel_utilization(row, latency_field):
    """Estimate the utilization used for co-sim timeline visualization.

    NeuSight predicts latency with a learned effective-bandwidth model:
      latency = num_wave * ops_per_wave / (roofline_bw * predicted_utilization)
    The released CSV does not persist predicted_utilization, so this converter
    reconstructs a comparable per-kernel utilization from achieved throughput.
    """
    latency_ms = float(row.get(latency_field, 0) or 0)
    if latency_ms <= 0:
        return 0.0

    device_peak_flops = 66908e9
    device_memory_bw = 3430e9
    op_name = row.get('OpName', '')

    try:
        fw_ops = parse_ops_field(row.get('FwOps', '[]'))
    except Exception:
        fw_ops = []

    total_flops = 0.0
    total_bytes = 0.0
    for op in fw_ops:
        if not isinstance(op, (list, tuple)) or len(op) != 2:
            continue
        name, args = op
        if name == 'Linear' and len(args) == 3:
            matrix_m, matrix_n, matrix_k = args
            total_flops += 2 * matrix_m * matrix_n * matrix_k
        elif name == 'BMM' and len(args) == 4:
            batch, matrix_m, matrix_n, matrix_k = args
            total_flops += 2 * batch * matrix_m * matrix_n * matrix_k
        elif str(name).startswith('VEC'):
            if len(args) == 2:
                batch, hidden = args
                total_bytes += batch * hidden * 2
            elif len(args) == 1:
                total_bytes += args[0] * 2
        elif name == 'MEM' and isinstance(args, (list, tuple)):
            for shape in args:
                if isinstance(shape, (list, tuple)):
                    total_bytes += reduce_mul(shape) * 2

    latency_seconds = latency_ms * 1e-3
    if op_name in ('Linear', 'BMM') and total_flops > 0:
        return min(total_flops / latency_seconds / device_peak_flops, 1.0)
    if str(op_name).startswith('VEC') and total_bytes > 0:
        return min(total_bytes / latency_seconds / device_memory_bw, 1.0)
    if op_name in ('ALLREDUCE', 'ALLREDUCE_ASYNC', 'SENDRECV'):
        return 0.0
    return 0.0


def parse_neusight_csv(csv_path, use_e2e=False, skip_communication=True):
    """解析 NeuSight 预测 CSV，返回 compute kernel 与可选通信 marker。

    use_e2e=True 时使用 e2e_latency（fw+bw+acc，适用于 training），
    否则使用 fw_latency（inference）。NeuSight CSV 中的通信行包含公式预测延迟；
    co-sim 转换时只把通信行作为 NS-3 的通信位置与通信量 marker，
    不把 NeuSight 公式通信延迟计入 GPU compute task。
    """
    kernels = []
    skipped_communication_rows = 0
    field = 'e2e_latency' if use_e2e else 'fw_latency'
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row_contains_communication(row):
                if skip_communication:
                    skipped_communication_rows += 1
                    continue
                communication_markers = extract_communication_metadata_list(row, include_all=use_e2e)
                for marker_index, (communication_op, communication_bytes) in enumerate(communication_markers):
                    marker_name = row['Name'] if marker_index == 0 else f"{row['Name']}_{marker_index}"
                    kernels.append({
                        'name': marker_name,
                        'op': row['OpName'],
                        'fw_latency_ms': 0.0,
                        'utilization': 0.0,
                        'is_communication': True,
                        'communication_op': communication_op,
                        'communication_bytes': communication_bytes,
                    })
                continue

            v = float(row[field])
            if v > 0:
                kernels.append({
                    'name': row['Name'],
                    'op': row['OpName'],
                    'fw_latency_ms': v,
                    'utilization': compute_neusight_kernel_utilization(row, field),
                    'is_communication': False,
                    'communication_op': '',
                    'communication_bytes': 0,
                })
    return kernels, skipped_communication_rows


def sum_neusight_prediction_ms(csv_path, num_layers, use_e2e=False):
    """Sum NeuSight's own predicted latency, including formula-based communication rows."""
    field = 'e2e_latency' if use_e2e else 'fw_latency'
    kernels = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            latency_ms = float(row.get(field, 0) or 0)
            if latency_ms > 0:
                kernels.append({
                    'name': row['Name'],
                    'fw_latency_ms': latency_ms,
                })
    layers = group_kernels_by_layer(kernels, num_layers)
    return sum(kernel['fw_latency_ms'] for _, layer_kernels in layers for kernel in layer_kernels)


def group_kernels_by_layer(kernels, num_layers):
    """
    将 kernel 按 transformer 层分组。

    自动检测 CSV 是否已展开多层：
      - 如果出现 transformer_h_1_* / transformer_h_2_* → 已展开，
        按层名拆分，不再复制
      - 否则认为是 single-layer trace，复制 single_layer_kernels 到所有层
    """
    import re
    # 支持多种命名约定：
    #   transformer_h_N_  (GPT-2, GPT-3, LLaMA)
    #   bert_encoder_layer_N_  (BERT)
    #   model_decoder_layers_N_  (OPT)
    layer_idx_pat = re.compile(r'(?:transformer_h|bert_encoder_layer|model_decoder_layers)_(\d+)_')
    max_layer_seen = -1
    for k in kernels:
        m = layer_idx_pat.search(k['name'])
        if m:
            max_layer_seen = max(max_layer_seen, int(m.group(1)))

    if max_layer_seen >= 1:
        # already unrolled. CSV 中只有少量 op (LayerNorm) 带 transformer_h_<i>_
        # 前缀，主计算 op (addmm/matmul 等) 是 fx 自动名不带前缀。
        # 按 CSV 顺序：从 transformer_h_<i>_ln_1 错出 layer i 起点，
        # 到下一个 layer (i+1) 的 ln_1 或 transformer_ln_f 为止全归 layer i。
        embed_kernels = []
        head_kernels = []
        per_layer = {i: [] for i in range(max_layer_seen + 1)}
        cur = -1  # -1 = embed, max_layer_seen+1 = head
        first_layer_seen = False
        # 通用 layer marker 模式
        layer_marker_pat = re.compile(r'(?:transformer_h|bert_encoder_layer|model_decoder_layers)_(\d+)_')
        for k in kernels:
            n = k['name']
            m_marker = layer_marker_pat.match(n)
            if m_marker:
                layer_id = int(m_marker.group(1))
                # 检测是否是该层第一个 kernel（用于切换 cur）
                # 对 transformer: _ln_1 结尾；对 BERT/OPT: 任何 layer_N_ 开头的 kernel 如果 cur != layer_id
                is_layer_start = False
                if re.match(r'transformer_h_\d+_ln_1$', n):
                    is_layer_start = True
                elif cur != layer_id:
                    is_layer_start = True
                if is_layer_start:
                    # 第一次遇到 layer 0 时，把 embed 中的大 kernel 移回 layer 0
                    if not first_layer_seen and layer_id == 0:
                        first_layer_seen = True
                        real_embed = []
                        for ek in embed_kernels:
                            if ek['fw_latency_ms'] > 0.01:
                                per_layer[0].append(ek)
                            else:
                                real_embed.append(ek)
                        embed_kernels = real_embed
                    cur = layer_id
                per_layer[cur].append(k)
                continue
            if 'transformer_ln_f' in n or 'lm_head' in n:
                cur = max_layer_seen + 2  # head
                head_kernels.append(k)
                continue
            if cur == -1:
                embed_kernels.append(k)
            elif cur == max_layer_seen + 2:
                head_kernels.append(k)
            else:
                per_layer[cur].append(k)

        layers = []
        if embed_kernels:
            layers.append(('embed', embed_kernels))
        for i in sorted(per_layer.keys()):
            if per_layer[i]:
                layers.append((f'layer_{i}', per_layer[i]))
        if head_kernels:
            layers.append(('head', head_kernels))
        return layers

    # single_layer trace: replicate to num_layers
    embed_kernels = []
    single_layer_kernels = []
    head_kernels = []
    in_layer = False
    for k in kernels:
        name = k['name']
        if 'transformer_h_' in name or 'bert_encoder_layer_' in name or 'model_decoder_layers_' in name:
            in_layer = True
            single_layer_kernels.append(k)
        elif in_layer and ('ln_f' in name or 'lm_head' in name):
            in_layer = False
            head_kernels.append(k)
        elif in_layer:
            single_layer_kernels.append(k)
        elif 'wte' in name or 'wpe' in name or 'embed' in name.lower():
            embed_kernels.append(k)
        elif 'ln_f' in name or 'lm_head' in name:
            head_kernels.append(k)
        else:
            embed_kernels.append(k)

    layers = []
    if embed_kernels:
        layers.append(('embed', embed_kernels))
    for i in range(num_layers):
        layers.append((f'layer_{i}', single_layer_kernels))
    if head_kernels:
        layers.append(('head', head_kernels))
    return layers


def get_communication_markers(kernels, communication_ops):
    if isinstance(communication_ops, str):
        expected_ops = {communication_ops}
    else:
        expected_ops = set(communication_ops)
    return [
        kernel for kernel in kernels
        if kernel.get('is_communication')
        and kernel.get('communication_op') in expected_ops
    ]


def get_allreduce_transfer_bytes(marker, tp):
    marker_bytes = int(marker.get('communication_bytes') or 0)
    if marker_bytes <= 0:
        raise ValueError('ALLREDUCE marker is missing communication byte size in NeuSight CSV')
    return marker_bytes * (tp - 1)


def get_sendrecv_transfer_bytes(marker):
    marker_bytes = int(marker.get('communication_bytes') or 0)
    if marker_bytes <= 0:
        raise ValueError('SENDRECV marker is missing communication byte size in NeuSight CSV')
    return marker_bytes


def add_network_children(jobs, source_task, child_name, packet_size, source_host, destination_host):
    for job in jobs:
        if job['name'] == source_task:
            job['children'].append({
                'child': child_name,
                'size': int(packet_size),
                'src_host': source_host,
                'dst_host': destination_host,
            })
            return


def add_local_child(jobs, source_task, child_name):
    for job in jobs:
        if job['name'] == source_task:
            job['children'].append({'child': child_name})
            return


def generate_jobs(layers, tp, pp, num_layers, hidden_dim,
                  bandwidth_gbps=200, delay_us=1, batch_size=2, seq_len=2048):
    """
    生成 jobs.json 内容

    TP: 每层的 kernel 在 TP 组内各 GPU 上并行执行，层后必须按 NeuSight CSV 的
    AllReduce marker 生成 NS-3 通信边；缺少 marker 或通信字节时直接报错。
    TP 下的计算延迟必须来自 NeuSight TP-aware profile，不能在转换器里用 base profile 除以 TP。
    PP: 不同层分配到不同 pipeline stage，stage 间必须按 CSV 的 SENDRECV marker 生成 NS-3 通信边；
    缺少 marker 或通信字节时直接报错。
    """
    total_gpus = tp * pp
    BW_BPS = bandwidth_gbps * 1e9  # Gbps → bps
    jobs = []
    task_id = 0

    # PP: 分配层到不同 stage
    layers_data = [l for l in layers if l[0].startswith('layer_')]
    embed_layer = [l for l in layers if l[0] == 'embed']
    head_layer = [l for l in layers if l[0] == 'head']

    layers_per_stage = max(1, math.ceil(len(layers_data) / pp))

    # 跟踪每个 GPU 上一个完成的 task（用于链接依赖）
    prev_task_per_gpu = {}

    for stage_idx in range(pp):
        stage_layers = layers_data[stage_idx * layers_per_stage: (stage_idx + 1) * layers_per_stage]
        if stage_idx == 0 and embed_layer:
            stage_layers = embed_layer + stage_layers
        if stage_idx == pp - 1 and head_layer:
            stage_layers = stage_layers + head_layer

        for layer_name, kernels in stage_layers:
            # 每层的总计算时间来自 NeuSight 当前 profile。
            # TP 场景必须传入 NeuSight --options tpN 生成的 TP-aware CSV，
            # 转换器禁止用 base profile / TP 做近似拆分。
            total_latency_us = sum(k['fw_latency_ms'] * 1000 for k in kernels)
            layer_utilization = 0.0
            if total_latency_us > 0:
                layer_utilization = sum(
                    k.get('utilization', 0.0) * k['fw_latency_ms'] * 1000 for k in kernels
                ) / total_latency_us
            per_gpu_latency_us = max(1, int(round(total_latency_us)))

            # 为 TP 组内每个 GPU 创建计算任务
            layer_tasks = []
            for tp_rank in range(tp):
                gpu_idx = stage_idx * tp + tp_rank
                host_name = f'host{gpu_idx + 1}'
                task_name = f'{layer_name}_r{gpu_idx}'

                job = {
                    'name': task_name,
                    'period': '1',
                    'cpu_task': {'ram': '100', 'pes_number': 1, 'length': 10},
                    'gpu_task': {
                        'kernels': [{
                            'block_num': 1,
                            'thread_num': 1,
                            'thread_length': per_gpu_latency_us,
                            'hardware': 'CPU',
                            'requested_gddram_size': 0,
                            'task_input_size': 0,
                            'task_output_size': 0,
                            'utilization': round(layer_utilization, 6),
                            'type': '浮点',
                            'calcuType': 0
                        }],
                        'requested_gddram_size': 0,
                        'task_input_size': 0,
                        'task_output_size': 0
                    },
                    'host': host_name,
                    'ifManager': False,
                    'ifInMaster': False,
                    'deadline': '999999999',
                    'appArgs': [],
                    'children': [],
                    'middlewareArgs': []
                }

                # 链接到同 GPU 上一个 task（同主机顺序依赖，无需网络传输）
                if gpu_idx in prev_task_per_gpu:
                    prev_name = prev_task_per_gpu[gpu_idx]
                    for j in jobs:
                        if j['name'] == prev_name:
                            j['children'].append({'child': task_name})
                            break

                prev_task_per_gpu[gpu_idx] = task_name
                layer_tasks.append((task_name, gpu_idx))
                jobs.append(job)
                task_id += 1

            # TP > 1: 层内/层后 AllReduce（同步激活值）。
            # NeuSight CSV 中的 ALLREDUCE/ALLREDUCE_ASYNC 行只作为 marker：
            #   - fw_latency_ms 已在 parse 阶段置 0，不计入 compute baseline；
            #   - communication_bytes 只用于决定 NS-3 packet size。
            if tp > 1:
                allreduce_markers = get_communication_markers(kernels, ('ALLREDUCE', 'ALLREDUCE_ASYNC'))
                if not allreduce_markers and layer_name.startswith('layer_'):
                    raise ValueError(
                        f'TP={tp} requires ALLREDUCE markers with byte sizes in NeuSight CSV for {layer_name}'
                    )

                if allreduce_markers:
                    total_allreduce_bytes = sum(
                        get_allreduce_transfer_bytes(marker, tp)
                        for marker in allreduce_markers
                    )

                    for src_rank in range(tp):
                        src_gpu = stage_idx * tp + src_rank
                        dst_gpu = stage_idx * tp + ((src_rank + 1) % tp)
                        src_task = layer_tasks[src_rank][0]
                        src_host = f'host{src_gpu + 1}'
                        dst_host = f'host{dst_gpu + 1}'
                        for j in jobs:
                            if j['name'] == src_task:
                                j['children'].append({
                                    'child': f'__ar_{src_task}_ring__',
                                    'size': total_allreduce_bytes,
                                    'src_host': src_host,
                                    'dst_host': dst_host
                                })
                                break

        # PP > 1: pipeline stage 之间的通信或依赖
        if stage_idx < pp - 1 and stage_layers:
            next_stage_start_gpu = (stage_idx + 1) * tp
            sendrecv_markers = []
            for _, stage_layer_kernels in stage_layers:
                sendrecv_markers.extend(get_communication_markers(stage_layer_kernels, 'SENDRECV'))
            if not sendrecv_markers:
                raise ValueError(
                    f'PP={pp} requires SENDRECV markers with byte sizes between stage {stage_idx} and {stage_idx + 1}'
                )
            activation_size = sum(
                get_sendrecv_transfer_bytes(marker)
                for marker in sendrecv_markers
            )

            for tp_rank in range(tp):
                src_gpu = stage_idx * tp + tp_rank
                dst_gpu = next_stage_start_gpu + tp_rank
                src_task = prev_task_per_gpu[src_gpu]
                src_host = f'host{src_gpu + 1}'
                dst_host = f'host{dst_gpu + 1}'
                for j in jobs:
                    if j['name'] == src_task:
                        next_stage_first_layer = layers_data[(stage_idx + 1) * layers_per_stage]
                        next_task = f'{next_stage_first_layer[0]}_r{dst_gpu}'
                        j['children'].append({
                            'child': next_task,
                            'size': activation_size,
                            'src_host': src_host,
                            'dst_host': dst_host
                        })
                        break

    return jobs


def generate_hosts(total_gpus):
    """生成 hosts.json: flops_per_core=1 使执行时间=thread_length"""
    hosts = []
    for i in range(total_gpus):
        hosts.append({
            'name': f'host{i + 1}',
            'video_card_infos': [{
                'gpu_infos': [{
                    'cores': 1,
                    'core_per_sm': 1,
                    'max_block_per_sm': 1,
                    'gddram': 80000,
                    'flops_per_core': 1,
                    'int_flops_per_core': 1,
                    'matrix_flops_per_core': 1,
                    'bw': 64
                }],
                'name': 'GPU',
                'pcie_bw': 64
            }],
            'cpu_infos': [{'cores': 1, 'mips': 1, 'int_mips': 1, 'matrix_mips': 1}],
            'ram': 1024,
            'ifMaster': i == 0
        })
    return hosts


def generate_topology(total_gpus, tp, bandwidth_gbps=200, delay_us=1, topo_type='chain'):
    """生成 topology.yaml"""
    lines = ['sim:', '  stop: 86400s', '', 'nodes:']

    if topo_type == 'chain':
        # p2p chain（和 HPC goal2cosim 一致，NS-3 验证通过）
        for i in range(total_gpus):
            lines.append(f'  - id: host{i + 1}')
            lines.append('    type: host')
        lines.append('')
        lines.append('links:')
        for i in range(total_gpus - 1):
            lines.append(f'  - type: p2p')
            lines.append(f'    endpoints: [host{i + 1}, host{i + 2}]')
            lines.append(f'    dataRate: {bandwidth_gbps}Gbps')
            lines.append(f'    delay: {delay_us}us')

    elif topo_type == 'star':
        if total_gpus == 2:
            lines.append('  - id: host1')
            lines.append('    type: host')
            lines.append('  - id: host2')
            lines.append('    type: host')
            lines.append('')
            lines.append('links:')
            lines.append('  - type: p2p')
            lines.append('    endpoints: [host1, host2]')
            lines.append(f'    dataRate: {bandwidth_gbps}Gbps')
            lines.append(f'    delay: {delay_us}us')
        else:
            lines.append('  - id: s0')
            lines.append('    type: switch')
            for i in range(total_gpus):
                lines.append(f'  - id: host{i + 1}')
                lines.append('    type: host')
            lines.append('')
            lines.append('links:')
            for i in range(total_gpus):
                lines.append(f'  - type: csma')
                lines.append(f'    endpoints: [host{i + 1}, s0]')
                lines.append(f'    dataRate: {bandwidth_gbps}Gbps')
                lines.append(f'    delay: {delay_us}us')

    elif topo_type == 'dual_switch':
        lines.append('  - id: s0')
        lines.append('    type: switch')
        lines.append('  - id: s1')
        lines.append('    type: switch')
        for i in range(total_gpus):
            lines.append(f'  - id: host{i + 1}')
            lines.append('    type: host')
        lines.append('')
        lines.append('links:')
        # Switch interconnect
        lines.append('  - type: csma')
        lines.append('    endpoints: [s0, s1]')
        lines.append(f'    dataRate: {bandwidth_gbps}Gbps')
        lines.append(f'    delay: 5us')
        # Hosts to switches (first half to s0, second half to s1)
        half = total_gpus // 2
        for i in range(total_gpus):
            sw = 's0' if i < half else 's1'
            lines.append(f'  - type: csma')
            lines.append(f'    endpoints: [host{i + 1}, {sw}]')
            lines.append(f'    dataRate: {bandwidth_gbps}Gbps')
            lines.append(f'    delay: {delay_us}us')

    elif topo_type == 'direct_p2p':
        for i in range(total_gpus):
            lines.append(f'  - id: host{i + 1}')
            lines.append('    type: host')
        lines.append('')
        lines.append('links:')
        # 全互联 p2p（类似 NVSwitch）
        for i in range(total_gpus):
            for j in range(i + 1, total_gpus):
                lines.append(f'  - type: p2p')
                lines.append(f'    endpoints: [host{i + 1}, host{j + 1}]')
                lines.append(f'    dataRate: {bandwidth_gbps}Gbps')
                lines.append(f'    delay: {delay_us}us')

    lines.append('')
    lines.append('apps:')
    for i in range(total_gpus):
        lines.append(f'  - type: fncs')
        lines.append(f'    node: host{i + 1}')
        lines.append(f'    name: host{i + 1}')

    return '\n'.join(lines) + '\n'


def convert(csv_path, tp=1, pp=1, num_layers=32, hidden_dim=2560,
            bandwidth_gbps=200, topo_type='star', output_dir='./output',
            use_e2e=False, dp=1, delay_us=1,
            batch_size=2, seq_len=2048):
    """完整转换流程。

    Inference: dp 仅复制独立 replica，不生成梯度 AllReduce。
    Training: use_e2e=True 时才会生成 DP 梯度同步边。
    """
    os.makedirs(output_dir, exist_ok=True)
    total_gpus = tp * pp * dp

    # 1. 解析 NeuSight CSV
    import re
    csv_base = os.path.splitext(os.path.basename(csv_path))[0]
    tp_profile_match = re.search(r'-tp(\d+)(?:-|$)', csv_base)
    pp_profile_match = re.search(r'-pp(\d+)_\d+(?:-|$)', csv_base)
    csv_tp_degree = int(tp_profile_match.group(1)) if tp_profile_match else 1
    csv_pp_degree = int(pp_profile_match.group(1)) if pp_profile_match else 1
    uses_distributed_profile = csv_tp_degree > 1 or csv_pp_degree > 1
    kernels, skipped_communication_rows = parse_neusight_csv(
        csv_path,
        use_e2e=use_e2e,
        skip_communication=not uses_distributed_profile,
    )
    label = 'e2e' if use_e2e else 'fw'
    print(f'  Parsed {len(kernels)} kernels, skipped {skipped_communication_rows} communication rows, total {label}_latency={sum(k["fw_latency_ms"] for k in kernels):.2f}ms')

    # 2. 按层分组
    layers = group_kernels_by_layer(kernels, num_layers)
    print(f'  Grouped into {len(layers)} layers')

    # 3. 生成 jobs.json
    if tp > 1 and csv_tp_degree != tp:
        raise ValueError(
            f'TP={tp} requires NeuSight TP-aware profile generated with --options tp{tp}; '
            f'got CSV "{csv_base}". Do not use base profile divided by TP.'
        )
    if pp > 1 and csv_pp_degree != pp:
        raise ValueError(
            f'PP={pp} requires NeuSight PP-aware profile generated with --options pp{pp}_*; '
            f'got CSV "{csv_base}". Do not use base profile with synthetic PP split.'
        )
    if uses_distributed_profile:
        print('  Using NeuSight distributed profile: communication rows are NS-3 markers only')
    elif tp > 1:
        print(f'  Detected NeuSight TP-aware profile: tp{csv_tp_degree}; keep predicted per-rank compute latency')
    else:
        print('  Using NeuSight profile compute latency unchanged')

    jobs = generate_jobs(layers, tp, pp, num_layers, hidden_dim,
                         bandwidth_gbps=bandwidth_gbps, delay_us=delay_us,
                         batch_size=batch_size, seq_len=seq_len)
    with open(os.path.join(output_dir, 'jobs.json'), 'w') as f:
        json.dump(jobs, f, indent=2)
    print(f'  Generated {len(jobs)} tasks in jobs.json')

    # 3.5 DP 复制：把 tp×pp 任务在另外 dp-1 份主机上复制。
    # 推理 DP 没有梯度同步；训练 DP 才在 replica 末尾生成梯度 AllReduce。
    if dp > 1:
        base_jobs = list(jobs)
        gpus_per_replica = tp * pp
        for d in range(1, dp):
            offset = d * gpus_per_replica
            for j in base_jobs:
                nj = json.loads(json.dumps(j))
                nj['name'] = f"{j['name']}_dp{d}"
                # remap host
                old_idx = int(nj['host'].replace('host', '')) - 1
                nj['host'] = f'host{old_idx + offset + 1}'
                # remap children: same replica internal deps + remap hosts
                for c in nj.get('children', []):
                    if 'src_host' in c:
                        sidx = int(c['src_host'].replace('host', '')) - 1
                        c['src_host'] = f'host{sidx + offset + 1}'
                    if 'dst_host' in c:
                        didx = int(c['dst_host'].replace('host', '')) - 1
                        c['dst_host'] = f'host{didx + offset + 1}'
                    if not c['child'].startswith('__') and not c['child'].endswith(f'_dp{d}'):
                        c['child'] = f"{c['child']}_dp{d}"
                jobs.append(nj)

        if not use_e2e:
            print(f'  Inference DP={dp}: replicated independent replicas to {total_gpus} GPUs without gradient AllReduce')
        else:
            raise ValueError(
                'Training DP requires explicit DP communication markers in the NeuSight profile. '
                'Refusing to synthesize gradient AllReduce bytes from model dimensions.'
            )

    with open(os.path.join(output_dir, 'jobs.json'), 'w') as fout:
        json.dump(jobs, fout, indent=2)

    # 4. 生成 hosts.json
    hosts = generate_hosts(total_gpus)
    with open(os.path.join(output_dir, 'hosts.json'), 'w') as f:
        json.dump(hosts, f, indent=2)

    # 5. 生成 topology.yaml
    topo = generate_topology(total_gpus, tp, bandwidth_gbps, topo_type=topo_type)
    with open(os.path.join(output_dir, 'topology.yaml'), 'w') as f:
        f.write(topo)

    # 6. 生成 faults.json (空)
    with open(os.path.join(output_dir, 'faults.json'), 'w') as f:
        json.dump([], f)

    return jobs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NeuSight CSV → Co-simulation input converter')
    parser.add_argument('--csv', required=True, help='Path to NeuSight prediction CSV')
    parser.add_argument('--tp', type=int, default=1, help='Tensor Parallelism degree')
    parser.add_argument('--pp', type=int, default=1, help='Pipeline Parallelism degree')
    parser.add_argument('--num-layers', type=int, default=32, help='Number of transformer layers')
    parser.add_argument('--hidden-dim', type=int, default=2560, help='Hidden dimension (n_embd)')
    parser.add_argument('--bandwidth', type=int, default=200, help='Link bandwidth in Gbps')
    parser.add_argument('--topology', default='chain', choices=['chain', 'star', 'dual_switch', 'direct_p2p'])
    parser.add_argument('--output-dir', default='./cosim_input', help='Output directory')
    parser.add_argument('--dp', type=int, default=1, help='Data Parallelism degree')
    parser.add_argument('--use-e2e', action='store_true', help='Use e2e_latency (training)')
    parser.add_argument('--batch-size', type=int, default=0,
                        help='Training batch size (0=auto-extract from CSV filename)')
    parser.add_argument('--seq-len', type=int, default=0,
                        help='Sequence length (0=auto-extract from CSV filename)')
    args = parser.parse_args()

    # Auto-extract batch_size and seq_len from filename if not specified
    # NeuSight CSV naming: {model}-{mode}-{seq_len}-{batch_size}[-{parallel}].csv
    import re as _re
    csv_base = os.path.splitext(os.path.basename(args.csv))[0]
    _seq_bat = _re.search(r'-(\d+)-(\d+)(?:-|$)', csv_base)
    auto_seq = int(_seq_bat.group(1)) if _seq_bat else 2048
    auto_bat = int(_seq_bat.group(2)) if _seq_bat else 2
    seq_len = args.seq_len if args.seq_len > 0 else auto_seq
    batch_size = args.batch_size if args.batch_size > 0 else auto_bat

    print(f'Converting: {args.csv}')
    print(f'Config: TP={args.tp}, PP={args.pp}, DP={args.dp}, layers={args.num_layers}, hidden={args.hidden_dim}, use_e2e={args.use_e2e}')
    mode_label = 'training' if args.use_e2e else 'inference'
    print(f'{mode_label.capitalize()}: batch_size={batch_size}, seq_len={seq_len}')
    print(f'Network: {args.bandwidth}Gbps, topology={args.topology}')
    convert(args.csv, args.tp, args.pp, args.num_layers, args.hidden_dim,
            args.bandwidth, args.topology, args.output_dir,
            use_e2e=args.use_e2e, dp=args.dp,
            delay_us=1, batch_size=batch_size, seq_len=seq_len)
    print('Done!')
