#!/usr/bin/env python3
"""
协同仿真扩展能力 Case Study 实验脚本

实验 1: 变更 GPU 规格（NeuSight 预测不同 GPU 上的算子时间）
实验 2: 变更并行方式（TP/PP 配置）
实验 3: 变更网络拓扑
实验 4: 变更带宽
"""

import os
import sys
import json
import csv
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

COSIM_DIR = Path('/Users/taco/Desktop/cosim')
NEUSIGHT_DIR = COSIM_DIR / 'NeuSight'
NEUSIGHT_PRED_BASE = NEUSIGHT_DIR / 'scripts/asplos/results/prediction'
GPUSIM_DIR = COSIM_DIR / 'GPUsim'
NS3_DIR = COSIM_DIR / 'ns3'
FNCS_BROKER = COSIM_DIR / 'fncs/builds/local/bin/fncs_broker'
NS3_BIN = NS3_DIR / 'build/src/fncs/examples/ns3-dev-fncs-example'
RESULTS_DIR = COSIM_DIR / 'case_study_results'

NEUSIGHT_CSV = {
    'H100':     NEUSIGHT_PRED_BASE / 'NVIDIA_H100_80GB_HBM3/neusight/gpt3_27-inf-2048-2.csv',
    'A100_40G': NEUSIGHT_PRED_BASE / 'NVIDIA_A100-PCIE-40GB/neusight/gpt3_27-inf-2048-2.csv',
    'V100':     NEUSIGHT_PRED_BASE / 'Tesla_V100-PCIE-32GB/neusight/gpt3_27-inf-2048-2.csv',
    'T4':       NEUSIGHT_PRED_BASE / 'Tesla_T4/neusight/gpt3_27-inf-2048-2.csv',
    'L4':       NEUSIGHT_PRED_BASE / 'NVIDIA_L4/neusight/gpt3_27-inf-2048-2.csv',
}

NEUSIGHT_DEVICE_CONFIGS = NEUSIGHT_DIR / 'scripts/asplos/data/device_configs'

sys.path.insert(0, str(COSIM_DIR))
from neusight2cosim import convert

SIM_TIMEOUT = 120


def kill_simulation():
    os.system('pkill -f "fncs_broker|SimEngine|fncs-example" 2>/dev/null')
    time.sleep(0.5)


def run_neusight_prediction(device_config_path, result_dir):
    """运行 NeuSight 预测器，返回输出 CSV 路径"""
    env = os.environ.copy()
    env['PYTHONPATH'] = str(NEUSIGHT_DIR) + ':' + env.get('PYTHONPATH', '')
    cmd = [
        'python3', str(NEUSIGHT_DIR / 'scripts/pred.py'),
        '--predictor_name', 'neusight',
        '--predictor_path', str(NEUSIGHT_DIR / 'scripts/asplos/data/predictor/MLP_WAVE'),
        '--device_config_path', str(device_config_path),
        '--model_config_path', str(NEUSIGHT_DIR / 'scripts/asplos/data/DLmodel_configs/gpt3_27.json'),
        '--sequence_length', '2048',
        '--batch_size', '2',
        '--execution_type', 'inf',
        '--tile_dataset_dir', str(NEUSIGHT_DIR / 'scripts/asplos/data/dataset/train'),
        '--result_dir', str(result_dir),
        '--running_device', 'cpu:0',
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
    if proc.returncode != 0:
        print(f'    NeuSight error: {proc.stderr[-200:]}')
        return None
    device_name = json.load(open(device_config_path))['Device'].replace(' ', '_')
    csv_path = Path(result_dir) / 'prediction' / device_name / 'neusight' / 'gpt3_27-inf-2048-2.csv'
    if csv_path.exists():
        return str(csv_path)
    return None


def run_simulation(input_dir):
    """运行协同仿真并返回结果（通过 shell script 启动以正确传递环境变量）"""
    kill_simulation()

    jobs_path = os.path.join(input_dir, 'jobs.json')
    hosts_path = os.path.join(input_dir, 'hosts.json')
    faults_path = os.path.join(input_dir, 'faults.json')
    topo_path = os.path.join(input_dir, 'topology.yaml')
    output_dir = os.path.join(input_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    with open(hosts_path) as f:
        num_gpus = len(json.load(f))

    gpusim_zpl = os.path.join(input_dir, 'gpusim_fncs.zpl')
    ns3_zpl = os.path.join(input_dir, 'ns3_fncs.zpl')
    with open(gpusim_zpl, 'w') as f:
        f.write('name = gpusim\ntime_delta = 1ns\nbroker = tcp://localhost:5570\n')
        f.write('values\n    finish\n        topic = ns3/finish\n        default = ""\n')
        f.write('        type = string\n        list = false\n')
    with open(ns3_zpl, 'w') as f:
        f.write('name = ns3\ntime_delta = 1ns\nbroker = tcp://localhost:5570\n')
        f.write('values\n    cloudsim/transfer\n        topic = gpusim/cloudsim/transfer\n')
        f.write('        default = ""\n        type = string\n        list = false\n')
        f.write('    cloudsim/end\n        topic = gpusim/cloudsim/end\n')

    script = os.path.join(input_dir, 'run.sh')
    with open(script, 'w') as f:
        f.write(f'''#!/bin/bash
{FNCS_BROKER} 2 > /dev/null 2>&1 &
sleep 1
FNCS_CONFIG_FILE={ns3_zpl} FNCS_BROKER=tcp://localhost:5570 FNCS_NAME=ns3 FNCS_TIME_DELTA=1ns FNCS_FATAL=yes \\
  {NS3_BIN} --topo={topo_path} > /dev/null 2>&1 &
sleep 1
export PATH="/opt/homebrew/opt/openjdk/bin:$PATH"
FNCS_CONFIG_FILE={gpusim_zpl} FNCS_BROKER=tcp://localhost:5570 FNCS_NAME=gpusim FNCS_TIME_DELTA=1ns FNCS_FATAL=yes \\
  java --enable-native-access=ALL-UNNAMED \\
    -Djava.library.path={GPUSIM_DIR}/lib \\
    -cp "{GPUSIM_DIR}/out/production/gpuworkflowsim:{GPUSIM_DIR}/jars/*" \\
    backend.SimEngine {output_dir} {hosts_path} {jobs_path} {faults_path} -1 false false 0 \\
    > /dev/null 2>&1
pkill -f "fncs_broker|fncs-example" 2>/dev/null
''')
    os.chmod(script, 0o755)

    try:
        proc = subprocess.run(['bash', script], timeout=SIM_TIMEOUT,
                              capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        print(f'    TIMEOUT ({SIM_TIMEOUT}s)')
        kill_simulation()
        return None

    job_run_path = os.path.join(output_dir, 'jobRun.xml')
    if not os.path.exists(job_run_path):
        print(f'    ERROR: jobRun.xml not found')
        return None

    try:
        tree = ET.parse(job_run_path)
        root = tree.getroot()
        tasks = []
        for job in root.findall('Job'):
            rec = job.find('RunningRecord')
            if rec is not None:
                tasks.append({
                    'name': job.get('name'),
                    'start': float(rec.get('start', 0)),
                    'end': float(rec.get('end', 0)),
                    'duration': float(rec.get('duration', 0)),
                    'host': rec.get('host', ''),
                })

        if not tasks:
            return None

        total_time = max(t['end'] for t in tasks)
        per_gpu_compute = {}
        for t in tasks:
            per_gpu_compute[t['host']] = per_gpu_compute.get(t['host'], 0) + t['duration']
        avg_compute = sum(per_gpu_compute.values()) / max(len(per_gpu_compute), 1)
        comm_time = total_time - avg_compute

        return {
            'total_time': total_time,
            'compute_time': avg_compute,
            'comm_time': max(comm_time, 0),
            'num_tasks': len(tasks),
        }
    except Exception as e:
        print(f'    ERROR parsing results: {e}')
        return None


def csv_total_latency(csv_path):
    """读取 NeuSight CSV 的总 fw_latency(ms)"""
    with open(csv_path) as f:
        return sum(float(r['fw_latency']) for r in csv.DictReader(f) if float(r['fw_latency']) > 0)


def print_table(title, headers, rows):
    widths = [max(len(h), max((len(str(r[i])) for r in rows), default=0)) + 2 for i, h in enumerate(headers)]
    fmt = ''.join(f'{{:<{w}}}' if i == 0 else f'{{:>{w}}}' for i, w in enumerate(widths))
    print(f'\n--- {title} ---')
    print(fmt.format(*headers))
    print('-' * sum(widths))
    for r in rows:
        print(fmt.format(*r))


# ═══════════════════ 实验 1 ═══════════════════

def experiment_1_gpu_specs():
    print('\n' + '=' * 70)
    print('实验 1: 变更 GPU 规格 (NeuSight 预测)')
    print('  模型: GPT-3 27层   TP=2 PP=1   seq=2048 batch=2')
    print('=' * 70)

    exp_dir = RESULTS_DIR / 'exp1_gpu_spec'

    # 生成一个假设性"下一代 GPU"配置 (H100 ×1.5 算力)
    next_gen_config = exp_dir / 'NextGen_GPU.json'
    os.makedirs(exp_dir, exist_ok=True)
    h100_cfg = json.load(open(NEUSIGHT_DEVICE_CONFIGS / 'NVIDIA_H100_80GB_HBM3.json'))
    next_gen = {k: (int(v * 1.5) if isinstance(v, (int, float)) and k != 'Dev_Mem' else v) for k, v in h100_cfg.items()}
    next_gen['Device'] = 'NextGen_GPU'
    json.dump(next_gen, open(next_gen_config, 'w'), indent=2)

    # 为 NextGen 运行 NeuSight 预测
    print('  [NextGen] Running NeuSight prediction...')
    neusight_out = str(exp_dir / 'neusight_out')
    next_gen_csv = run_neusight_prediction(next_gen_config, neusight_out)

    gpu_list = list(NEUSIGHT_CSV.items())
    if next_gen_csv:
        gpu_list.append(('NextGen', Path(next_gen_csv)))
        print(f'    NeuSight prediction done: {csv_total_latency(next_gen_csv):.2f} ms')

    results = []
    for gpu_name, csv_path in gpu_list:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            continue
        latency_ms = csv_total_latency(str(csv_path))
        print(f'  [{gpu_name}] NeuSight fw_latency={latency_ms:.2f}ms, running cosim...')
        config_dir = str(exp_dir / gpu_name)
        convert(str(csv_path), tp=2, pp=1, num_layers=27, hidden_dim=2560,
                bandwidth_gbps=200, topo_type='star', output_dir=config_dir)
        result = run_simulation(config_dir)
        if result:
            result['neusight_ms'] = latency_ms
            results.append((gpu_name, result))
            print(f'    total={result["total_time"]:.0f}  compute={result["compute_time"]:.0f}  comm={result["comm_time"]:.0f}')
        else:
            results.append((gpu_name, None))

    if results and results[0][1]:
        baseline = results[0][1]['total_time']
        rows = []
        for name, r in results:
            if r:
                rows.append((name, f'{r["neusight_ms"]:.2f}', f'{r["total_time"]:.0f}',
                             f'{r["compute_time"]:.0f}', f'{r["comm_time"]:.0f}',
                             f'{baseline / r["total_time"]:.2f}x'))
        print_table('实验 1: GPU 规格变化',
                    ['GPU', 'NeuSight(ms)', '总时间', '计算时间', '通信时间', '加速比'], rows)
    return results


# ═══════════════════ 实验 2 ═══════════════════

def experiment_2_parallelism():
    print('\n' + '=' * 70)
    print('实验 2: 变更并行方式 (H100, 2 GPU)')
    print('=' * 70)

    csv_path = str(NEUSIGHT_CSV['H100'])
    exp_dir = RESULTS_DIR / 'exp2_parallelism'
    configs = [
        ('TP2_PP1', 2, 1),
        ('TP1_PP2', 1, 2),
    ]
    results = []
    for name, tp, pp in configs:
        print(f'  [{name}] Running...')
        config_dir = str(exp_dir / name)
        convert(csv_path, tp=tp, pp=pp, num_layers=27, hidden_dim=2560,
                bandwidth_gbps=200, topo_type='star', output_dir=config_dir)
        result = run_simulation(config_dir)
        if result:
            results.append((name, result))
            print(f'    total={result["total_time"]:.0f}  compute={result["compute_time"]:.0f}  comm={result["comm_time"]:.0f}  tasks={result["num_tasks"]}')
        else:
            results.append((name, None))

    rows = [(n, f'{r["total_time"]:.0f}', f'{r["compute_time"]:.0f}',
             f'{r["comm_time"]:.0f}', str(r['num_tasks'])) for n, r in results if r]
    print_table('实验 2: 并行方式变化',
                ['配置', '总时间', '计算时间', '通信时间', '任务数'], rows)
    return results


# ═══════════════════ 实验 3 ═══════════════════

def experiment_3_topology():
    print('\n' + '=' * 70)
    print('实验 3: 变更网络拓扑 (H100 TP=2 PP=1)')
    print('=' * 70)

    csv_path = str(NEUSIGHT_CSV['H100'])
    exp_dir = RESULTS_DIR / 'exp3_topology'
    topos = ['star', 'direct_p2p']
    results = []
    for topo in topos:
        print(f'  [{topo}] Running...')
        config_dir = str(exp_dir / topo)
        convert(csv_path, tp=2, pp=1, num_layers=27, hidden_dim=2560,
                bandwidth_gbps=200, topo_type=topo, output_dir=config_dir)
        result = run_simulation(config_dir)
        if result:
            results.append((topo, result))
            print(f'    total={result["total_time"]:.0f}  comm={result["comm_time"]:.0f}')
        else:
            results.append((topo, None))

    rows = []
    for n, r in results:
        if r:
            pct = r['comm_time'] / r['total_time'] * 100
            rows.append((n, f'{r["total_time"]:.0f}', f'{r["comm_time"]:.0f}', f'{pct:.1f}%'))
    print_table('实验 3: 拓扑变化', ['拓扑', '总时间', '通信时间', '通信占比'], rows)
    return results


# ═══════════════════ 实验 4 ═══════════════════

def experiment_4_bandwidth():
    print('\n' + '=' * 70)
    print('实验 4: 变更网络带宽 (H100 TP=2 PP=1)')
    print('=' * 70)

    csv_path = str(NEUSIGHT_CSV['H100'])
    exp_dir = RESULTS_DIR / 'exp4_bandwidth'
    bandwidths = [400, 200, 100, 50]
    results = []
    for bw in bandwidths:
        label = f'{bw}Gbps'
        print(f'  [{label}] Running...')
        config_dir = str(exp_dir / label)
        convert(csv_path, tp=2, pp=1, num_layers=27, hidden_dim=2560,
                bandwidth_gbps=bw, topo_type='star', output_dir=config_dir)
        result = run_simulation(config_dir)
        if result:
            results.append((label, result))
            print(f'    total={result["total_time"]:.0f}  comm={result["comm_time"]:.0f}')
        else:
            results.append((label, None))

    rows = []
    for n, r in results:
        if r:
            pct = r['comm_time'] / r['total_time'] * 100
            rows.append((n, f'{r["total_time"]:.0f}', f'{r["compute_time"]:.0f}',
                         f'{r["comm_time"]:.0f}', f'{pct:.1f}%'))
    print_table('实验 4: 带宽变化', ['带宽', '总时间', '计算时间', '通信时间', '通信占比'], rows)
    return results


# ═══════════════════ Main ═══════════════════

if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print('=' * 60)
    print('  协同仿真扩展能力 Case Study')
    print('  AI应用: GPT-3 27层 推理 (seq=2048, batch=2)')
    print('=' * 60)

    experiments = sys.argv[1:] if len(sys.argv) > 1 else ['1', '2', '3', '4']

    all_results = {}
    if '1' in experiments:
        all_results['gpu_spec'] = experiment_1_gpu_specs()
    if '2' in experiments:
        all_results['parallelism'] = experiment_2_parallelism()
    if '3' in experiments:
        all_results['topology'] = experiment_3_topology()
    if '4' in experiments:
        all_results['bandwidth'] = experiment_4_bandwidth()

    print('\n' + '=' * 60)
    print('  所有实验完成!')
    print('=' * 60)
