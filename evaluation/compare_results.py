#!/usr/bin/env python3
"""
Compare SP-SLAM3 benchmark results: baseline vs optical flow.
Usage: python3 evaluation/compare_results.py
"""
import os, glob
import numpy as np

GT_DIR = "evaluation/Ground_truth/EuRoC_left_cam"

def read_tum(filepath):
    data = {}
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split(',') if ',' in line else line.split()
            ts = float(parts[0])
            data[ts] = [float(v) for v in parts[1:]]
    return data

def associate(gt, est, max_diff=20000000):
    gt_stamps = sorted(gt.keys())
    matches = []
    for es in sorted(est.keys()):
        best, best_diff = None, max_diff
        for gs in gt_stamps:
            d = abs(gs - es)
            if d < best_diff:
                best, best_diff = gs, d
        if best is not None:
            matches.append((best, es))
    return matches

def compute_ate(gt_file, est_file):
    gt = read_tum(gt_file)
    est = read_tum(est_file)
    matches = associate(gt, est)
    if len(matches) < 10:
        return None, 0, len(est)
    gt_xyz = np.array([[gt[a][0], gt[a][1], gt[a][2]] for a, b in matches]).T
    est_xyz = np.array([[est[b][0], est[b][1], est[b][2]] for a, b in matches]).T
    mm = est_xyz.mean(axis=1, keepdims=True)
    dm = gt_xyz.mean(axis=1, keepdims=True)
    mc, dc = est_xyz - mm, gt_xyz - dm
    U, _, Vt = np.linalg.svd(dc @ mc.T)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0: S[2, 2] = -1
    R = U @ S @ Vt
    s = np.sum(dc * (R @ mc)) / np.sum(mc * mc)
    t = dm - s * R @ mm
    aligned = s * R @ est_xyz + t
    error = np.sqrt(np.sum((aligned - gt_xyz)**2, axis=0))
    return np.sqrt(np.mean(error**2)), len(matches), len(est)

def eval_dir(result_dir):
    sequences = {
        'MH_01_easy': 'MH01_GT.txt',
        'MH_02_easy': 'MH02_GT.txt',
        'MH_03_medium': 'MH03_GT.txt',
        'MH_04_difficult': 'MH04_GT.txt',
        'MH_05_difficult': 'MH05_GT.txt',
    }
    total_images = {
        'MH_01_easy': 3682, 'MH_02_easy': 3040, 'MH_03_medium': 2700,
        'MH_04_difficult': 2033, 'MH_05_difficult': 2273,
    }
    results = {}
    for seq, gt_name in sequences.items():
        seq_dir = os.path.join(result_dir, seq)
        gt_file = os.path.join(GT_DIR, gt_name)
        run_files = sorted(glob.glob(os.path.join(seq_dir, 'run_*.txt')))
        rmses, track_pcts = [], []
        for rf in run_files:
            rmse, pairs, total_est = compute_ate(gt_file, rf)
            if rmse is not None:
                rmses.append(rmse)
                track_pcts.append(100.0 * total_est / total_images[seq])
        if rmses:
            results[seq] = {
                'median': np.median(rmses), 'best': np.min(rmses),
                'worst': np.max(rmses), 'track_pct': np.mean(track_pcts),
                'runs': len(rmses)
            }
    return results

orb_paper = {
    'MH_01_easy': 0.034, 'MH_02_easy': 0.036, 'MH_03_medium': 0.035,
    'MH_04_difficult': 0.048, 'MH_05_difficult': 0.033,
}

baseline_dir = "evaluation/results_baseline"
optflow_dir = "evaluation/results_optflow"

# Fallback: if results_optflow doesn't exist, check results/
if not os.path.isdir(optflow_dir):
    optflow_dir = "evaluation/results"

has_baseline = os.path.isdir(baseline_dir)
has_optflow = os.path.isdir(optflow_dir)

if has_baseline:
    baseline = eval_dir(baseline_dir)
if has_optflow:
    optflow = eval_dir(optflow_dir)

print()
print("=" * 110)
print("  SP-SLAM3 Benchmark Comparison — EuRoC (Monocular, Sim3, median of 5 runs)")
print("=" * 110)
print()

if has_baseline and has_optflow:
    print(f'{"Sequence":<18} {"Baseline":>10} {"Track%":>8} {"OptFlow":>10} {"Track%":>8} {"Change":>10} {"ORB-SLAM3":>10} {"vs ORB":>8}')
    print("-" * 110)
    for seq in orb_paper:
        b = baseline.get(seq)
        o = optflow.get(seq)
        orb = orb_paper[seq]
        if b and o:
            change = (1 - o['median'] / b['median']) * 100
            best_sp = min(b['median'], o['median'])
            ratio = orb / o['median']
            sign = "+" if change > 0 else ""
            print(f'{seq:<18} {b["median"]:>10.4f} {b["track_pct"]:>7.1f}% {o["median"]:>10.4f} {o["track_pct"]:>7.1f}% {sign}{change:>8.1f}% {orb:>10.3f} {ratio:>7.2f}x')
        elif b:
            print(f'{seq:<18} {b["median"]:>10.4f} {b["track_pct"]:>7.1f}% {"N/A":>10} {"":>8} {"":>10} {orb:>10.3f}')
    print("-" * 110)

    b_avg = np.mean([v['median'] for v in baseline.values()])
    o_avg = np.mean([v['median'] for v in optflow.values()])
    orb_avg = np.mean(list(orb_paper.values()))
    change = (1 - o_avg / b_avg) * 100
    sign = "+" if change > 0 else ""
    print(f'{"AVERAGE":<18} {b_avg:>10.4f} {"":>8} {o_avg:>10.4f} {"":>8} {sign}{change:>8.1f}% {orb_avg:>10.3f} {orb_avg/o_avg:>7.2f}x')

elif has_baseline:
    print(f'{"Sequence":<18} {"SP-SLAM3":>10} {"Track%":>8} {"ORB-SLAM3":>10} {"vs ORB":>8}')
    print("-" * 70)
    for seq in orb_paper:
        b = baseline.get(seq)
        orb = orb_paper[seq]
        if b:
            ratio = orb / b['median']
            print(f'{seq:<18} {b["median"]:>10.4f} {b["track_pct"]:>7.1f}% {orb:>10.3f} {ratio:>7.2f}x')
    print("-" * 70)

elif has_optflow:
    print(f'{"Sequence":<18} {"SP-SLAM3":>10} {"Track%":>8} {"ORB-SLAM3":>10} {"vs ORB":>8}')
    print("-" * 70)
    for seq in orb_paper:
        o = optflow.get(seq)
        orb = orb_paper[seq]
        if o:
            ratio = orb / o['median']
            print(f'{seq:<18} {o["median"]:>10.4f} {o["track_pct"]:>7.1f}% {orb:>10.3f} {ratio:>7.2f}x')
    print("-" * 70)

print()
print("=" * 110)
print("  README Table:")
print("=" * 110)
print()

if has_optflow:
    src = optflow
    label = "SP-SLAM3"
elif has_baseline:
    src = baseline
    label = "SP-SLAM3"

print(f"| Sequence | {label} (median) | ORB-SLAM3 (paper) | Track Rate | Result |")
print("|----------|-------------------|-------------------|------------|--------|")
for seq in orb_paper:
    r = src.get(seq)
    orb = orb_paper[seq]
    if r:
        ratio = orb / r['median']
        if ratio >= 1:
            print(f'| {seq} | **{r["median"]:.4f} m** | {orb:.3f} m | {r["track_pct"]:.1f}% | {ratio:.2f}x better |')
        else:
            print(f'| {seq} | {r["median"]:.4f} m | **{orb:.3f} m** | {r["track_pct"]:.1f}% | {ratio:.2f}x |')
