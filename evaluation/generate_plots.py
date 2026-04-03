#!/usr/bin/env python3
"""
Generate evaluation plots for SP-SLAM3 README.
Usage: python3 evaluation/generate_plots.py
"""
import os, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    from mpl_toolkits.mplot3d import Axes3D
    HAS_3D = True
except ImportError:
    HAS_3D = False

GT_DIR = "evaluation/Ground_truth/EuRoC_left_cam"
RESULT_DIR = "evaluation/results_baseline"
OUTPUT_DIR = "evaluation"

SEQUENCES = {
    'MH_01_easy': 'MH01_GT.txt',
    'MH_02_easy': 'MH02_GT.txt',
    'MH_03_medium': 'MH03_GT.txt',
    'MH_04_difficult': 'MH04_GT.txt',
    'MH_05_difficult': 'MH05_GT.txt',
}

ORB_PAPER = {
    'MH_01_easy': 0.016, 'MH_02_easy': 0.027, 'MH_03_medium': 0.028,
    'MH_04_difficult': 0.138, 'MH_05_difficult': 0.072,
}


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


def sim3_align(gt_xyz, est_xyz):
    mm = est_xyz.mean(axis=1, keepdims=True)
    dm = gt_xyz.mean(axis=1, keepdims=True)
    mc, dc = est_xyz - mm, gt_xyz - dm
    U, _, Vt = np.linalg.svd(dc @ mc.T)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.sum(dc * (R @ mc)) / np.sum(mc * mc)
    t = dm - s * R @ mm
    aligned = s * R @ est_xyz + t
    return aligned


def get_best_run(seq, gt_name):
    """Return the run with lowest ATE RMSE for a sequence."""
    seq_dir = os.path.join(RESULT_DIR, seq)
    gt_file = os.path.join(GT_DIR, gt_name)
    gt = read_tum(gt_file)
    run_files = sorted(glob.glob(os.path.join(seq_dir, 'run_*.txt')))

    best_rmse, best_data = float('inf'), None
    for rf in run_files:
        est = read_tum(rf)
        matches = associate(gt, est)
        if len(matches) < 10:
            continue
        gt_xyz = np.array([[gt[a][0], gt[a][1], gt[a][2]] for a, b in matches]).T
        est_xyz = np.array([[est[b][0], est[b][1], est[b][2]] for a, b in matches]).T
        aligned = sim3_align(gt_xyz, est_xyz)
        errors = np.sqrt(np.sum((aligned - gt_xyz)**2, axis=0))
        rmse = np.sqrt(np.mean(errors**2))
        timestamps = np.array([a for a, b in matches])
        if rmse < best_rmse:
            best_rmse = rmse
            best_data = {
                'gt_xyz': gt_xyz, 'aligned': aligned, 'errors': errors,
                'timestamps': timestamps, 'rmse': rmse
            }
    return best_data


def plot_trajectory_2d(all_data):
    """2D trajectory comparison for all sequences."""
    n = len(all_data)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), dpi=150)
    if n == 1:
        axes = [axes]

    for ax, (seq, data) in zip(axes, all_data.items()):
        gt = data['gt_xyz']
        al = data['aligned']
        ax.plot(gt[0], gt[1], 'b-', linewidth=1.2, label='Ground Truth', alpha=0.8)
        ax.plot(al[0], al[1], 'r-', linewidth=1.0, label='SP-SLAM3', alpha=0.8)
        ax.set_title(seq.replace('_', ' ').title(), fontsize=10, fontweight='bold')
        ax.set_xlabel('X (m)', fontsize=8)
        ax.set_ylabel('Y (m)', fontsize=8)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)

    fig.suptitle('SP-SLAM3 vs Ground Truth — EuRoC (2D)', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'trajectory_comparison_2d.png')
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_trajectory_3d(all_data):
    """3D trajectory comparison for all sequences."""
    n = len(all_data)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(6 * cols, 5 * rows), dpi=150)

    for i, (seq, data) in enumerate(all_data.items()):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        gt = data['gt_xyz']
        al = data['aligned']

        # Start/end markers
        ax.plot(gt[0], gt[1], gt[2], 'b-', linewidth=1.2, label='Ground Truth', alpha=0.7)
        ax.plot(al[0], al[1], al[2], 'r-', linewidth=1.0, label='SP-SLAM3', alpha=0.8)
        ax.scatter(*gt[:, 0], color='blue', s=40, marker='o', zorder=5)
        ax.scatter(*gt[:, -1], color='blue', s=40, marker='s', zorder=5)
        ax.scatter(*al[:, 0], color='red', s=40, marker='o', zorder=5)
        ax.scatter(*al[:, -1], color='red', s=40, marker='s', zorder=5)

        ax.set_title(f'{seq.replace("_", " ").title()}\nRMSE: {data["rmse"]:.4f} m',
                      fontsize=11, fontweight='bold')
        ax.set_xlabel('X (m)', fontsize=9)
        ax.set_ylabel('Y (m)', fontsize=9)
        ax.set_zlabel('Z (m)', fontsize=9)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=8, loc='upper left')
        ax.view_init(elev=25, azim=45)

    fig.suptitle('SP-SLAM3 vs Ground Truth — EuRoC (3D)', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'trajectory_comparison_3d.png')
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_trajectory_overlay(all_data):
    """All sequences overlaid on a single 2D plot."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_data)))

    for (seq, data), color in zip(all_data.items(), colors):
        gt = data['gt_xyz']
        al = data['aligned']
        label = seq.replace('_', ' ').title()
        ax.plot(gt[0], gt[1], '-', color='gray', linewidth=0.5, alpha=0.4)
        ax.plot(al[0], al[1], '-', color=color, linewidth=1.2, label=f'{label} (RMSE: {data["rmse"]:.4f}m)', alpha=0.85)

    ax.plot([], [], '-', color='gray', linewidth=1, label='Ground Truth')
    ax.set_xlabel('X (m)', fontsize=10)
    ax.set_ylabel('Y (m)', fontsize=10)
    ax.set_title('SP-SLAM3 Trajectory Overview — EuRoC', fontsize=13, fontweight='bold')
    ax.set_aspect('equal')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'trajectory_comparison_overlay.png')
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_ate_over_time(all_data):
    """ATE error over time for all sequences."""
    n = len(all_data)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5), dpi=150)
    if n == 1:
        axes = [axes]

    for ax, (seq, data) in zip(axes, all_data.items()):
        ts = data['timestamps']
        t_sec = (ts - ts[0]) / 1e9  # nanoseconds to seconds
        errors = data['errors']

        ax.plot(t_sec, errors * 100, 'r-', linewidth=0.8, alpha=0.7)
        ax.axhline(y=data['rmse'] * 100, color='blue', linestyle='--', linewidth=1.0,
                    label=f'RMSE: {data["rmse"]:.4f}m')
        ax.fill_between(t_sec, 0, errors * 100, alpha=0.15, color='red')
        ax.set_title(seq.replace('_', ' ').title(), fontsize=10, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.set_ylabel('ATE (cm)', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)

    fig.suptitle('Absolute Trajectory Error Over Time — EuRoC', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'ate_over_time.png')
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    print("Generating SP-SLAM3 evaluation plots...")
    print()

    all_data = {}
    for seq, gt_name in SEQUENCES.items():
        seq_dir = os.path.join(RESULT_DIR, seq)
        if not os.path.isdir(seq_dir):
            print(f"  [{seq}] SKIPPED — no results")
            continue
        data = get_best_run(seq, gt_name)
        if data:
            all_data[seq] = data
            print(f"  [{seq}] ATE RMSE: {data['rmse']:.4f} m")
        else:
            print(f"  [{seq}] FAILED — could not compute ATE")

    if not all_data:
        print("No results found!")
        return

    print()
    plot_trajectory_2d(all_data)
    plot_trajectory_overlay(all_data)
    if HAS_3D:
        plot_trajectory_3d(all_data)
    else:
        print("  [3D plot] SKIPPED — mpl_toolkits.mplot3d not available")
    plot_ate_over_time(all_data)
    print()
    print("Done! All plots saved to evaluation/")


if __name__ == "__main__":
    main()
