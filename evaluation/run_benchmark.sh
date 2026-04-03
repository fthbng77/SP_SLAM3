#!/bin/bash
# SP-SLAM3 EuRoC Benchmark — 5 runs per sequence, compute median ATE RMSE
# Usage: cd SP_SLAM3 && bash evaluation/run_benchmark.sh

set -e

SLAM_DIR="/home/fatih/SP_SLAM3"
VOCAB="$SLAM_DIR/Vocabulary/superpoint_voc.dbow3"
CONFIG="$SLAM_DIR/Examples/Monocular/EuRoC.yaml"
BINARY="$SLAM_DIR/Examples/Monocular/mono_euroc"
GT_DIR="$SLAM_DIR/evaluation/Ground_truth/EuRoC_left_cam"
TS_DIR="$SLAM_DIR/Examples/Monocular/EuRoC_TimeStamps"
SEQ_DIR="$SLAM_DIR/Examples/Monocular"
RESULT_DIR="${1:-$SLAM_DIR/evaluation/results}"

export LD_LIBRARY_PATH="$SLAM_DIR/lib:$LD_LIBRARY_PATH"

NUM_RUNS=5

SEQUENCES=("MH_01_easy" "MH_02_easy" "MH_03_medium" "MH_04_difficult" "MH_05_difficult")
TIMESTAMPS=("MH01" "MH02" "MH03" "MH04" "MH05")
GT_FILES=("MH01_GT.txt" "MH02_GT.txt" "MH03_GT.txt" "MH04_GT.txt" "MH05_GT.txt")

mkdir -p "$RESULT_DIR"

echo "========================================================"
echo "  SP-SLAM3 EuRoC Benchmark"
echo "  Sequences: ${#SEQUENCES[@]} | Runs per sequence: $NUM_RUNS"
echo "========================================================"
echo ""

for i in "${!SEQUENCES[@]}"; do
    SEQ="${SEQUENCES[$i]}"
    TS="${TIMESTAMPS[$i]}"
    GT="${GT_FILES[$i]}"

    SEQ_PATH="$SEQ_DIR/$SEQ"
    TS_PATH="$TS_DIR/$TS.txt"
    GT_PATH="$GT_DIR/$GT"

    if [ ! -d "$SEQ_PATH/mav0/cam0/data" ]; then
        echo "[$SEQ] SKIPPED — dataset not found"
        continue
    fi

    echo "[$SEQ] Starting $NUM_RUNS runs..."

    mkdir -p "$RESULT_DIR/$SEQ"

    for run in $(seq 1 $NUM_RUNS); do
        echo -n "  Run $run/$NUM_RUNS... "

        cd "$SLAM_DIR"

        # Run SLAM (suppress output, viewer disabled for headless benchmarking)
        SLAM_NO_VIEWER=1 timeout 300 "$BINARY" "$VOCAB" "$CONFIG" "$SEQ_PATH" "$TS_PATH" > /dev/null 2>&1 || true

        # Save trajectory
        if [ -f "$SLAM_DIR/CameraTrajectory.txt" ]; then
            cp "$SLAM_DIR/CameraTrajectory.txt" "$RESULT_DIR/$SEQ/run_${run}.txt"
            frames=$(wc -l < "$RESULT_DIR/$SEQ/run_${run}.txt")
            echo "done ($frames frames)"
        else
            echo "FAILED (no trajectory output)"
        fi
    done
    echo ""
done

echo "========================================================"
echo "  Computing ATE RMSE for all runs..."
echo "========================================================"

python3 - "$RESULT_DIR" "$GT_DIR" << 'PYTHON_SCRIPT'
import sys, os, glob
import numpy as np

result_dir = sys.argv[1]
gt_dir = sys.argv[2]

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
        return None, 0
    gt_xyz = np.array([[gt[a][0], gt[a][1], gt[a][2]] for a, b in matches]).T
    est_xyz = np.array([[est[b][0], est[b][1], est[b][2]] for a, b in matches]).T
    # Sim3 alignment
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
    rmse = np.sqrt(np.mean(error**2))
    return rmse, len(matches)

sequences = {
    "MH_01_easy": "MH01_GT.txt",
    "MH_02_easy": "MH02_GT.txt",
    "MH_03_medium": "MH03_GT.txt",
    "MH_04_difficult": "MH04_GT.txt",
    "MH_05_difficult": "MH05_GT.txt",
}

# ORB-SLAM3 paper values (median of 5 runs, monocular)
orb_paper = {
    "MH_01_easy": 0.016,
    "MH_02_easy": 0.027,
    "MH_03_medium": 0.028,
    "MH_04_difficult": 0.138,
    "MH_05_difficult": 0.072,
}

print()
print(f'{"Sequence":<18} {"Runs":>5} {"Median":>10} {"Mean":>10} {"Best":>10} {"Worst":>10} {"ORB-SLAM3":>10} {"vs ORB":>10}')
print("-" * 85)

all_results = {}

for seq, gt_name in sequences.items():
    seq_dir = os.path.join(result_dir, seq)
    gt_file = os.path.join(gt_dir, gt_name)

    if not os.path.isdir(seq_dir):
        print(f'{seq:<18} {"SKIPPED":>5}')
        continue

    run_files = sorted(glob.glob(os.path.join(seq_dir, "run_*.txt")))
    rmses = []
    for rf in run_files:
        rmse, pairs = compute_ate(gt_file, rf)
        if rmse is not None:
            rmses.append(rmse)

    if not rmses:
        print(f'{seq:<18} {"FAILED":>5}')
        continue

    median_rmse = np.median(rmses)
    mean_rmse = np.mean(rmses)
    best_rmse = np.min(rmses)
    worst_rmse = np.max(rmses)
    orb_val = orb_paper[seq]
    ratio = orb_val / median_rmse

    all_results[seq] = {
        "median": median_rmse, "mean": mean_rmse,
        "best": best_rmse, "worst": worst_rmse,
        "runs": len(rmses), "orb": orb_val, "ratio": ratio
    }

    print(f'{seq:<18} {len(rmses):>5} {median_rmse:>10.4f} {mean_rmse:>10.4f} {best_rmse:>10.4f} {worst_rmse:>10.4f} {orb_val:>10.3f} {ratio:>9.2f}x')

print("-" * 85)

if all_results:
    all_sp = [v["median"] for v in all_results.values()]
    all_orb = [v["orb"] for v in all_results.values()]
    print(f'{"AVERAGE":<18} {"":>5} {np.mean(all_sp):>10.4f} {"":>10} {"":>10} {"":>10} {np.mean(all_orb):>10.3f} {np.mean(all_orb)/np.mean(all_sp):>9.2f}x')

print()
print("=== README Table (copy-paste) ===")
print()
print("| Sequence | SP-SLAM3 | ORB-SLAM3 (paper) | Improvement |")
print("|----------|----------|-------------------|-------------|")
for seq, r in all_results.items():
    print(f'| {seq} | **{r["median"]:.4f} m** | {r["orb"]:.3f} m | {r["ratio"]:.2f}x |')

PYTHON_SCRIPT

echo ""
echo "Benchmark complete! Results saved in: $RESULT_DIR"
