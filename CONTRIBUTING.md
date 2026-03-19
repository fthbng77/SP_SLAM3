# Contributing to SP-SLAM3

Thank you for your interest in contributing to SP-SLAM3! This guide will help you get started.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone --recursive https://github.com/<your-username>/SP_SLAM3.git
   ```
3. Follow the [README](README.md) to install prerequisites and build the project

## Building

```bash
chmod +x build.sh
./build.sh
```

Make sure the build completes without errors before submitting changes.

## Testing

SP-SLAM3 uses the [EuRoC MAV dataset](https://www.research-collection.ethz.ch/entities/researchdata/bcaf173e-5dac-484b-bc37-faf97a594f1f) for benchmarking. After making changes, verify that performance has not regressed:

1. **Run a sequence:**
   ```bash
   export LD_LIBRARY_PATH=$(pwd)/lib:$LD_LIBRARY_PATH
   ./Examples/Monocular/mono_euroc \
     Vocabulary/superpoint_voc.dbow3 \
     Examples/Monocular/EuRoC.yaml \
     Examples/Monocular/MH_01_easy \
     Examples/Monocular/EuRoC_TimeStamps/MH01.txt
   ```

2. **Evaluate trajectory accuracy with [evo](https://github.com/MichaelGrupp/evo):**
   ```bash
   pip install evo
   evo_ape tum evaluation/Ground_truth/EuRoC_left_cam/MH01_GT.txt CameraTrajectory.txt -as
   ```

3. **Check against baseline:** SP-SLAM3 achieves **0.0131 m** ATE RMSE on MH_01_easy. Your changes should not significantly regress this result.

If your change involves model export scripts, also verify the export:
```bash
pip install git+https://github.com/cvg/LightGlue.git
python scripts/export_lightglue.py --output lightglue.pt
```

## Submitting Changes

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes in focused, logical commits
3. Write clear commit messages describing **why**, not just what
4. Push your branch and open a Pull Request against `main`

### PR Guidelines

- Keep PRs focused on a single feature or fix
- Include benchmark results if your change affects tracking, matching, or optimization
- Update the README if you add new features or change configuration options

## Code Style

- C++ code follows the existing ORB-SLAM3 style (no clang-format enforced)
- Python scripts use standard PEP 8 conventions
- Use meaningful variable names consistent with the existing codebase

## Project Structure

| Directory | Description |
|-----------|-------------|
| `src/` | Core SLAM source files (SPextractor, SPmatcher, LightGlue, etc.) |
| `include/` | Header files |
| `Examples/` | Example binaries and configuration files |
| `scripts/` | Model export scripts (LightGlue, CosPlace, NetVLAD) |
| `evaluation/` | Ground truth data and evaluation scripts |
| `Thirdparty/` | DBoW3, g2o, Sophus (do not modify) |
| `Vocabulary/` | SuperPoint BoW vocabulary files |

## Reporting Issues

When reporting a bug, please include:
- OS and GPU info
- CUDA / LibTorch / OpenCV versions
- Steps to reproduce
- Relevant log output

## License

By contributing, you agree that your contributions will be licensed under the [GPLv3 license](https://www.gnu.org/licenses/gpl-3.0.html).
