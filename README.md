# SP-SLAM3

**SuperPoint + ORB-SLAM3**: A visual SLAM system that replaces the handcrafted ORB feature extractor with the learned [SuperPoint](https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork) deep feature detector and descriptor.

In the SP-SLAM3 pipeline, input images are converted to grayscale and fed into the SuperPoint detector pipeline (A). The SuperPoint encoder-decoder pipeline consists of a learned encoder, utilizing several convolutional layers, and two non-learned decoders for joint feature and descriptor extraction. The detected features are then processed by the ORB-SLAM3 backend, which comprises three primary components operating in parallel threads: the Tracking, Local Mapping, and Loop & Map Merging threads (B). The backend extracts keyframes, initializes and updates the map, and performs both local and global motion and pose estimation within the Local Mapping Thread and Loop & Map Merging thread. If a loop closure is detected, the pose estimation is further refined.

![Pipeline Overview](imgs/img_1.jpg)
![Results](imgs/img_2.jpg)

[![Watch the video](https://img.youtube.com/vi/AWvs2rZ45cA/hqdefault.jpg)](https://youtu.be/AWvs2rZ45cA)

This repository was forked from [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3). The pre-trained SuperPoint model comes from the official [MagicLeap repository](https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork).

## Key Changes from ORB-SLAM3

| Component | ORB-SLAM3 | SP-SLAM3 |
|-----------|-----------|----------|
| Feature Extractor | ORB (handcrafted) | SuperPoint (learned, CNN-based) |
| Descriptor | 256-bit binary (Hamming distance) | 256-dim float (L2 distance) |
| Vocabulary | DBoW2 | DBoW3 with SuperPoint descriptors |
| Matcher | ORBmatcher (Hamming) | SPmatcher (L2 norm) |

---

# 1. Prerequisites

Tested on **Ubuntu 20.04**.

## C++17 Compiler

SP-SLAM3 uses C++17 thread and chrono functionalities.

## OpenCV

We use [OpenCV](http://opencv.org) to manipulate images and features. **Required at least 3.0. Tested with OpenCV 3.4.16**.

```bash
sudo apt-get update

sudo apt-get install build-essential cmake git pkg-config libgtk-3-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \
    gfortran openexr libatlas-base-dev python3-dev python3-numpy \
    libtbb2 libtbb-dev libdc1394-22-dev

cd ~
git clone https://github.com/opencv/opencv.git
cd opencv
git checkout 3.4.16

cd ~
git clone https://github.com/opencv/opencv_contrib.git
cd opencv_contrib
git checkout 3.4.16

cd ~/opencv
mkdir build && cd build

cmake -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
      -D BUILD_EXAMPLES=ON ..

make -j$(nproc)
sudo make install
sudo ldconfig
```

## Eigen3

Required by g2o. **Required at least 3.1.0. Tested with Eigen3 3.4.0**.

```bash
sudo apt install libeigen3-dev
```

## DBoW3, Pangolin and g2o (Included in Thirdparty folder)

We use a BoW vocabulary based on the [DBoW3](https://github.com/rmsalinas/DBow3) library to perform place recognition, and [g2o](https://github.com/RainerKuemmerle/g2o) library is used to perform non-linear optimizations. All these libraries are included in the *Thirdparty* folder.

### Vocabulary

The SuperPoint vocabulary file (`Vocabulary/superpoint_voc.yml.gz`) is included in this repository. For more information please refer to [this repo](https://github.com/Kasper-Borzdynski/Ms-Deep_SLAM.git).

## NVIDIA Driver & CUDA Toolkit 12.2 with cuDNN 8.9.1

Follow these [instructions](https://developer.nvidia.com/cuda-12.2-download-archive) for the installation of CUDA Toolkit 12.2.

If not installed during the CUDA Toolkit installation process, install the NVIDIA driver:

```bash
sudo apt-get install nvidia-driver-535
```

Export CUDA paths:

```bash
echo 'export PATH=/usr/local/cuda-12.2/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
sudo ldconfig
```

Verify NVIDIA driver availability:

```bash
nvidia-smi
```

## LibTorch 2.1.0 (with GPU | CUDA 12.1)

If only CPU is available, install the CPU version of LibTorch. The system will automatically fall back to CPU mode.

```bash
wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip -O libtorch.zip
sudo unzip libtorch.zip -d /usr/local
```

Set the `TORCH_DIR` environment variable (optional, defaults to `/usr/local/libtorch/share/cmake/Torch`):

```bash
export TORCH_DIR=/usr/local/libtorch/share/cmake/Torch
```

---

# 2. Building SP-SLAM3

Clone the repository:

```bash
git clone --recursive https://github.com/fthbng77/SP_SLAM3.git
```

Build the project:

```bash
cd SP_SLAM3
chmod +x build.sh
./build.sh
```

If LibTorch is installed in a custom path:

```bash
TORCH_DIR=/path/to/libtorch/share/cmake/Torch ./build.sh
```

---

# 3. Running (Monocular)

```bash
cd SP_SLAM3
export LD_LIBRARY_PATH=$(pwd)/lib:$LD_LIBRARY_PATH

# Run with SuperPoint vocabulary (recommended)
./Examples/Monocular/mono_webcam Vocabulary/superpoint_voc.yml Examples/Monocular/EuRoC.yaml

# Or run with ORB vocabulary
./Examples/Monocular/mono_webcam Vocabulary/ORBvoc.txt Examples/Monocular/EuRoC.yaml
```

### Configuration

Edit `Examples/Monocular/EuRoC.yaml` to adjust parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `ORBextractor.nFeatures` | Number of features per image | 800 |
| `ORBextractor.nLevels` | Scale pyramid levels (1 = no pyramid, recommended) | 1 |
| `ORBextractor.iniThFAST` | SuperPoint confidence threshold | 0.155 |
| `ORBextractor.minThFAST` | Fallback threshold (if too few features detected) | 0.055 |

> **Note:** Parameter names use `ORBextractor` prefix for backward compatibility with ORB-SLAM3.

---

# 4. Architecture

```
Input Image (Grayscale)
       │
       ▼
┌──────────────┐
│  SuperPoint   │  ── CNN encoder + dual decoder
│  (LibTorch)   │     → Keypoints + 256-dim float descriptors
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  SPextractor  │  ── OctTree distribution + NMS
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────┐
│           ORB-SLAM3 Backend              │
│                                          │
│  Tracking ─── Local Mapping ─── Loop     │
│  Thread       Thread            Closing  │
│                                 Thread   │
└──────────────────────────────────────────┘
```

---

# 5. Roadmap

### Phase 2 — Matching Improvement

- [ ] **LightGlue integration** — Replace brute-force L2 matching with [LightGlue](https://github.com/cvg/LightGlue), an attention-based learned matcher designed for SuperPoint descriptors. Priority targets:
  - `SearchForInitialization` — Monocular initialization (most critical, determines map quality)
  - `SearchForTriangulation` — New map point creation
  - `SearchByProjection` — Can be used as fallback for difficult scenes
- [ ] **Matching threshold calibration** — Current `TH_HIGH=0.70` and `TH_LOW=0.30` were adapted from ORB's Hamming thresholds. These need empirical optimization on benchmark datasets for L2 descriptor matching.

### Phase 3 — Place Recognition Improvement

- [ ] **NetVLAD / learned place recognition** — DBoW3 is designed for binary descriptors and is suboptimal for float descriptors. Replacing it with [NetVLAD](https://arxiv.org/abs/1511.07247) or [CosPlace](https://github.com/gmberton/CosPlace) for loop closing and relocalization would significantly improve robustness.
- [ ] **SuperPoint vocabulary regeneration** — Evaluate the current vocabulary quality and size. Retrain with a larger and more diverse dataset if needed.

### Phase 4 — Performance Optimization

- [ ] **TensorRT / ONNX Runtime** — Replace LibTorch inference with TensorRT for 2-3x speedup on NVIDIA GPUs, or ONNX Runtime for cross-platform acceleration.
- [ ] **Remove image pyramid** — SuperPoint is inherently scale-invariant. The current pyramid code (`nLevels=1` workaround) can be fully removed to eliminate overhead.
- [ ] **Half precision (FP16) inference** — Run SuperPoint in FP16 mode for faster inference with minimal accuracy loss.

---

# 6. License

SP-SLAM3 is released under the [GPLv3 license](https://www.gnu.org/licenses/gpl-3.0.html), same as ORB-SLAM3.

# 7. References

- [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) - C. Campos, R. Elvira, J. J. G. Rodriguez, J. M. M. Montiel and J. D. Tardos
- [SuperPoint](https://arxiv.org/abs/1712.07629) - D. DeTone, T. Malisiewicz and A. Rabinovich (MagicLeap)
