#!/bin/bash
# SP-SLAM3 Build Script for Ubuntu 20.04
# Usage: bash build.sh [--no-cuda] [--libtorch-path /path/to/libtorch]
#
# Prerequisites for GPU support:
#   NVIDIA driver + CUDA toolkit must be installed before running this script.
#   Recommended: CUDA 11.8 or 12.x
#   See: https://developer.nvidia.com/cuda-downloads

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NPROC=$(nproc)
USE_CUDA=ON
LIBTORCH_DIR=""
LIBTORCH_DEFAULT_PATH="/usr/local/libtorch"

# ─── Argument parsing ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --no-cuda)
      USE_CUDA=OFF
      shift ;;
    --libtorch-path=*)
      LIBTORCH_DIR="${1#*=}"
      shift ;;
    --libtorch-path)
      LIBTORCH_DIR="$2"
      shift 2 ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: bash build.sh [--no-cuda] [--libtorch-path /path/to/libtorch]"
      exit 1 ;;
  esac
done

echo "============================================================"
echo "  SP-SLAM3 Build Script — Ubuntu 20.04"
echo "  CUDA: $USE_CUDA"
echo "============================================================"

# ─── CUDA pre-check ──────────────────────────────────────────────────────────
if [ "$USE_CUDA" = "ON" ]; then
  if ! command -v nvcc &>/dev/null; then
    echo ""
    echo "WARNING: --no-cuda not set but 'nvcc' not found."
    echo "  If you want GPU support, install the CUDA toolkit first:"
    echo "  https://developer.nvidia.com/cuda-downloads"
    echo "  Continuing with CPU-only build. Pass --no-cuda to silence this warning."
    USE_CUDA=OFF
  fi
fi

# ─── 1. System dependencies ──────────────────────────────────────────────────
echo ""
echo "[1/6] Installing system dependencies..."

sudo apt-get update -qq
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    wget \
    unzip \
    libssl-dev \
    libeigen3-dev \
    libboost-all-dev \
    libopencv-dev \
    libgtk-3-dev \
    libgl1-mesa-dev \
    libglew-dev \
    libpython3-dev \
    python3-numpy \
    libwayland-dev \
    libxkbcommon-dev \
    wayland-protocols \
    libegl1-mesa-dev \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev

# ─── 2. Pangolin ─────────────────────────────────────────────────────────────
echo ""
echo "[2/6] Checking Pangolin..."

if ! pkg-config --exists pangolin 2>/dev/null && \
   [ ! -f /usr/local/lib/cmake/Pangolin/PangolinConfig.cmake ] && \
   [ ! -f /usr/local/share/Pangolin/PangolinConfig.cmake ]; then
  echo "  Pangolin not found — building from source..."
  PANGOLIN_SRC="/tmp/Pangolin"
  if [ ! -d "$PANGOLIN_SRC" ]; then
    git clone --depth=1 --branch v0.6 https://github.com/stevenlovegrove/Pangolin.git "$PANGOLIN_SRC"
  fi
  mkdir -p "$PANGOLIN_SRC/build"
  cd "$PANGOLIN_SRC/build"
  cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF
  make -j"$NPROC"
  sudo make install
  sudo ldconfig
  cd "$SCRIPT_DIR"
  echo "  Pangolin installed."
else
  echo "  Pangolin already installed, skipping."
fi

# ─── 3. LibTorch ─────────────────────────────────────────────────────────────
echo ""
echo "[3/6] Checking LibTorch..."

# Prefer user-supplied path, then default install path
if [ -n "$LIBTORCH_DIR" ]; then
  TORCH_CMAKE_PATH="$LIBTORCH_DIR/share/cmake/Torch"
elif [ -d "$LIBTORCH_DEFAULT_PATH/share/cmake/Torch" ]; then
  TORCH_CMAKE_PATH="$LIBTORCH_DEFAULT_PATH/share/cmake/Torch"
  echo "  LibTorch found at $LIBTORCH_DEFAULT_PATH, skipping download."
else
  echo "  LibTorch not found — downloading (~2 GB)..."

  if [ "$USE_CUDA" = "ON" ]; then
    CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
    CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
    CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)
    echo "  Detected CUDA $CUDA_VER"

    if [ "$CUDA_MAJOR" -ge 12 ]; then
      TORCH_URL="https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.2.2%2Bcu121.zip"
    elif [ "$CUDA_MAJOR" -eq 11 ] && [ "$CUDA_MINOR" -ge 8 ]; then
      TORCH_URL="https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.2.2%2Bcu118.zip"
    else
      TORCH_URL="https://download.pytorch.org/libtorch/cu117/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcu117.zip"
    fi
  else
    echo "  Using CPU-only LibTorch."
    TORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.2.2%2Bcpu.zip"
  fi

  wget --show-progress -O /tmp/libtorch.zip "$TORCH_URL"
  sudo unzip -q /tmp/libtorch.zip -d /tmp/libtorch_extract/
  sudo mv /tmp/libtorch_extract/libtorch "$LIBTORCH_DEFAULT_PATH"
  sudo rm -rf /tmp/libtorch_extract /tmp/libtorch.zip

  # Register libtorch with the dynamic linker
  echo "$LIBTORCH_DEFAULT_PATH/lib" | sudo tee /etc/ld.so.conf.d/libtorch.conf > /dev/null
  sudo ldconfig

  TORCH_CMAKE_PATH="$LIBTORCH_DEFAULT_PATH/share/cmake/Torch"
  echo "  LibTorch installed at $LIBTORCH_DEFAULT_PATH"
fi

# ─── 4. Sophus ───────────────────────────────────────────────────────────────
echo ""
echo "[4/6] Building Sophus..."

SOPHUS_BUILD="$SCRIPT_DIR/Thirdparty/Sophus/build"
if [ ! -f /usr/local/lib/cmake/Sophus/SophusConfig.cmake ] && \
   [ ! -f /usr/local/share/sophus/cmake/SophusConfig.cmake ]; then
  mkdir -p "$SOPHUS_BUILD"
  cd "$SOPHUS_BUILD"
  cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SOPHUS_TESTS=OFF -DBUILD_SOPHUS_EXAMPLES=OFF
  make -j"$NPROC"
  sudo make install
  cd "$SCRIPT_DIR"
  echo "  Sophus installed."
else
  echo "  Sophus already installed, skipping."
fi

# ─── 5. Thirdparty libraries (DBoW3, g2o) ───────────────────────────────────
echo ""
echo "[5/6] Building DBoW3 and g2o..."

# DBoW3
DBOW3_BUILD="$SCRIPT_DIR/Thirdparty/DBoW3/build"
mkdir -p "$DBOW3_BUILD"
cd "$DBOW3_BUILD"
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j3   # intentionally limited to avoid linker OOM on small machines
cd "$SCRIPT_DIR"

# g2o
G2O_BUILD="$SCRIPT_DIR/Thirdparty/g2o/build"
mkdir -p "$G2O_BUILD"
cd "$G2O_BUILD"
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j"$NPROC"
cd "$SCRIPT_DIR"

# ─── 6. SP-SLAM3 ─────────────────────────────────────────────────────────────
echo ""
echo "[6/6] Building SP-SLAM3..."

SLAM_BUILD="$SCRIPT_DIR/build"
mkdir -p "$SLAM_BUILD"
cd "$SLAM_BUILD"
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_CUDA="$USE_CUDA" \
  -DTorch_DIR="$TORCH_CMAKE_PATH"
make -j"$NPROC"
cd "$SCRIPT_DIR"

echo ""
echo "============================================================"
echo "  Build completed successfully!"
echo ""
echo "  Binaries:"
echo "    Examples/Monocular/mono_euroc"
echo "    Examples/Monocular/mono_webcam"
echo "    Examples/Monocular/mono_live"
echo ""
echo "  Usage example:"
echo "    ./Examples/Monocular/mono_euroc \\"
echo "      Vocabulary/ORBvoc.txt \\"
echo "      Examples/Monocular/EuRoC.yaml \\"
echo "      /path/to/MH_01"
echo "============================================================"
