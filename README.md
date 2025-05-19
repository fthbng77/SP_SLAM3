# SUPERSLAM3

superpoint ile orbslam birlikte kullanılmıştır.



In the SUPERSLAM3 pipeline, input images are converted to grayscale and fed into the Superpoint detector pipeline (A). The Superpoint encoder-decoder pipeline consists of a learned encoder, utilizing several convolutional layers, and two non-learned decoders for joint feature and descriptor extraction. The detected features are then processed by the ORB-SLAM3 backend, which comprises three primary components operating in parallel threads: the Tracking, Local Mapping, and Loop & Map Merging threads (B). The backend extracts keyframes, initializes and updates the map, and performs both local and global motion and pose estimation within the Local Mapping Thread and Loop & Map Merging thread. If a loop closure is detected, the pose estimation is further refined.

This repository was forked from [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3). The pre-trained model of SuperPoint come from the official [MagicLeap 
repository](https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork).

# 1. Prerequisites
We have tested the libraries and executables on **Ubuntu 20.04**.

## C++17 or C++0x Compiler
ORBSLAM3 uses the new thread and chrono functionalities of C++17.

## OpenCV
We use [OpenCV](http://opencv.org) to manipulate images and features. Dowload and install instructions can be found at: http://opencv.org. **Required at least 3.0. Tested with OpenCV 3.4.11**.
```
sudo apt-get update

sudo apt-get install build-essential cmake git pkg-config libgtk-3-dev     libavcodec-dev libavformat-dev libswscale-dev libv4l-dev     libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev     gfortran openexr libatlas-base-dev python3-dev python3-numpy libtbb2 libtbb-dev libdc1394-22-dev

cd ~

git clone https://github.com/opencv/opencv.git
cd opencv
git checkout 3.4.16

cd ~
git clone https://github.com/opencv/opencv_contrib.git
cd opencv_contrib
git checkout 3.4.16

cd opencv
mkdir build && cd build

cmake -D CMAKE_BUILD_TYPE=Release       -D CMAKE_INSTALL_PREFIX=/usr/local       -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules       -D BUILD_EXAMPLES=ON ..

make -j$(nproc)
sudo make install
sudo ldconfig
```
## Eigen3
Required by g2o (see below). Download and install instructions can be found at: http://eigen.tuxfamily.org. **Required at least 3.1.0. Tested with Eigen3 3.4.0**.

## DBoW3, DBoW2, Pangolin and g2o (Included in Thirdparty folder)
We use a BOW vocabulary based on the [BOW3](https://github.com/rmsalinas/DBow3) library to perform place recognition, and [g2o](https://github.com/RainerKuemmerle/g2o) library is used to perform non-linear optimizations. All these libraries are included in the *Thirdparty* folder.

### Download Vocabulary

For more informations please refer to [this repo](https://github.com/Kasper-Borzdynski/Ms-Deep_SLAM.git).

## Nvidia-driver & Cuda Toolkit 12.2 with cuDNN 8.9.1
Please, follow these [instructions](https://developer.nvidia.com/cuda-12.2-download-archive) for the installation of the Cuda Toolkit 12.2.

If not installed during the Cuda Toolkit installation process, please install the nvidia driver 535:
``` shell
sudo apt-get install nvidia-driver-535
```

Export Cuda paths 
``` shell
echo 'export PATH=/usr/local/cuda-12.2/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
sudo ldconfig  
```

Verify the Nvidia driver availability:
``` shell
nvidia-smi
```


## LibTorch 1.6.0 version (with GPU | Cuda Toolkit 12.2, cuDNN 8.9.1)
If only CPU can be used, install cpu-version LibTorch. Some code change about tensor device should be required.

```shell
wget -O LibTorch.zip wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip -O libtorch.zip
sudo unzip libtorch.zip -d /usr/local
```

# 2. Building SUPERSLAM3 library and examples
Clone the repository:
```shell
git clone --recursive https://github.com/fthbng77/SP_SLAM3.git
```


Build the project:

```shell
cd SP_SLAM3
chmod +x build.sh
./build.sh
```

# 3. Monocular
Bu şekilde superpoint kullanılarak çalıştırılabilir yaml dosyasındaki threshold
```shell
cd SP_SLAM3
export LD_LIBRARY_PATH=/home/fatih/SP_SLAM3/lib:$LD_LIBRARY_PATH
./Examples/Monocular/mono_webcam Vocabulary/ORBvoc.txt Examples/Monocular/EuRoC.yaml
```
