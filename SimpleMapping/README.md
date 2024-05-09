# SimpleMapping

# 1. Installation
We list the required dependencies for SimpleMapping in 1.1 and further dependencies in 1.2. Also consider the full [README](https://github.com/UZ-SLAMLab/ORB_SLAM3?tab=readme-ov-file#2-prerequisites) of ORB-SLAM3. The code is tested on Ubuntu 22.04.

## 1.1 Dependencies

For the paper we used CUDA 11.1, cuDNN 8.2.1, and libtorch-1.9.0+cu111 with CUDA support and CXX11 ABI. However, we assume that the code should work with a broad range of versions because it doesn't use version-specific features. We can sadly not offer a convenient installation script due to (a) different CUDA installation options and (b) the cuDNN download method that needs user input. You have to:

+ Install CUDA from [nvidia.com](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html). Make sure that this doesn't interfere with other packages on your system and maybe ask your system administrator. 

+ Install cuDNN from [nvidia.com](https://docs.nvidia.com/deeplearning/cudnn/latest/installation/overview.html). Make sure to install a version that exactly matches your CUDA and PyTorch versions.

+ Install **LibTorch** from [pytorch.org](https://pytorch.org/get-started/locally/). For our exact version you can use
```
wget https://download.pytorch.org/libtorch/cu111/libtorch-cxx11-abi-shared-with-deps-1.9.0%2Bcu111.zip
```

To ensure `cmake` can find the installed libraries, please double check the path stated in `CMakeLists.txt`, for our version:
```
set(CMAKE_PREFIX_PATH ~/libtorch)
set(CUDNN_LIBRARY /storage/software/cuda/cuda-11.1/cudnn/v8.2.1.32/cuda/lib64)
set(CUDNN_LIBRARY_PATH /storage/software/cuda/cuda-11.1/cudnn/v8.2.1.32/cuda/lib64)
set(CUDNN_INCLUDE_PATH /storage/software/cuda/cuda-11.1/cudnn/v8.2.1.32/cuda/include) 
```

## 1.2 Further Dependencies
### suitesparse and eigen3 (required).
We use suitesparse version `1:5.1.2-2` and eigen version `3.3.9` corresponding to `worl.major.minor`.
```
sudo apt install libsuitesparse-dev libeigen3-dev libboost-all-dev
```
### OpenCV (required).
We use version `4.2.0`.
```
sudo apt install libopencv-dev
```
### Pangolin (required).
Used for 3D visualization & the GUI. Install from https://github.com/stevenlovegrove/Pangolin. We use version `0.6`.

### DBoW2 and g2o (Included in Thirdparty folder)
We use modified versions of the DBoW2 library to perform place recognition and g2o library to perform non-linear optimizations. Both modified libraries are included in the Thirdparty folder.

# 2. Build

+ Clone the repository:
```
git clone https://github.com/yingyexin/SimpleMapping.git
```

+ We provide a script `build.sh` to build the *Thirdparty* libraries. Execute:
```
cd SimpleMapping
chmod +x build.sh
./build.sh
```

+ To build *SimpleMapping*, please make sure you have installed required dependencies, and specify your path to the cudnn library in the `CMakeLists.txt`:
```
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
```
This will create **libSimpleMapping.so**  at `lib` folder and the executables in `Examples` folder.

# 3. EuRoC Example
EuRoC dataset was recorded with two pinhole cameras and an inertial sensor. 

1. Download a sequence (ASL format) from the official [website](http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)

2. Modify the calibration file `Examples/Monocular-Inertial/EuRoC.yaml`, please pay attention to the following new configuration:

* `Camera.newWidth` and `Camera.newHeight`: should align with the trained MVSNet.
* `LoopClosing: 0`: disable the loop closure to avoid messy fussion.
* `dr_timing: 0`: whether to enable the runtime efficiency test.

3. Download our [pretrained model](https://drive.google.com/file/d/1X58ShAUyMHheOWki3C0VL4TQ3lFveI-M/view?usp=drivesdk), extract the `model.pt` and store it under folder `exported_models/`

4. Execute the following command to process e.g. V101 sequences with Monucular-Inertial sensor configuration, please double check the dataset path and model path :
```
bash test.sh
# or explicitly execute
./Examples/Monocular-Inertial/mono_inertial_euroc ./Vocabulary/ORBvoc.txt \
./Examples/Monocular-Inertial/EuRoC.yaml PATH/TO/EUROC/V1_01_easy \
./Examples/Monocular-Inertial/EuRoC_TimeStamps/V101.txt V101 ./exported_models/ True
```
Specially, the last argument is map viewer flag, and `V101` is folder name and the running results will be stored in `./result/debug/V101` as the following:
```
results/debug/V101
	|__ image/							# running images of the referecne frame (input)
	|__ sparse/							# corresponding sparse depth (input)
	|__ depth/							# predicted dense depth (output)
	|__ CameraWorldTrajectory.txt 		# camera pose Twb in the world coordinate
	|__ CameraTrajectory.txt			# camera pose Twb0, b0 is first keyframe
	|__ KeyFrameTrajectory.txt			# keyframe camera poses in the world coordinate
	|__ mesh.obj						# resulting 3D mesh (output)
```
