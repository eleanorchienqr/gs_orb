./Examples/Monocular/mono_euroc ./Vocabulary/ORBvoc.txt ./Examples/Monocular/EuRoC.yaml /home/ray/Desktop/Dataset/EuROC/MH_03_medium ./Examples/Monocular/EuRoC_TimeStamps/MH03.txt dataset-MH03_mono

./Examples/Monocular/mono_tum ./Vocabulary/ORBvoc.txt ./Examples/Monocular/TUM3.yaml /home/ray/Desktop/Dataset/TUM/rgbd_dataset_freiburg3_long_office_household

evo_traj euroc /home/ray/Desktop/Dataset/EuROC/MH_03_medium/mav0/state_groundtruth_estimate0/data.csv --save_as_tum 

evo_traj tum f_dataset-MH03_mono.txt --ref=./Groundtruth/MH03_GT.tum --align --plot -vas
evo_ape tum f_dataset-MH03_mono.txt ./Groundtruth/MH03_GT.tum --align --plot -vas

```
wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcu121.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.4.0+cu121.zip -d Thirdparty/
```

# Gaussian Renderer GUI

Folder `Shaders` includes a vertex shader `gau_vert.glsl` and a fragment shader `gau_frag.glsl`.

`OpenGL 4.6.0` with `GLFW 3.4` and `imgui` for visualization under a new thread `GaussianMapping`, referred by `instant-ngp` and `Orbeez-SLAM`.

# LibTorch

# Several Tips

Opencv 4.8, referring to [Install OpenCV 4.10 with CUDA 12.6 and CUDNN 8.9 in Ubuntu 24.04](https://gist.github.com/raulqf/f42c718a658cddc16f9df07ecc627be7).
```
cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D WITH_TBB=ON \
-D ENABLE_FAST_MATH=1 \
-D CUDA_FAST_MATH=1 \
-D WITH_CUBLAS=1 \
-D WITH_CUDA=OFF \
-D BUILD_opencv_cudacodec=OFF \
-D WITH_CUDNN=ON \
-D OPENCV_DNN_CUDA=OFF \
-D CUDA_ARCH_BIN=7.5 \
-D WITH_V4L=ON \
-D WITH_QT=OFF \
-D WITH_OPENGL=ON \
-D WITH_GSTREAMER=ON \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D OPENCV_PC_FILE_NAME=opencv.pc \
-D OPENCV_ENABLE_NONFREE=ON \
-D OPENCV_EXTRA_MODULES_PATH=/home/ray/Desktop/ORB_SLAM3/Thirdparty/opencv_contrib-4.8.0/modules \
-D INSTALL_PYTHON_EXAMPLES=OFF \
-D INSTALL_C_EXAMPLES=OFF \
-D BUILD_EXAMPLES=OFF ..
```

cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=OFF -D OPENCV_GENERATE_PKGCONFIG=ON -D OPENCV_EXTRA_MODULES_PATH=~/home/ray/Desktop/ORB_SLAM3/Thirdparty/opencv_contrib-4.4.0/opencv_contrib-4.4.0/modules -D BUILD_EXAMPLES=ON ..

Python version should be checked. Version 3.7 is tested fine.

```
cmake -DGAUSSIANSPLATTING=ON ..
```

