./Examples/Monocular/mono_euroc ./Vocabulary/ORBvoc.txt ./Examples/Monocular/EuRoC.yaml /home/ray/Desktop/Dataset/EuROC/MH_03_medium ./Examples/Monocular/EuRoC_TimeStamps/MH03.txt dataset-MH03_mono

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

Opencv 4.4 needs tiff 4.0;

Python version should be checked. Version 3.7 is tested fine.

```
cmake -DGAUSSIANSPLATTING=OFF ..
```