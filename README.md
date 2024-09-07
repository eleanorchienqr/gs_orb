./Examples/Monocular/mono_euroc ./Vocabulary/ORBvoc.txt ./Examples/Monocular/EuRoC.yaml /home/ray/Desktop/Dataset/EuROC/MH_03_medium ./Examples/Monocular/EuRoC_TimeStamps/MH03.txt dataset-MH03_mono

evo_traj euroc /home/ray/Desktop/Dataset/EuROC/MH_03_medium/mav0/state_groundtruth_estimate0/data.csv --save_as_tum 

evo_traj tum f_dataset-MH03_mono.txt --ref=./Groundtruth/MH03_GT.tum --align --plot -vas
evo_ape tum f_dataset-MH03_mono.txt ./Groundtruth/MH03_GT.tum --align --plot -vas

```
wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcu121.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.4.0+cu121.zip -d Thirdparty/
```
