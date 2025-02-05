/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef CONFIG_H
#define CONFIG_H

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>


namespace ORB_SLAM3
{

class ViewerConfig
{

};

class CameraConfig
{

};

class ORBExtractorConfig
{

};

class IMUConfig
{

};

class ConfigParser
{
public:
    bool ParseConfigFile(std::string &strConfigFile);

private:

    ViewerConfig mViewerConfig;
    CameraConfig mCameraConfig;
    ORBExtractorConfig mORBConfig;
    IMUConfig mIMUConfig;

};

class GaussianOptimConfig
{

private:
    size_t iterations = 3000; // 30'000 // MonoGS
    // learning rate
    float mPosLrInit = 0.0016f; // MonoGS
    float mPosLrFinal = 0.0000016f; // MonoGS
    float mPosLrDelayMult = 0.01f;// MonoGS
    int64_t mPosLrMaxSteps = 30000; // 30'000 // MonoGS
    float mFeatureLr = 0.0025f; // MonoGS
    float mOpacityLr = 0.05f;// MonoGS
    float mScalingLr = 0.001f;// MonoGS
    float mRotationLr = 0.001f;// MonoGS
    // loss
    float mLambdaDssim = 0.2f;
    // float convergence_threshold = 0.007f;
    // densify, prune and reset opacity
    float mPercentDense = 0.01f;  // MonoGS
    float mMinOpacity = 0.005f; // MonoGS
    uint64_t mDensificationInterval = 10; //10 // MonoGS
    uint64_t mOpacityResetInterval = 300; // 3'000 // MonoGS
    uint64_t mDensifyFromIter = 50; // 500 // MonoGS
    uint64_t mDensifyUntilIter = 1500; // 15'000
    float mDensifyGradThreshold = 0.0002f; // MonoGS
    // other
    bool mEmptyGpuCache = false;
    bool mEarlyrStopping = false;
};

struct OptimizationParameters 
{
    size_t iterations = 150; // 30'000 // MonoGS
    // learning rate
    float position_lr_init = 0.0016f; // MonoGS
    float position_lr_final = 0.0000016f; // MonoGS
    float position_lr_delay_mult = 0.01f;// MonoGS
    int64_t position_lr_max_steps = 30000; // 30'000 // MonoGS
    float feature_lr = 0.0025f; // MonoGS
    float opacity_lr = 0.05f;// MonoGS
    float scaling_lr = 0.001f;// MonoGS
    float rotation_lr = 0.001f;// MonoGS
    // loss
    float lambda_dssim = 0.2f;
    float convergence_threshold = 0.007f;
    // densify, prune and reset opacity
    float percent_dense = 0.01f;  // MonoGS
    float min_opacity = 0.005f; // MonoGS
    uint64_t densification_interval = 150; //10 // MonoGS
    uint64_t opacity_reset_interval = 300; // 3'000 // MonoGS
    uint64_t densify_from_iter = 0; // 500 // MonoGS
    uint64_t densify_until_iter = 150; // 15'000
    float densify_grad_threshold = 0.0002f; // MonoGS
    // other
    bool empty_gpu_cache = false;
    bool early_stopping = false;
};

struct MonoGSOptimizationParameters 
{
    size_t iterations = 1050;  // 1050
    // preprocess param
    int downsampling_factor = 32;
    // learning rate
    float position_lr_init = 0.0016f; 
    float position_lr_final = 0.0000016f; 
    float position_lr_delay_mult = 0.01f;
    int64_t position_lr_max_steps = 30000; 
    float feature_lr = 0.0025f; 
    float opacity_lr = 0.05f;
    float scaling_lr = 0.001f;
    float rotation_lr = 0.001f;
    // loss
    float rgb_boundary_threshold = 0.01f;
    float lambda_dssim = 0.2f;
    // densify, prune and reset opacity
    float percent_dense = 0.01f;  
    float min_opacity = 0.005f;
    uint64_t densification_interval = 100; 
    uint64_t opacity_reset_interval = 500; 
    uint64_t densify_from_iter = 0; 
    uint64_t densify_until_iter = 1050; 
    float densify_grad_threshold = 0.0002f; 
    // other
    bool empty_gpu_cache = false;
};

}

#endif // CONFIG_H
