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

#include <torch/torch.h>


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
    uint64_t densification_interval = 100; // 100
    uint64_t opacity_reset_interval = 500; // 500
    uint64_t densify_from_iter = 0; 
    uint64_t densify_until_iter = 1050; 
    float densify_grad_threshold = 0.0002f; 
    // other
    bool empty_gpu_cache = false;
};

struct ScaffoldOptimizationParams
{
    int Iter = 1050;  // 30000
    float PercentDense = 0.01;

    // [Learning rate part] Fixed learning rate
    float FeatureLR = 0.0075;
    float ScalingLR = 0.007;
    float RotationLR = 0.002;

    // [Learning rate part] Scheduled learning rate
    float AnchorLRInit = 0.0;
    float AnchorLRFinal  = 0.0;
    float AnchorLRDelayMult = 0.01;
    int AnchorLRMaxSteps = 30'000;

    float OffsetLRInit = 0.01;
    float OffsetLRFinal  = 0.0001;
    float OffsetLRDelayMult = 0.01;
    int OffsetLRMaxSteps = 30'000;

    float FeatureBankMLPLRInit = 0.01;
    float FeatureBankMLPLRFinal  = 0.00001;
    float FeatureBankMLPLRDelayMult = 0.01;
    int FeatureBankMLPLRMaxSteps = 30'000;

    float OpacityMLPLRInit = 0.002;
    float OpacityMLPLRFinal  = 0.00002;
    float OpacityMLPLRDelayMult = 0.01;
    int OpacityMLPLRMaxSteps = 30'000;

    float CovarianceMLPLRInit = 0.004;
    float CovarianceMLPLRFinal  = 0.004;
    float CovarianceMLPLRDelayMult = 0.01;
    int CovarianceMLPLRMaxSteps = 30'000;

    float ColorMLPLRInit = 0.008;
    float ColorMLPLRFinal  = 0.0005;
    float ColorMLPLRDelayMult = 0.01;
    int ColorMLPLRMaxSteps = 30'000;

    // [Learning rate part] Spatial scale
    float SpatialLRScale = 6.0;

    // Anchor densification
    int StartStatistic = 0;     // 500
    int UpdateFrom = 0;         // 1500
    int UpdateInterval = 100;     // 100
    int UpdateUntil  = 15'000;

    float MinOpacity = 0.005;
    float SuccessTh = 0.8;
    float DensifyGradTh = 0.0002;
};

struct RenderOptimizationParams
{
    int Iter = 105;  // 30000
    float PercentDense = 0.01;

    // [Learning rate part] Fixed learning rate
    float FeatureLR = 0.0075;
    float ScalingLR = 0.007;
    float RotationLR = 0.002;

    // [Learning rate part] Scheduled learning rate
    float AnchorLRInit = 0.0;
    float AnchorLRFinal  = 0.0;
    float AnchorLRDelayMult = 0.01;
    int AnchorLRMaxSteps = 30'000;

    float OffsetLRInit = 0.01;
    float OffsetLRFinal  = 0.0001;
    float OffsetLRDelayMult = 0.01;
    int OffsetLRMaxSteps = 30'000;

    float FeatureBankMLPLRInit = 0.01;
    float FeatureBankMLPLRFinal  = 0.00001;
    float FeatureBankMLPLRDelayMult = 0.01;
    int FeatureBankMLPLRMaxSteps = 30'000;

    float OpacityMLPLRInit = 0.002;
    float OpacityMLPLRFinal  = 0.00002;
    float OpacityMLPLRDelayMult = 0.01;
    int OpacityMLPLRMaxSteps = 30'000;

    float CovarianceMLPLRInit = 0.004;
    float CovarianceMLPLRFinal  = 0.004;
    float CovarianceMLPLRDelayMult = 0.01;
    int CovarianceMLPLRMaxSteps = 30'000;

    float ColorMLPLRInit = 0.008;
    float ColorMLPLRFinal  = 0.0005;
    float ColorMLPLRDelayMult = 0.01;
    int ColorMLPLRMaxSteps = 30'000;

    // [Learning rate part] Spatial scale
    float SpatialLRScale = 6.0;

    // Anchor densification
    int StartStatistic = 0;     // 500
    int UpdateFrom = 0;         // 1500
    int UpdateInterval = 10;     // 100
    int UpdateUntil  = 15'000;

    float MinOpacity = 0.005;
    float SuccessTh = 0.8;
    float DensifyGradTh = 0.0002;
};

struct RenderModelParams
{
    int SampleNum = 8;

    // Network params
    int PosEncodeDim = 5;

    int FeatureEncoderLayerNum = 3;
    int FeatureEncoderHiddenDim = 32;
    int FeatureDim = 32;

    int FeatureDecoderLayerNum = 3;
    int FeatureDecoderHiddenDim = 32;
};

// Render Net Structure
struct PosEncoding : torch::nn::Module{

};

struct FeatureEncoder : torch::nn::Module{
    
};

struct FeatureDecoder : torch::nn::Module{
    
};

// Scaffold MLP structures
struct FeatureBankMLP : torch::nn::Module {
    FeatureBankMLP(int64_t InputDim, int64_t OutputDim, int64_t FeatureDim)
    {
        // register_module() is needed if we want to use the parameters() method later on
        linear1 = register_module("linear1", torch::nn::Linear(InputDim, FeatureDim));
        linear2 = register_module("linear2", torch::nn::Linear(FeatureDim, OutputDim));
    }

    torch::Tensor forward(torch::Tensor x) 
    {
        x = torch::relu(linear1->forward(x));
        x = torch::softmax(linear2->forward(x), 1);
        return x;
        // return linear1->forward(x);

    }
    // Use one of many "standard library" modules.
    torch::nn::Linear linear1{nullptr}, linear2{nullptr};
};

struct OpacityMLP : torch::nn::Module {
    OpacityMLP(int FeatureDim, int OffsetNum)
    : linear1(torch::nn::Linear(FeatureDim + 3, FeatureDim)),
        linear2(torch::nn::Linear(FeatureDim, OffsetNum))
    {
        // register_module() is needed if we want to use the parameters() method later on
        register_module("linear1", linear1);
        register_module("linear2", linear2);
    }

    torch::Tensor forward(torch::Tensor x) 
    {
        x = torch::relu(linear1(x));
        x = torch::tanh(linear2(x));
        return x;
    }

    torch::nn::Linear linear1, linear2;
};

struct CovarianceMLP : torch::nn::Module {
    CovarianceMLP(int FeatureDim, int OffsetNum)
    : linear1(torch::nn::Linear(FeatureDim + 3, FeatureDim)),
        linear2(torch::nn::Linear(FeatureDim, 7 * OffsetNum))
    {
        // register_module() is needed if we want to use the parameters() method later on
        register_module("linear1", linear1);
        register_module("linear2", linear2);
    }

    torch::Tensor forward(torch::Tensor x) 
    {
        x = torch::relu(linear1(x));
        x = linear2(x);
        return x;
    }

    torch::nn::Linear linear1, linear2;
};

struct ColorMLP : torch::nn::Module {
    ColorMLP(int FeatureDim, int OffsetNum)
    : linear1(torch::nn::Linear(FeatureDim + 3, FeatureDim)),
        linear2(torch::nn::Linear(FeatureDim, 3 * OffsetNum))
    {
        // register_module() is needed if we want to use the parameters() method later on
        register_module("linear1", linear1);
        register_module("linear2", linear2);
    }

    torch::Tensor forward(torch::Tensor x) 
    {
        x = torch::relu(linear1(x));
        x = torch::sigmoid(linear2(x));
        return x;
    }

    torch::nn::Linear linear1, linear2;
};

struct FreqColorMLP : torch::nn::Module {
    FreqColorMLP(int FeatureDim, int OffsetNum)
    : linear1(torch::nn::Linear(FeatureDim + 3, FeatureDim)),
        linear2(torch::nn::Linear(FeatureDim, 3 * OffsetNum))
    {
        // register_module() is needed if we want to use the parameters() method later on
        register_module("linear1", linear1);
        register_module("linear2", linear2);
    }

    torch::Tensor forward(torch::Tensor x) 
    {
        x = torch::relu(linear1(x));
        x = torch::sigmoid(torch::cos(linear2(x)));
        return x;
    }

    torch::nn::Linear linear1, linear2;
};

// Learning rate update func
struct ExponLRFunc {
    float lr_init;
    float lr_final;
    float lr_delay_steps;
    float lr_delay_mult;
    int64_t max_steps;
    ExponLRFunc(float lr_init = 0.f, float lr_final = 1.f, float lr_delay_mult = 1.f, int64_t max_steps = 1000000, float lr_delay_steps = 0.f)
        : lr_init(lr_init),
          lr_final(lr_final),
          lr_delay_mult(lr_delay_mult),
          max_steps(max_steps),
          lr_delay_steps(lr_delay_steps) {}

    float operator()(int64_t step) const {
        if (step < 0 || (lr_init == 0.0 && lr_final == 0.0)) {
            return 0.0;
        }
        float delay_rate;
        if (lr_delay_steps > 0. && step != 0) {
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * std::sin(0.5 * M_PI * std::clamp((float)step / lr_delay_steps, 0.f, 1.f));
        } else {
            delay_rate = 1.0;
        }
        float t = std::clamp(static_cast<float>(step) / static_cast<float>(max_steps), 0.f, 1.f);
        float log_lerp = std::exp(std::log(lr_init) * (1.f - t) + std::log(lr_final) * t);
        return delay_rate * log_lerp;
    }
};


}

#endif // CONFIG_H
