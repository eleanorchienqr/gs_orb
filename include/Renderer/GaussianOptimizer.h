#include <torch/torch.h>
#include "Config.h"
#include "MapGaussian.h"
#include "KeyFrame.h"

namespace GaussianSplatting{

struct Expon_lr_func {
    float lr_init;
    float lr_final;
    float lr_delay_steps;
    float lr_delay_mult;
    int64_t max_steps;
    Expon_lr_func(float lr_init = 0.f, float lr_final = 1.f, float lr_delay_mult = 1.f, int64_t max_steps = 1000000, float lr_delay_steps = 0.f)
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

class GaussianOptimizer
{
public:
    // constructer
    GaussianOptimizer(const ORB_SLAM3::OptimizationParameters &OptimParams);

    void InitializeOptimization(const std::vector<ORB_SLAM3::KeyFrame *> &vpKFs, const std::vector<ORB_SLAM3::MapGaussian *> &vpMG);
    void TrainingSetup();
    // Optimize(int nIteration);

    torch::Tensor GetViewMatrix(Sophus::SE3f &Tcw);
    void SetProjMatrix();

protected:
    ORB_SLAM3::OptimizationParameters mOptimParams;

    // Gaussian associated members
    int mSizeofGaussians;
    torch::Tensor mMeans3D;
    torch::Tensor mOpacity;
    torch::Tensor mScales;
    torch::Tensor mRotation;
    torch::Tensor mMeans2D;
    torch::Tensor mFeatures;
    torch::Tensor mCov3DPrecomp = torch::Tensor();
    torch::Tensor mColorsPrecomp = torch::Tensor();

    // Cameras/KeyFrames associated members
    int mSizeofCameras;
    std::vector<cv::Mat> mTrainedImages;
    std::vector<torch::Tensor> mViewMatrices;
    std::vector<torch::Tensor> mProjMatrices;
    std::vector<torch::Tensor> mCameraCenters;

    int mImHeight;
    int mImWidth;
    float mTanFovx;
    float mTanFovy;
    torch::Tensor mProjMatrix;
    
    torch::Tensor mBackground = torch::tensor({1.f, 1.f, 1.f});
    float mScaleModifier = 1.f;
    int mSHDegree = 10;
    bool mPrefiltered = false;
    float mNear = 0.01f;
    float mFar = 100.0f;

    // Training associated members
    float mSpatialLRScale = 6.0;
    float mPercentDense;
    torch::Tensor mPosGradientAccum;
    torch::Tensor mDenom;
    Expon_lr_func mPosSchedulerArgs;
    torch::optim::Adam* mOptimizer;
};

}