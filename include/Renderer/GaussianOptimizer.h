#include <torch/torch.h>
#include "Config.h"
#include "MapGaussian.h"
#include "KeyFrame.h"

namespace GaussianSplatting{

class GaussianOptimizer
{
public:
    // constructer
    GaussianOptimizer(const ORB_SLAM3::OptimizationParameters &OptimParams);

    void InitializeOptimization(const std::vector<ORB_SLAM3::KeyFrame *> &vpKFs, const std::vector<ORB_SLAM3::MapGaussian *> &vpMG);
    // Optimize(int nIteration);

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
    std::vector<torch::Tensor> mViewmatrices;
    std::vector<torch::Tensor> mProjmatrices;
    std::vector<torch::Tensor> mVameraCenters;

    int mImHeight;
    int mImWidth;
    float mTanFovx;
    float mTanFovy;
    
    torch::Tensor mBackground = torch::tensor({1.f, 1.f, 1.f});
    float mScaleModifier = 1.f;
    int mSHDegree = 10;
    bool mPrefiltered = false;

    // Training associated members
};

}