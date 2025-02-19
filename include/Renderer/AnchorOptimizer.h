#include <torch/torch.h>

#include "Config.h"
#include "KeyFrame.h"

namespace GaussianSplatting{

class AnchorOptimizer
{

public:
    // Constructer
    // AnchorOptimizer(const ORB_SLAM3::MonoGSOptimizationParameters &OptimParams, const std::vector<ORB_SLAM3::KeyFrame *> &vpKFs, const std::vector<ORB_SLAM3::MapPoint *> &vpMP);

    // Optimization body
    // void InitializeOptimization()
    // void TrainingSetup();
    // void Optimize();

    // Densification and prune
    // void AddDensificationStats(torch::Tensor& viewspace_point_tensor, torch::Tensor& update_filter);
    // void DensifyAndPrune(float max_grad, float min_opacity, float max_screen_size);
    // void DensifyAndClone(torch::Tensor& grads, float grad_threshold);
    // void DensifyAndSplit(torch::Tensor& grads, float grad_threshold, float min_opacity, float max_screen_size);
    // void PrunePoints(torch::Tensor mask);

    // void DensificationPostfix(torch::Tensor& newMeans3D, torch::Tensor& newFeaturesDC, torch::Tensor& newFeaturesRest,
    //                           torch::Tensor& newScales, torch::Tensor& newRotation, torch::Tensor& newOpacity);
    // void CatTensorstoOptimizer(torch::Tensor& extension_tensor, torch::Tensor& old_tensor, int param_position);
    // void PruneOptimizer(torch::Tensor& old_tensor, const torch::Tensor& mask, int param_position);
    // void ResetOpacity();
    
    // Learning rate updater
    // void UpdateLR(float iteration);

    // SH degree updater
    // void UpdateSHDegree();

    // Utils
    inline torch::Tensor InverseSigmoid(torch::Tensor x) {return torch::log(x / (1 - x));}

    // Tree management
    // void UpdateIndiceForestAfterClone(const torch::Tensor indices);
    // void UpdateIndiceForestAfterSplit(const torch::Tensor indices);
    // void UpdateIndiceForestAfterPrune(const torch::Tensor indices);

protected:
    // 1. Setting
    ORB_SLAM3::OptimizationParameters mOptimParams;

    int mSizeofAnchors;
    int mSizeofOffsets = 5;
    int mFeatureDim = 32;
    int mAppearanceDim = 32;

    float mVoxelSize = 0.01;

    // 2. Learnable members
    torch::Tensor mAchorPos;        // [mSizeofAnchors, 3]
    torch::Tensor mAchorFeatures;   // [mSizeofAnchors, 32]
    torch::Tensor mAchorScales;     // [mSizeofAnchors, 1]
    torch::Tensor mAchorRotations;     // [mSizeofAnchors, 1]
    torch::Tensor mOffsets;         // [mSizeofAnchors, mSizeofOffsets, 3]

    struct mFeatureMLP : torch::nn::Module { };             // [input_dim, output_dim] = [3+1, mFeatureDim]
    struct mOpacityMLP : torch::nn::Module { };             // [input_dim, output_dim] = [mFeatureDim+3+1, mSizeofOffsets]
    struct mCovarianceMLP : torch::nn::Module { };          // [input_dim, output_dim] = [mFeatureDim+3+1, 7*mSizeofOffsets]
    struct mColorMLP : torch::nn::Module { };               // [input_dim, output_dim] = [mFeatureDim+3+1mAppearanceDim, 3*mSizeofOffsets]
    struct mAppearanceEmbedding : torch::nn::Module { };    // [input_dim, output_dim] = [mSizeofCameras, mAppearanceDim]

    // 3. Anchor mangement members
    torch::Tensor mOpacityAccum;            // [mSizeofAnchors, 1]
    torch::Tensor mOffsetGradientAccum;     // [mSizeofAnchors*mSizeofOffsets, 1]
    torch::Tensor mOffsetDenom;             // [mSizeofAnchors*mSizeofOffsets, 1]

    // 4. Cameras associated members
    int mSizeofCameras;
    std::vector<cv::Mat> mTrainedImages;
    std::vector<torch::Tensor> mTrainedImagesTensor;
    std::vector<torch::Tensor> mViewMatrices;
    std::vector<torch::Tensor> mProjMatrices;
    std::vector<torch::Tensor> mCameraCenters;

    float mNerfNormRadius;
    torch::Tensor mNerfNormTranslation;

    // 5. Render params
    bool mWhiteBackground = true;
    torch::Tensor mBackground = torch::tensor({1.f, 1.f, 1.f}).to(torch::kCUDA);
    
    float mScaleModifier = 1.f;
    bool mPrefiltered = false;

    float mNear = 0.01f;
    float mFar = 100.0f;

    // 6. Traning and loss members


    // 7. MapPoint label management
    std::vector<long> mvpAnchorRootIndex;  
};

}