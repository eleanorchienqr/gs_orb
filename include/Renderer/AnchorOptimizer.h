#include <torch/torch.h>

#include "Config.h"
#include "Converter.h"
#include "KeyFrame.h"

namespace GaussianSplatting{

class AnchorOptimizer
{

public:
    // Constructer
    AnchorOptimizer(int SizeofInitAnchors, const int AnchorFeatureDim, const int AnchorSizeofOffsets, const int CamNum, 
                    torch::Tensor AnchorWorldPos, torch::Tensor AnchorFeatures, torch::Tensor AnchorScales, 
                    torch::Tensor AnchorRotations, torch::Tensor AnchorOffsets,
                    ORB_SLAM3::FeatureBankMLP FBNet, ORB_SLAM3::OpacityMLP OpacityNet, ORB_SLAM3::CovarianceMLP CovNet, ORB_SLAM3::ColorMLP ColorNet,
                    const int ImHeight, const int ImWidth, const float TanFovx, const float TanFovy,
                    std::vector<torch::Tensor> ViewMatrices, std::vector<cv::Mat> TrainedImages);
    void TrainingSetup();
    void Optimize();

protected:
    // Setters
    void SetProjMatrix();

    // Filters
    void PrefilterVoxel(const torch::Tensor ViewMatrix, const torch::Tensor ProjMatrix, const torch::Tensor CamCenter, torch::Tensor& VisibleVoxelIndices);
    void GenerateNeuralGaussian(const torch::Tensor CamCenter, const torch::Tensor VisibleVoxelIndices, 
                                torch::Tensor& GauPos, torch::Tensor& GauColor, torch::Tensor& GauOpacity, 
                                torch::Tensor& GauScale, torch::Tensor& GauRot,
                                torch::Tensor& NeuralOpacity, torch::Tensor& NeuralGauIndices);

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
    void UpdateLR(const float iteration);

    // Loss
    torch::Tensor L1Loss(const torch::Tensor& network_output, const torch::Tensor& gt);

    // Utils
    inline torch::Tensor InverseSigmoid(torch::Tensor x) {return torch::log(x / (1 - x));}

    // Tree management
    // void UpdateIndiceForestAfterClone(const torch::Tensor indices);
    // void UpdateIndiceForestAfterSplit(const torch::Tensor indices);
    // void UpdateIndiceForestAfterPrune(const torch::Tensor indices);

    // Debug
    static void PrintCUDAUse();

protected:
    // 1. Setting
    ORB_SLAM3::ScaffoldOptimizationParams mOptimizationParams;

    struct mModelParams
    {

    };

    int mSizeofAnchors;
    int mSizeofOffsets = 5;
    int mFeatureDim = 32;
    int mAppearanceDim = 32;

    float mVoxelSize = 0.01;

    // 2. Learnable members
    torch::Tensor mAchorPos;            // [mSizeofAnchors, 3]
    torch::Tensor mAchorFeatures;       // [mSizeofAnchors, 32]
    torch::Tensor mAchorScales;         // [mSizeofAnchors, 3]
    torch::Tensor mOffsets;             // [mSizeofAnchors, mSizeofOffsets, 3]
    torch::Tensor mAchorRotations;      // [mSizeofAnchors, 4] in func PrefilterVoxel

    ORB_SLAM3::FeatureBankMLP mFeatureMLP;                  // [input_dim, output_dim] = [3+1, mFeatureDim]
    ORB_SLAM3::OpacityMLP mOpacityMLP;                      // [input_dim, output_dim] = [mFeatureDim+3+1, mSizeofOffsets]
    ORB_SLAM3::CovarianceMLP mCovarianceMLP;                // [input_dim, output_dim] = [mFeatureDim+3+1, 7*mSizeofOffsets]
    ORB_SLAM3::ColorMLP mColorMLP;                          // [input_dim, output_dim] = [mFeatureDim+3+1mAppearanceDim, 3*mSizeofOffsets]
    struct mAppearanceEmbedding : torch::nn::Module { };    // [input_dim, output_dim] = [mSizeofCameras, mAppearanceDim]

    // 3. Anchor mangement members
    torch::Tensor mOpacityAccum;            // [mSizeofAnchors, 1]
    torch::Tensor mOffsetGradientAccum;     // [mSizeofAnchors*mSizeofOffsets, 1]
    torch::Tensor mOffsetDenom;             // [mSizeofAnchors*mSizeofOffsets, 1]
    torch::Tensor mAnchorDenom;             // [mSizeofAnchors, 1]

    // 4. Cameras associated members
    int mSizeofCameras, mImHeight, mImWidth;
    float mTanFovx, mTanFovy;
    torch::Tensor mProjMatrix;
    std::vector<cv::Mat> mTrainedImages;
    std::vector<torch::Tensor> mTrainedImagesTensor;
    std::vector<torch::Tensor> mViewMatrices;
    std::vector<torch::Tensor> mProjMatrices;
    std::vector<torch::Tensor> mCameraCenters;

    float mNerfNormRadius;
    torch::Tensor mNerfNormTranslation;

    // 5. Render params
    float mNear = 0.01f;
    float mFar = 100.0f;

    float mScaleModifier = 1.f;
    bool mPrefiltered = false;

    bool mWhiteBackground = true;
    torch::Tensor mBackground = torch::tensor({1.f, 1.f, 1.f}).to(torch::kCUDA);
    
    // 6. Traning and loss members
    std::unique_ptr<torch::optim::Adam> mOptimizer;
    // std::unique_ptr<torch::optim::Adam> mAttributesOptimizer;
    // std::unique_ptr<torch::optim::Adam> mFeatureBankMLPOptimizer;
    // std::unique_ptr<torch::optim::Adam> mOpacityMLPOptimizer;
    // std::unique_ptr<torch::optim::Adam> mCovarianceMLPOptimizer;
    // std::unique_ptr<torch::optim::Adam> mColorMLPOptimizer;

    ORB_SLAM3::ExponLRFunc mAnchorSchedulerArgs;
    ORB_SLAM3::ExponLRFunc mOffsetSchedulerArgs;
    ORB_SLAM3::ExponLRFunc mFeatureBankMLPSchedulerArgs;
    ORB_SLAM3::ExponLRFunc mOpacityMLPSchedulerArgs;
    ORB_SLAM3::ExponLRFunc mCovarianceMLPSchedulerArgs;
    ORB_SLAM3::ExponLRFunc mColorMLPSchedulerArgs;

    // 7. MapPoint label management
    std::vector<long> mvpAnchorRootIndex;  

    // Voxel size management
    // torch::Tensor mVoxelSizes;
};

}