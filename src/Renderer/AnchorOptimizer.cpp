#include "Renderer/AnchorOptimizer.h"
#include "Renderer/Rasterizer.h"

#include <Thirdparty/simple-knn/spatial.h>

namespace GaussianSplatting{
    
AnchorOptimizer::AnchorOptimizer(int SizeofInitAnchors, const int AnchorFeatureDim, const int AnchorSizeofOffsets, const int CamNum, 
                torch::Tensor AnchorWorldPos, torch::Tensor AnchorFeatures, torch::Tensor AnchorScales, 
                torch::Tensor AnchorRotations, torch::Tensor AnchorOffsets,
                ORB_SLAM3::FeatureBankMLP FBNet, ORB_SLAM3::OpacityMLP OpacityNet, ORB_SLAM3::CovarianceMLP CovNet, ORB_SLAM3::ColorMLP ColorNet,
                const int ImHeight, const int ImWidth, const float TanFovx, const float TanFovy,
                std::vector<torch::Tensor> ViewMatrices, std::vector<cv::Mat> TrainedImages):
                mSizeofAnchors(SizeofInitAnchors), mFeatureDim(AnchorFeatureDim), mSizeofOffsets(AnchorSizeofOffsets), mSizeofCameras(CamNum),
                mAchorPos(AnchorWorldPos), mAchorFeatures(AnchorFeatures), mAchorScales(AnchorScales), mAchorRotations(AnchorRotations), mOffsets(AnchorOffsets),
                mFeatureMLP(FBNet),mOpacityMLP(OpacityNet), mCovarianceMLP(CovNet), mColorMLP(ColorNet),
                mImHeight(ImHeight), mImWidth(ImWidth), mTanFovx(TanFovx), mTanFovy(TanFovy), 
                mViewMatrices(ViewMatrices), mTrainedImages(TrainedImages)
                {
                    //TODO mProjMatrices, mCameraCenters, mTrainedImagesTensor move to AnchorOptimization constructor
                    SetProjMatrix();
                    
                    mProjMatrices.reserve(CamNum);
                    mCameraCenters.reserve(CamNum);
                    mTrainedImagesTensor.reserve(CamNum);

                    for(size_t i=0; i < CamNum; i++)
                    {
                        torch::Tensor ViewMatrix = mViewMatrices[i];
                        torch::Tensor FullProjMatrix = ViewMatrix.unsqueeze(0).bmm(mProjMatrix.unsqueeze(0)).squeeze(0);
                        torch::Tensor CamCenter = ViewMatrix.inverse()[3].slice(0, 0, 3);

                        torch::Tensor TrainedImageTensor = ORB_SLAM3::Converter::CVMatToTensor(mTrainedImages[i]);

                        mProjMatrices.push_back(FullProjMatrix.to(torch::kCUDA));
                        mCameraCenters.push_back(CamCenter.to(torch::kCUDA));
                        mTrainedImagesTensor.push_back(TrainedImageTensor.to(torch::kCUDA));    // 1.0 / 225.

                        std::cout << "[>>>AnchorOptimization] ViewMatrix: " << i << ", " << mViewMatrices[i] << std::endl;
                        std::cout << "[>>>AnchorOptimization] ProjMatrix: " << i << ", " << mProjMatrices[i] << std::endl;
                    }

                    // Anchor Management Members initialization
                    mOpacityAccum = torch::zeros({mSizeofAnchors, 1}, torch::dtype(torch::kFloat)).to(torch::kCUDA);                             // [mSizeofAnchors, 1]
                    mOffsetGradientAccum = torch::zeros({mSizeofAnchors * mSizeofOffsets, 1}, torch::dtype(torch::kFloat)).to(torch::kCUDA);     // [mSizeofAnchors*mSizeofOffsets, 1]
                    mOffsetDenom = torch::zeros({mSizeofAnchors * mSizeofOffsets, 1}, torch::dtype(torch::kFloat)).to(torch::kCUDA);             // [mSizeofAnchors*mSizeofOffsets, 1]
                    mAnchorDenom = torch::zeros({mSizeofAnchors, 1}, torch::dtype(torch::kFloat)).to(torch::kCUDA);                              // [mSizeofAnchors, 1]
                }

void AnchorOptimizer::TrainingSetup()
{
    // Scheduler setting
    mAnchorSchedulerArgs = ORB_SLAM3::ExponLRFunc(mOptimizationParams.AnchorLRInit * mOptimizationParams.SpatialLRScale,
                                                  mOptimizationParams.AnchorLRFinal * mOptimizationParams.SpatialLRScale,
                                                  mOptimizationParams.AnchorLRDelayMult,
                                                  mOptimizationParams.AnchorLRMaxSteps);
    mOffsetSchedulerArgs = ORB_SLAM3::ExponLRFunc(mOptimizationParams.OffsetLRInit * mOptimizationParams.SpatialLRScale,
                                                  mOptimizationParams.OffsetLRFinal * mOptimizationParams.SpatialLRScale,
                                                  mOptimizationParams.OffsetLRDelayMult,
                                                  mOptimizationParams.OffsetLRMaxSteps);
    mFeatureBankMLPSchedulerArgs = ORB_SLAM3::ExponLRFunc(mOptimizationParams.FeatureBankMLPLRInit,
                                                  mOptimizationParams.FeatureBankMLPLRFinal,
                                                  mOptimizationParams.FeatureBankMLPLRDelayMult,
                                                  mOptimizationParams.FeatureBankMLPLRMaxSteps);
    mOpacityMLPSchedulerArgs = ORB_SLAM3::ExponLRFunc(mOptimizationParams.OpacityMLPLRInit,
                                                  mOptimizationParams.OpacityMLPLRFinal,
                                                  mOptimizationParams.OpacityMLPLRDelayMult,
                                                  mOptimizationParams.OpacityMLPLRMaxSteps);
    mCovarianceMLPSchedulerArgs = ORB_SLAM3::ExponLRFunc(mOptimizationParams.CovarianceMLPLRInit,
                                                  mOptimizationParams.CovarianceMLPLRFinal,
                                                  mOptimizationParams.CovarianceMLPLRDelayMult,
                                                  mOptimizationParams.CovarianceMLPLRMaxSteps);
    mColorMLPSchedulerArgs = ORB_SLAM3::ExponLRFunc(mOptimizationParams.ColorMLPLRInit,
                                                  mOptimizationParams.ColorMLPLRFinal,
                                                  mOptimizationParams.ColorMLPLRDelayMult,
                                                  mOptimizationParams.ColorMLPLRMaxSteps);

    // Gradient setting
    mAchorPos.set_requires_grad(true);
    mOffsets.set_requires_grad(true);
    mAchorFeatures.set_requires_grad(true);
    mAchorScales.set_requires_grad(true);
    mAchorRotations.set_requires_grad(true);

    // Optimizer setting
    std::vector<torch::optim::OptimizerParamGroup> OptimizerParamsGroups;
    OptimizerParamsGroups.reserve(9);

    OptimizerParamsGroups.push_back(torch::optim::OptimizerParamGroup({mAchorPos}, std::make_unique<torch::optim::AdamOptions>(mOptimizationParams.AnchorLRInit * mOptimizationParams.SpatialLRScale)));
    OptimizerParamsGroups.push_back(torch::optim::OptimizerParamGroup({mOffsets}, std::make_unique<torch::optim::AdamOptions>(mOptimizationParams.OffsetLRInit * mOptimizationParams.SpatialLRScale)));
    OptimizerParamsGroups.push_back(torch::optim::OptimizerParamGroup({mAchorFeatures}, std::make_unique<torch::optim::AdamOptions>(mOptimizationParams.FeatureLR)));
    OptimizerParamsGroups.push_back(torch::optim::OptimizerParamGroup({mAchorScales}, std::make_unique<torch::optim::AdamOptions>(mOptimizationParams.ScalingLR)));
    OptimizerParamsGroups.push_back(torch::optim::OptimizerParamGroup({mAchorRotations}, std::make_unique<torch::optim::AdamOptions>(mOptimizationParams.RotationLR)));

    OptimizerParamsGroups.push_back(torch::optim::OptimizerParamGroup({mFeatureMLP.parameters()}, std::make_unique<torch::optim::AdamOptions>(mOptimizationParams.FeatureBankMLPLRInit)));
    OptimizerParamsGroups.push_back(torch::optim::OptimizerParamGroup({mOpacityMLP.parameters()}, std::make_unique<torch::optim::AdamOptions>(mOptimizationParams.OpacityMLPLRInit)));
    OptimizerParamsGroups.push_back(torch::optim::OptimizerParamGroup({mCovarianceMLP.parameters()}, std::make_unique<torch::optim::AdamOptions>(mOptimizationParams.CovarianceMLPLRInit)));
    OptimizerParamsGroups.push_back(torch::optim::OptimizerParamGroup({mColorMLP.parameters()}, std::make_unique<torch::optim::AdamOptions>(mOptimizationParams.ColorMLPLRInit)));    

    static_cast<torch::optim::AdamOptions&>(OptimizerParamsGroups[0].options()).eps(1e-15);
    static_cast<torch::optim::AdamOptions&>(OptimizerParamsGroups[1].options()).eps(1e-15);
    static_cast<torch::optim::AdamOptions&>(OptimizerParamsGroups[2].options()).eps(1e-15);
    static_cast<torch::optim::AdamOptions&>(OptimizerParamsGroups[3].options()).eps(1e-15);
    static_cast<torch::optim::AdamOptions&>(OptimizerParamsGroups[4].options()).eps(1e-15);

    static_cast<torch::optim::AdamOptions&>(OptimizerParamsGroups[5].options()).eps(1e-15);
    static_cast<torch::optim::AdamOptions&>(OptimizerParamsGroups[6].options()).eps(1e-15);
    static_cast<torch::optim::AdamOptions&>(OptimizerParamsGroups[7].options()).eps(1e-15);
    static_cast<torch::optim::AdamOptions&>(OptimizerParamsGroups[8].options()).eps(1e-15);

    mOptimizer = std::make_unique<torch::optim::Adam>(OptimizerParamsGroups, torch::optim::AdamOptions(0.f).eps(1e-15));  
}

void AnchorOptimizer::Optimize()
{
    // Version 1: use ini KeyFrame as 
    torch::Tensor ViewMatrix = mViewMatrices[0];
    torch::Tensor ProjMatrix = mProjMatrices[0];
    torch::Tensor CamCenter = mCameraCenters[0];
    torch::Tensor GTImg = mTrainedImagesTensor[0];

    // Important variables
    torch::Tensor VisibleVoxelIndices = torch::Tensor();
    torch::Tensor NeuralOpacity = torch::Tensor(); // After VisibleVoxelMask Before NeuralGauMask
    torch::Tensor NeuralGauMask = torch::Tensor(); 

    torch::Tensor GauPos = torch::Tensor();        // After VisibleVoxelMask + NeuralGauMask
    torch::Tensor GauColor = torch::Tensor();      // After VisibleVoxelMask + NeuralGauMask
    torch::Tensor GauOpacity = torch::Tensor();    // After VisibleVoxelMask + NeuralGauMask
    torch::Tensor GauScale = torch::Tensor();      // After VisibleVoxelMask + NeuralGauMask
    torch::Tensor GauRot = torch::Tensor();        // After VisibleVoxelMask + NeuralGauMask

    for (int iter = 1; iter < mOptimizationParams.Iter + 1; ++iter) 
    {
        

        // Filter anchors in the Frustum VisibleVoxelMask [AnchorNum]
        PrefilterVoxel(ViewMatrix, ProjMatrix, CamCenter, VisibleVoxelIndices);

        // Neural Gaussian derivation
        GenerateNeuralGaussian(CamCenter, VisibleVoxelIndices, 
            GauPos, GauColor, GauOpacity, GauScale, GauRot, 
            NeuralOpacity, NeuralGauMask);

        // Rasterization

        // Loss and backward

        // Anchor management

    }

}

// Filters
void AnchorOptimizer::PrefilterVoxel(const torch::Tensor ViewMatrix, const torch::Tensor ProjMatrix, const torch::Tensor CamCenter, torch::Tensor& VisibleVoxelIndices)
{

    // Note: no gradient
    // Important variables
    torch::Tensor Cov3DPrecomp = torch::Tensor();
    
    // Set up rasterization configuration
    GaussianRasterizationSettings raster_settings = {
        .image_height = static_cast<int>(mImHeight),
        .image_width = static_cast<int>(mImWidth),
        .tanfovx = mTanFovx,
        .tanfovy = mTanFovy,
        .bg = mBackground,
        .scale_modifier = mScaleModifier,
        .viewmatrix = ViewMatrix,
        .projmatrix = ProjMatrix,
        .sh_degree = 1,
        .camera_center = CamCenter,
        .prefiltered = false};
    
    GaussianRasterizer rasterizer = GaussianRasterizer(raster_settings);
    torch::cuda::synchronize();

    torch::Tensor radii = rasterizer.visible_filter(
        mAchorPos, 
        torch::exp(mAchorScales).to(torch::kCUDA), 
        torch::nn::functional::normalize(mAchorRotations).to(torch::kCUDA),
        Cov3DPrecomp);

    torch::Tensor VisibleVoxelMask = radii > 0;
    VisibleVoxelIndices = torch::nonzero(VisibleVoxelMask == true).squeeze(-1);
}

void AnchorOptimizer::GenerateNeuralGaussian(const torch::Tensor CamCenter, const torch::Tensor VisibleVoxelIndices, 
                                             torch::Tensor& GauPos, torch::Tensor& GauColor, torch::Tensor& GauOpacity, 
                                             torch::Tensor& GauScale, torch::Tensor& GauRot,
                                             torch::Tensor& NeuralOpacity, torch::Tensor& NeuralGauMask)
{
    std::cout << "[>>>AnchorOptimization] GenerateNeuralGaussian"  << std::endl;
    // 1. Filter [mAchorPos, mAchorFeatures, mOffsets, mAchorScales]
    mAchorPos = mAchorPos.index_select(0, VisibleVoxelIndices);
    mAchorFeatures = mAchorFeatures.index_select(0, VisibleVoxelIndices);
    mOffsets = mOffsets.index_select(0, VisibleVoxelIndices);
    mAchorScales = mAchorScales.index_select(0, VisibleVoxelIndices);

    // 2. Get Cam-Anchor direction and distance [CamAnchorView, CamAnchorDist]
    // torch::Tensor CamAnchorView = mAchorPos - CamCenter [3];
    torch::Tensor CamCenterBroadcast = CamCenter.unsqueeze(0).repeat({mAchorPos.size(0), 1}); // [3] -> [AnchorNum, 3]
    torch::Tensor CamAnchorDir = mAchorPos - CamCenterBroadcast;                              // [AnchorNum, 3]
    torch::Tensor CamAnchorDist = CamAnchorDir.norm(-1, true).unsqueeze(-1);                  // [AnchorNum, 1]
    torch::Tensor CamAnchorView = CamAnchorDir / CamAnchorDist.repeat({1, 3});                // [AnchorNum, 3]

    // 3. Get weighted Feature through mFeatureMLP
    torch::Tensor FeatureMLPInput = torch::cat({CamAnchorView, CamAnchorDist}, -1);           // [AnchorNum, 4] on GPU
    // torch::Tensor FeatureWeight = mFeatureMLP.forward(FeatureMLPInput);                       // [AnchorNum, 3] on GPU
    // struct Net : torch::nn::Module {
    //     Net()
    //     {
    //         linear1 = register_module("linear1", torch::nn::Linear(4, 32));
    //     }

    //     torch::Tensor forward(torch::Tensor input) {
    //         return linear1->forward(input);
    //     }

    //     torch::nn::Linear linear1{nullptr};
    // };

    // Net net();
    // std::cout << "[>>>AnchorOptimization] GenerateNeuralGaussian: MLPTest: " << net.forward(torch::ones({mAchorPos.size(0), 4})) << std::endl;
    // bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1) # [n, 1, 3]
    // ## multi-resolution feat
    // feat = feat.unsqueeze(dim=-1)
    // feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
    //     feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
    //     feat[:,::1, :1]*bank_weight[:,:,2:]
    // feat = feat.squeeze(dim=-1) # [n, c]

    // for (const auto& p : mFeatureMLP.parameters()) {
    //     std::cout << "[>>>AnchorOptimization] GenerateNeuralGaussian: FeatureMLP params: " <<  p << std::endl;
    // }

    // std::cout << "[>>>AnchorOptimization] GenerateNeuralGaussian: FeatureWeight: " <<  FeatureWeight << std::endl;
}

void AnchorOptimizer::UpdateLR(const float iteration)
{
    // TODO Learning rate update check 
    // This is hacky because you cant change in libtorch individual parameter learning rate
    // xyz is added first, since _optimizer->param_groups() return a vector, we assume that xyz stays first

    float AchorPosLR = mAnchorSchedulerArgs(iteration);
    float OffsetLR = mOffsetSchedulerArgs(iteration);

    float FeatureBankMLPLR = mFeatureBankMLPSchedulerArgs(iteration);
    float OpacityMLPLR = mOpacityMLPSchedulerArgs(iteration);
    float CovarianceMLPLR = mCovarianceMLPSchedulerArgs(iteration);
    float ColorMLPLR = mColorMLPSchedulerArgs(iteration);

    static_cast<torch::optim::AdamOptions&>(mOptimizer->param_groups()[0].options()).set_lr(AchorPosLR);
    static_cast<torch::optim::AdamOptions&>(mOptimizer->param_groups()[1].options()).set_lr(OffsetLR);

    static_cast<torch::optim::AdamOptions&>(mOptimizer->param_groups()[5].options()).set_lr(FeatureBankMLPLR);
    static_cast<torch::optim::AdamOptions&>(mOptimizer->param_groups()[6].options()).set_lr(OpacityMLPLR);
    static_cast<torch::optim::AdamOptions&>(mOptimizer->param_groups()[7].options()).set_lr(CovarianceMLPLR);
    static_cast<torch::optim::AdamOptions&>(mOptimizer->param_groups()[8].options()).set_lr(ColorMLPLR);
}

// Utils func
void AnchorOptimizer::SetProjMatrix()
{
    float top = mTanFovy * mNear;
    float bottom = -top;
    float right = mTanFovx * mNear;
    float left = -right;

    Eigen::Matrix4f P = Eigen::Matrix4f::Zero();

    float z_sign = 1.f;

    P(0, 0) = 2.f * mNear / (right - left);
    P(1, 1) = 2.f * mNear / (top - bottom);
    P(0, 2) = (right + left) / (right - left);
    P(1, 2) = (top + bottom) / (top - bottom);
    P(3, 2) = z_sign;
    P(2, 2) = z_sign * mFar / (mFar - mNear);
    P(2, 3) = -(mFar * mNear) / (mFar - mNear);

    // create torch::Tensor from Eigen::Matrix
    mProjMatrix = torch::from_blob(P.data(), {4, 4}, torch::kFloat).to(torch::kCUDA);
}


}