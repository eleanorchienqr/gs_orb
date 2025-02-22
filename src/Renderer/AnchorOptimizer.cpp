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
                    std::cout << "[GlobalAchorInitOptimization->AnchorOptimizer] constructor" << std::endl;
                    //TODO mProjMatrices, mCameraCenters, mTrainedImagesTensor move to AnchorOptimization constructor

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

void AnchorOptimizer::UpdateLR(float iteration)
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

}