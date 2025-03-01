#include "Renderer/AnchorOptimizer.h"
#include "Renderer/Rasterizer.h"

#include <Thirdparty/simple-knn/spatial.h>

#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAEvent.h>

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
    torch::Tensor NeuralGauIndices = torch::Tensor(); 

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
            GauPos, GauColor, GauOpacity, GauScale, GauRot, NeuralOpacity, NeuralGauIndices);

        // Rasterization
        torch::Tensor Cov3DPrecomp = torch::Tensor();
        torch::Tensor SHSFeature = torch::Tensor();
        torch::Tensor Means2D = torch::zeros_like(GauPos).to(torch::kCUDA).requires_grad_(true);
        Means2D.retain_grad();

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
            .prefiltered = mPrefiltered};

        GaussianRasterizer rasterizer = GaussianRasterizer(raster_settings);

        torch::cuda::synchronize();

        auto [rendererd_image, radii] = rasterizer.forward(
            GauPos, Means2D, GauOpacity, SHSFeature, 
            GauColor, GauScale, GauRot, Cov3DPrecomp);

        // Loss and backward
        torch::Tensor loss = L1Loss(rendererd_image, GTImg);
        loss.backward(); 

        // Anchor management and Optimizer setting
        { 
            torch::NoGradGuard no_grad;
            if (iter > mOptimizationParams.StartStatistic && iter < mOptimizationParams.UpdateUntil)
            {
                AddDensificationStats(Means2D, radii, NeuralOpacity, NeuralGauIndices, VisibleVoxelIndices);

                if (iter > mOptimizationParams.UpdateFrom && iter % mOptimizationParams.UpdateInterval == 0)
                {
                    // DensifyAndPrune;
                    // DensifyAndPrune(mOptimizationParams.UpdateInterval, mOptimizationParams.MinOpacity, mOptimizationParams.SuccessTh, mOptimizationParams.DensifyGradTh);
                    DensifyAndPrune(1, 0.005, 0.8, 0.0002);
                }

            }

            //  Optimizer step
            if (iter < mOptimizationParams.Iter) {
                mOptimizer->step();
                mOptimizer->zero_grad(true);
                UpdateLR(iter);
            }

        }


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
                                             torch::Tensor& NeuralOpacity, torch::Tensor& NeuralGauIndices)
{
    using Slice = torch::indexing::Slice;

    // 1. Filter [mAchorPos, mAchorFeatures, mOffsets, mAchorScales]
    torch::Tensor AchorPos = mAchorPos.index_select(0, VisibleVoxelIndices);
    torch::Tensor AchorFeatures = mAchorFeatures.index_select(0, VisibleVoxelIndices);
    torch::Tensor Offsets = mOffsets.index_select(0, VisibleVoxelIndices);
    torch::Tensor AchorScales = mAchorScales.index_select(0, VisibleVoxelIndices);

    // 2. Get Cam-Anchor direction and distance [CamAnchorView, CamAnchorDist]
    torch::Tensor CamCenterBroadcast = CamCenter.unsqueeze(0).repeat({AchorPos.size(0), 1}); // [3] -> [VisibleAnchorNum, 3]
    torch::Tensor CamAnchorDir = AchorPos - CamCenterBroadcast;                              // [VisibleAnchorNum, 3]
    torch::Tensor CamAnchorDist = CamAnchorDir.norm(-1, true).unsqueeze(-1);                 // [VisibleAnchorNum, 1]
    torch::Tensor CamAnchorView = CamAnchorDir / CamAnchorDist.repeat({1, 3});               // [VisibleAnchorNum, 3]

    // 3. Get weighted Feature through mFeatureMLP [Index refers to https://pytorch.org/cppdocs/notes/tensor_indexing.html]
    torch::Tensor FeatureMLPInput = torch::cat({CamAnchorView, CamAnchorDist}, -1);                     // [VisibleAnchorNum, 4] on GPU
    torch::Tensor FeatureWeight = mFeatureMLP.forward(FeatureMLPInput).unsqueeze(1).repeat({1,32,1});   // [VisibleAnchorNum, mFeatureDim, 3] on GPU
    AchorFeatures = AchorFeatures.index({Slice(), Slice(torch::indexing::None, torch::indexing::None, 4)}).repeat({1,4}) * FeatureWeight.index({Slice(), Slice(), 0}) \
        + AchorFeatures.index({Slice(), Slice(torch::indexing::None, torch::indexing::None, 2)}).repeat({1,2}) * FeatureWeight.index({Slice(), Slice(), 1}) \
        + AchorFeatures * FeatureWeight.index({Slice(), Slice(), 2});                                  // [VisibleAnchorNum, 32] on GPU
    
    // 4. Get Neural Opacity and Mask
    torch::Tensor RestMLPInput = torch::cat({AchorFeatures, CamAnchorView}, -1);    // [VisibleAnchorNum, mFeatureDim+3] on GPU
    NeuralOpacity = mOpacityMLP.forward(RestMLPInput);                              // [VisibleAnchorNum, mSizeofOffsets] on GPU
    NeuralOpacity = NeuralOpacity.view({-1});                                       // [VisibleAnchorNum * mSizeofOffsets] on GPU
    NeuralGauIndices = torch::nonzero(NeuralOpacity > 0.f = true).squeeze(-1);      // [GauNum]
    GauOpacity = NeuralOpacity.index_select(0, NeuralGauIndices).unsqueeze(-1);     // [GauNum, 1]

    // 5. Get GauColor and GauCov with Mask
    GauColor = mColorMLP.forward(RestMLPInput);
    GauColor = GauColor.view({-1, 3}).index_select(0, NeuralGauIndices);            // [GauNum, 3]

    torch::Tensor GauCov = mCovarianceMLP.forward(RestMLPInput);
    GauCov = GauCov.view({-1, 7}).index_select(0, NeuralGauIndices);                // [GauNum, 7]

    // 5. Get repeatedAnchor, repeatedScale, Offsets with Mask
    torch::Tensor  GauOffsets= Offsets.view({-1, 3}).index_select(0, NeuralGauIndices);                                                    // [GauNum, 3]
    torch::Tensor  RepeatedAnchors= AchorPos.unsqueeze(1).repeat({1, mSizeofOffsets,1}).view({-1, 3}).index_select(0, NeuralGauIndices);   // [GauNum, 3]
    torch::Tensor  RepeatedScales= AchorScales.unsqueeze(1).repeat({1, mSizeofOffsets,1}).view({-1, 3}).index_select(0, NeuralGauIndices); // [GauNum, 3]

    // 6, Get GauSale, GauRot, GauPos
    GauScale = RepeatedScales * torch::sigmoid(GauCov.index({Slice(), Slice(torch::indexing::None, 3)}));   // [GauNum, 3]
    GauRot = torch::nn::functional::normalize(GauCov.index({Slice(), Slice(3, torch::indexing::None)}));    // [GauNum, 4]
    GauPos = RepeatedAnchors + GauOffsets;                                                                  // [GauNum, 3]
}

// Densification
void AnchorOptimizer::AddDensificationStats(const torch::Tensor Means2D, const torch::Tensor radii, const torch::Tensor NeuralOpacity, const torch::Tensor NeuralGauIndices, const torch::Tensor VisibleVoxelIndices)
{
    // Update mOpacityAccum [mSizeofAnchors, 1] from NeuralOpacity [AnchorNum * mSizeofOffsets]
    torch::Tensor TempOpacity = NeuralOpacity.clone().view({-1}).detach();              // [VisibleAnchorNum * mSizeofOffsets] on GPU
    torch::Tensor OpacityMaskIndices = torch::nonzero(TempOpacity < 0.f).squeeze(-1);   // [IndicesNum]
    TempOpacity.index_put_({OpacityMaskIndices}, 0);
    TempOpacity = TempOpacity.view({-1, mSizeofOffsets});                               // [VisibleAnchorNum, mSizeofOffsets] on GPU

    mOpacityAccum.index_put_({VisibleVoxelIndices}, mOpacityAccum.index_select(0, VisibleVoxelIndices) + TempOpacity.sum(1, true));
    
    // Update mAnchorDenom [mSizeofAnchors, 1]
    mAnchorDenom.index_put_({VisibleVoxelIndices}, mAnchorDenom.index_select(0, VisibleVoxelIndices) + 1);

    // Update mOffsetGradientAccum , mOffsetDenom [mSizeofAnchors*mSizeofOffsets, 1]
    torch::Tensor VisibleVoxelMask = torch::zeros({mSizeofAnchors}, torch::dtype(torch::kBool)).to(torch::kCUDA);
    VisibleVoxelMask.index_put_({VisibleVoxelIndices}, true);
    VisibleVoxelMask = VisibleVoxelMask.unsqueeze(1).repeat({1, mSizeofOffsets}).view({-1});    // [mSizeofAnchors*mSizeofOffsets] -> [VisibleAnchorNum*mSizeofOffsets] is true

    torch::Tensor NeuralGausMask = torch::zeros({NeuralOpacity.size(0)}, torch::dtype(torch::kBool)).to(torch::kCUDA);
    NeuralGausMask.index_put_({NeuralGauIndices}, true);                                        // [VisibleAnchorNum*mSizeofOffsets]

    torch::Tensor IntegratedMask = torch::zeros({mSizeofAnchors * mSizeofOffsets}, torch::dtype(torch::kBool)).to(torch::kCUDA);
    IntegratedMask.index_put_({VisibleVoxelMask}, NeuralGausMask);                              // [mSizeofAnchors*mSizeofOffsets] 

    torch::Tensor VisibilityFilter = radii > 0.f;                                               // [GauNum] 
    
    torch::Tensor IntegratedMaskCopy = IntegratedMask.clone();                                  // [mSizeofAnchors*mSizeofOffsets] 
    IntegratedMask.index_put_({IntegratedMaskCopy}, VisibilityFilter);

    torch::Tensor GradNorm = Means2D.grad().index_select(0, VisibilityFilter.nonzero().squeeze()).slice(1, 0, 2).norm(2, -1, true);
    mOffsetGradientAccum.index_put_({IntegratedMask}, mOffsetGradientAccum.index_select(0, IntegratedMask.nonzero().squeeze()) + GradNorm); 
    mOffsetDenom.index_put_({IntegratedMask}, mOffsetDenom.index_select(0, IntegratedMask.nonzero().squeeze()) + 1); 
}

void AnchorOptimizer::DensifyAndPrune(const int UpdateInterval, const float MinOpacity, const float SuccessTh, const float DensifyGradTh) 
{
    // 1. Compute gradient norm and offset denom mask
    torch::Tensor OffsetGrads = mOffsetGradientAccum / mOffsetDenom;    // [mSizeofAnchors*mSizeofOffsets, 1]
    OffsetGrads.index_put_({OffsetGrads.isnan()}, 0.0);
    torch::Tensor OffsetGradNorm = OffsetGrads.norm(-1, true);          // [mSizeofAnchors*mSizeofOffsets]

    torch::Tensor OffsetDenomMask = mOffsetDenom > UpdateInterval * SuccessTh * 0.5;
    OffsetDenomMask = OffsetDenomMask.squeeze(-1);                      // [mSizeofAnchors*mSizeofOffsets]

    // 2. Anchor Densification
    DensifyAnchors(DensifyGradTh, OffsetDenomMask, OffsetGradNorm);
    
    // 3. Update mOffsetGradientAccum, mOffsetDenom

    // 4. Compute Prune mask

    // 5. Anchor Pruning

    // 6. Update mOffsetGradientAccum, mOffsetDenom, mOpacityAccum
}

void AnchorOptimizer::DensifyAnchors(const float DensifyGradTh, const torch::Tensor OffsetDenomMask, const torch::Tensor OffsetGradNorm)
{
    // Densify anchors through different resolutions
    const int OriginalAnchorNum = mAchorPos.size(0);                              //! mAchorPos sze increased through each following cycle
    for(int i = 0; i < mHierachyLayerNum; i++)
    {
        // 1. Select significant neural Gaussians
        const float ScaledGauGradTh = DensifyGradTh * std::pow(std::sqrt(mHierachyFVoxelSizeFactor), i);
        torch::Tensor GauGradMask = OffsetGradNorm >= ScaledGauGradTh;      // [OriginalAnchorNum*mSizeofOffsets]
        GauGradMask = torch::logical_and(GauGradMask, OffsetDenomMask);     // [OriginalAnchorNum*mSizeofOffsets]

        torch::Tensor RandomEliminationMask = torch::randn({GauGradMask.size(0)}).to(torch::kCUDA) > std::pow(0.5f, i + 1);

        torch::Tensor CandidateAnchorMask = torch::logical_and(GauGradMask, RandomEliminationMask); // [OriginalAnchorNum*mSizeofOffsets]

        // Make sure the rightness of Mask's demension related to mAchorPos's size
        const int CurrentAnchorNum = mAchorPos.size(0); 

        if(CurrentAnchorNum < OriginalAnchorNum)
            break;
        else if(CurrentAnchorNum > OriginalAnchorNum)
        {
            const int IncreasedAnchorNum = CurrentAnchorNum - OriginalAnchorNum;
            CandidateAnchorMask = torch::cat({CandidateAnchorMask, torch::zeros({IncreasedAnchorNum * mSizeofOffsets}, torch::dtype(torch::kBool)).to(torch::kCUDA)}, 0);    // [CurrentAnchorNum*mSizeofOffsets]
        }

        torch::Tensor NerualGauPos = mAchorPos.unsqueeze(1).repeat({1, mSizeofOffsets, 1}) + mOffsets * torch::exp(mAchorScales.unsqueeze(1).repeat({1, mSizeofOffsets, 1})); // [CurrentAnchorNum, mSizeofOffsets, 3]
        torch::Tensor NewAnchorPos = NerualGauPos.view({-1, 3}).index_select(0, CandidateAnchorMask.nonzero().squeeze(-1));

        // Update Optimizer

        std::cout << "[>>>AnchorOptimization] Anchor Management: DensifyAndPrune: NewAnchorPos " << NewAnchorPos.nonzero() << std::endl;
        PrintCUDAUse();

    }
    

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


void AnchorOptimizer::PrintCUDAUse( )
{
    size_t free_byte;
    size_t total_byte;
 
    cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);
 
    if (cudaSuccess != cuda_status) {
        printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status));
        exit(1);
    }
 
    double free_db = (double)free_byte;
    double total_db = (double)total_byte;
    double used_db_1 = (total_db - free_db) / 1024.0 / 1024.0;
    std::cout << "Now used GPU memory " << used_db_1 << "  MB\n";
}

// Loss
torch::Tensor AnchorOptimizer::L1Loss(const torch::Tensor& network_output, const torch::Tensor& gt) 
{
    return torch::abs((network_output - gt)).mean();
}

}