#include "GaussianOptimizer.h"
#include "Renderer/Rasterizer.h"

#include "Converter.h"

#include <algorithm>
#include <c10/cuda/CUDACachingAllocator.h>
#include <memory>
#include <cassert>

#include <Thirdparty/simple-knn/spatial.h>

namespace GaussianSplatting{

float LossMonitor::Update(float newLoss) {
    if (_loss_buffer.size() >= _buffer_size) {
        _loss_buffer.pop_front();
        _rate_of_change_buffer.pop_front();
    }
    const bool buffer_empty = _loss_buffer.empty();
    const float rateOfChange = buffer_empty ? 0.f : std::abs(newLoss - _loss_buffer.back());
    _rate_of_change_buffer.push_back(rateOfChange);
    _loss_buffer.push_back(newLoss);

    // return average rate of change
    return buffer_empty ? 0.f : std::accumulate(_rate_of_change_buffer.begin(), _rate_of_change_buffer.end(), 0.f) / _rate_of_change_buffer.size();
}

bool LossMonitor::IsConverging(float threshold) {
    if (_rate_of_change_buffer.size() < _buffer_size) {
        return false;
    }
    return std::accumulate(_rate_of_change_buffer.begin(), _rate_of_change_buffer.end(), 0.f) / _rate_of_change_buffer.size() <= threshold;
}


GaussianOptimizer::GaussianOptimizer(const ORB_SLAM3::OptimizationParameters &OptimParams):
    mOptimParams(OptimParams)
{

}


GaussianOptimizer::GaussianOptimizer(const ORB_SLAM3::MonoGSOptimizationParameters &OptimParams, const cv::Mat TrainedImage, const int ImHeight, const int ImWidth, 
                                     const float TanFovx, const float TanFovy, const float Fx, const float Fy, const float Cx, const float Cy, Sophus::SE3f& Tcw,
                                     const std::vector<long> vpGaussianRootIndex, const torch::Tensor pGauWorldPos, const torch::Tensor pGauOpacity, const torch::Tensor pGauScales, 
                                     const torch::Tensor pGauWorldRot, const torch::Tensor pGauFeatureDC, const torch::Tensor pGauFeaturest):
    mMonoGSOptimParams(OptimParams), mSHDegree(3),mTrainedImage(TrainedImage), mImHeight(ImHeight), mImWidth(ImWidth), mTanFovx(TanFovx), mTanFovy(TanFovy), mFx(Fx), mFy(Fy), mCx(Cx), mCy(Cy), 
    mvpGaussianRootIndex(vpGaussianRootIndex), mMeans3D(pGauWorldPos), mOpacity(pGauOpacity), mScales(pGauScales), mRotation(pGauWorldRot), mFeaturesDC(pGauFeatureDC), mFeaturesRest(pGauFeaturest)
{
    // Gaussian num
    mSizeofGaussians = pGauWorldPos.size(0);
    
    // Camera info
    mProjMatrix = SetProjMatrixMonoGS().to(torch::kCUDA);                                                 // camera-to-NDC matrix
    mViewMatrix = GetViewMatrix(Tcw).to(torch::kCUDA);                                                    // world-to-camera matrix
    mFullProjMatrix = mViewMatrix.unsqueeze(0).bmm(mProjMatrix.unsqueeze(0)).squeeze(0).to(torch::kCUDA); // world-to-camera matrix * camera-to-NDC matrix
    mCameraCenter = mViewMatrix.inverse()[3].slice(0, 0, 3).to(torch::kCUDA);   
    
    // Image tensor
    mTrainedImageTensor = CVMatToTensor(mTrainedImage).to(torch::kCUDA);                                  // [3, height, width]  

    // Calculate cameras associated members for densification
    mNerfNormTranslation = -mCameraCenter;
    mNerfNormRadius = 0.0f;

    // Setup Optimizer
    TrainingSetupMonoGS();

    // // // Setup SSIMparams
    // mSSIMWindow = CreateWindow().to(torch::kFloat32).to(torch::kCUDA, true);       

    // std::cout << "[GaussianSplatting::GaussianOptimizer] cx, cy, fx, fy, height, width: " << mCx << ", "
    //                                                                                     << mCy << ", "
    //                                                                                     << mFx << ", "
    //                                                                                     << mFy << ", "
    //                                                                                     << mImHeight << ", "
    //                                                                                     << mImWidth << ", "
    //                                                                                     << std::endl;                 
}

GaussianOptimizer::GaussianOptimizer(const ORB_SLAM3::MonoGSOptimizationParameters &OptimParams):
    mMonoGSOptimParams(OptimParams)
{

}

void GaussianOptimizer::InitializeOptimization(const std::vector<ORB_SLAM3::KeyFrame *> &vpKFs, const std::vector<ORB_SLAM3::MapPoint *> &vpMP, const bool bInitializeScale)
{

    const int FeaturestDim = std::pow(mSHDegree + 1, 2) - 1;
    
    // Get Gaussian Data
    mSizeofGaussians = 0;
    for(int i = 0; i < vpMP.size(); i++){
        ORB_SLAM3::MapPoint* pMP = vpMP[i];
        if(pMP)
            mSizeofGaussians += pMP->GetGaussianNum();
    }
    
    std::cout << ">>>>>>>[InitializeOptimization] The numbers of Gaussian Cluster in Map: " << mSizeofGaussians << std::endl;

    mMeans3D = torch::zeros({mSizeofGaussians, 3}, torch::dtype(torch::kFloat)).to(torch::kCUDA);
    mOpacity = torch::zeros({mSizeofGaussians, 1}, torch::dtype(torch::kFloat)).to(torch::kCUDA);
    mScales = torch::zeros({mSizeofGaussians, 3}, torch::dtype(torch::kFloat)).to(torch::kCUDA);
    mRotation = torch::zeros({mSizeofGaussians, 4}, torch::dtype(torch::kFloat)).to(torch::kCUDA);
    mFeaturesDC = torch::zeros({mSizeofGaussians, 1, 3}, torch::dtype(torch::kFloat)).to(torch::kCUDA);
    mFeaturesRest = torch::zeros({mSizeofGaussians, FeaturestDim, 3}, torch::dtype(torch::kFloat)).to(torch::kCUDA);

    mvpGaussianRootIndex = std::vector<long>(mSizeofGaussians, static_cast<long>(-1));

    {
        torch::NoGradGuard no_grad;
        int GaussianClusterIndex = 0;
        for(int i = 0; i < vpMP.size(); i++)
        {
            ORB_SLAM3::MapPoint* pMP = vpMP[i];
            if(pMP)
            {
                int GaussianClusterNum = pMP->GetGaussianNum();
                if(GaussianClusterNum)
                {
                    // std::cout << "[GaussianSplatting::InitializeOptimization] OptimizerParamsGroups mMeans3D Check " << mMeans3D.is_leaf() << std::endl;
                    // std::cout << "[InitializeOptimization] The numbers of Gaussian Cluster in Map: [ " << GaussianClusterIndex << ", " << GaussianClusterIndex + GaussianClusterNum << " ]" << std::endl;
                    // std::cout << "[InitializeOptimization] The numbers of Gaussian Cluster in Map: " << pMP->GetGauWorldPos() << std::endl;
                    mMeans3D.index_put_({torch::indexing::Slice(GaussianClusterIndex, GaussianClusterIndex + GaussianClusterNum), torch::indexing::Slice()},  pMP->GetGauWorldPos());
                    mOpacity.index_put_({torch::indexing::Slice(GaussianClusterIndex, GaussianClusterIndex + GaussianClusterNum), torch::indexing::Slice()},  pMP->GetGauOpacity());
                    mScales.index_put_({torch::indexing::Slice(GaussianClusterIndex, GaussianClusterIndex + GaussianClusterNum), torch::indexing::Slice()},   pMP->GetGauScale());
                    mRotation.index_put_({torch::indexing::Slice(GaussianClusterIndex, GaussianClusterIndex + GaussianClusterNum), torch::indexing::Slice()}, pMP->GetGauWorldRot());
                    mFeaturesDC.index_put_({torch::indexing::Slice(GaussianClusterIndex, GaussianClusterIndex + GaussianClusterNum), torch::indexing::Slice(), torch::indexing::Slice()}, pMP->GetGauFeatureDC());
                    mFeaturesRest.index_put_({torch::indexing::Slice(GaussianClusterIndex, GaussianClusterIndex + GaussianClusterNum), torch::indexing::Slice(), torch::indexing::Slice()}, pMP->GetGauFeaturest());

                    for(int j = 0; j < GaussianClusterNum; j++)
                        mvpGaussianRootIndex[GaussianClusterIndex + j] = i;

                    GaussianClusterIndex += GaussianClusterNum;
                }
                
            }
        }
    }
    

    if(bInitializeScale) 
    {
        torch::Tensor dist2 = torch::clamp_min(distCUDA2(mMeans3D), 0.0000001);
        mScales = torch::log(torch::sqrt(dist2)).unsqueeze(-1).repeat({1, 3});
    }

    std::cout << "[InitializeOptimization] mSizeofGaussians: " << mSizeofGaussians << std::endl;

    // Get Camera and Image Data
    mSizeofCameras = vpKFs.size();
    vpKFs[0]->GetGaussianRenderParams(mImHeight, mImWidth, mTanFovx, mTanFovy);
    mProjMatrix = SetProjMatrix();

    std::cout << "[GaussianSplatting::Optimize] mSizeofCameras: " << mSizeofCameras << std::endl;
    std::cout << "[GaussianSplatting::Optimize] mImHeight: " << mImHeight << std::endl;
    std::cout << "[GaussianSplatting::Optimize] mImWidth: " << mImWidth << std::endl;
    std::cout << "[GaussianSplatting::Optimize] mTanFovx: " << mTanFovx << std::endl;
    std::cout << "[GaussianSplatting::Optimize] mTanFovy: " << mTanFovy << std::endl;

    for(size_t i=0; i<vpKFs.size(); i++)
    {
        ORB_SLAM3::KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        Sophus::SE3f Tcw = pKF->GetPose();

        torch::Tensor ViewMatrix = GetViewMatrix(Tcw);
        torch::Tensor FullProjMatrix = ViewMatrix.unsqueeze(0).bmm(mProjMatrix.unsqueeze(0)).squeeze(0);
        torch::Tensor CamCenter = ViewMatrix.inverse()[3].slice(0, 0, 3);

        mTrainedImages.push_back(pKF->mImRGB);
        torch::Tensor TrainedImageTensor = CVMatToTensor(pKF->mImRGB);

        mTrainedImagesTensor.push_back(TrainedImageTensor.to(torch::kCUDA)); // 1.0 / 225.
        mViewMatrices.push_back(ViewMatrix.to(torch::kCUDA));
        mProjMatrices.push_back(FullProjMatrix.to(torch::kCUDA));
        mCameraCenters.push_back(CamCenter.to(torch::kCUDA));

        std::cout << "[GaussianSplatting::Optimize] ViewMatrix: " << i << ", " << mViewMatrices[i] << std::endl;
        std::cout << "[GaussianSplatting::Optimize] ProjMatrix: " << i << ", " << mProjMatrices[i] << std::endl;
    }

    // Calculate cameras associated members for densification
    auto [mNerfNormTranslation, mNerfNormRadius] = GetNerfppNorm();

    // Setup Optimizer
    TrainingSetup();
    // Setup Loss Monitor
    mLossMonitor = new GaussianSplatting::LossMonitor(200);
    // Setup SSIMparams
    mSSIMWindow = CreateWindow().to(torch::kFloat32).to(torch::kCUDA, true);
}

void GaussianOptimizer::InitializeOptimization(ORB_SLAM3::KeyFrame* pKF)
{
    std::cout << "[GaussianOptimizer MonoGS]" << std::endl;

    // Get Camera Data
    pKF->GetGaussianRenderParams(mImHeight, mImWidth, mTanFovx, mTanFovy);
    mFx = pKF->fx;
    mFy = pKF->fy;
    mCx = pKF->cx;
    mCy = pKF->cy;
    Sophus::SE3f Tcw = pKF->GetPose(); 

    mProjMatrix = SetProjMatrixMonoGS().to(torch::kCUDA);                                                 // camera-to-NDC matrix
    mViewMatrix = GetViewMatrix(Tcw).to(torch::kCUDA);                                                    // world-to-camera matrix
    mFullProjMatrix = mViewMatrix.unsqueeze(0).bmm(mProjMatrix.unsqueeze(0)).squeeze(0).to(torch::kCUDA); // world-to-camera matrix * camera-to-NDC matrix
    mCameraCenter = mViewMatrix.inverse()[3].slice(0, 0, 3).to(torch::kCUDA);                             

    std::cout << "[GaussianSplatting::OptimizeMonoGS] KeyFrame ID: " << pKF->mnId << std::endl;
    std::cout << "[GaussianSplatting::OptimizeMonoGS] mImHeight: " << mImHeight << std::endl;
    std::cout << "[GaussianSplatting::OptimizeMonoGS] mImWidth: " << mImWidth << std::endl;
    std::cout << "[GaussianSplatting::OptimizeMonoGS] mTanFovx: " << mTanFovx << std::endl;
    std::cout << "[GaussianSplatting::OptimizeMonoGS] mTanFovy: " << mTanFovy << std::endl;
    std::cout << "[GaussianSplatting::OptimizeMonoGS] mProjMatrix: " << mProjMatrix << std::endl;
    std::cout << "[GaussianSplatting::OptimizeMonoGS] mViewMatrix: " << mViewMatrix << std::endl;
    std::cout << "[GaussianSplatting::OptimizeMonoGS] View2WorldMatrix: " << mViewMatrix.inverse() << std::endl;
    std::cout << "[GaussianSplatting::OptimizeMonoGS] mFullProjMatrix: " << mFullProjMatrix << std::endl;

    // RGB and Depth Initialization
    mTrainedImage = pKF->mImRGB;
    mTrainedImageTensor = CVMatToTensor(mTrainedImage).to(torch::kCUDA); // [3, height, width]
    mInitalDepthTensor = 2.f * torch::ones({1, mTrainedImageTensor.size(1), mTrainedImageTensor.size(2)}, torch::dtype(torch::kFloat)).to(torch::kCUDA);
    mInitalDepthTensor += torch::randn({1,mInitalDepthTensor.size(1), mInitalDepthTensor.size(2)}).to(torch::kCUDA) * 0.3;

    // std::cout << "[GaussianSplatting::OptimizeMonoGS] mInitalDepthTensor: " << mInitalDepthTensor << std::endl;
    // Single Frame Gaussian attribution Initialization
    InitializeGaussianFromRGBD(pKF->cx, pKF->cy, pKF->fx, pKF->fy);

    // Calculate cameras associated members for densification
    mNerfNormTranslation = -mCameraCenter;
    mNerfNormRadius = 0.0f;

    // Setup Optimizer
    TrainingSetupMonoGS();

    // Setup SSIMparams
    mSSIMWindow = CreateWindow().to(torch::kFloat32).to(torch::kCUDA, true);
}

void GaussianOptimizer::InitializeGaussianFromRGBD(float cx, float cy, float fx, float fy)
{
    const int FeaturestDim = std::pow(mSHDegree + 1, 2) - 1;
    mSizeofGaussians = mImHeight * mImWidth * (1.f / 32.f); // downsampling_factor
    // std::cout << "[GaussianSplatting::OptimizeMonoGS] InitializeGaussianFromRGBD: " << mSizeofGaussians << std::endl;
    const auto pointType = torch::TensorOptions().dtype(torch::kFloat32);

    // Do not need downsampling
    mOpacity = ORB_SLAM3::Converter::InverseSigmoid(0.5 * torch::ones({mSizeofGaussians, 1}, pointType)).to(torch::kCUDA);
    mScales = torch::zeros({mSizeofGaussians, 3}, pointType).to(torch::kCUDA);
    mRotation = torch::zeros({mSizeofGaussians, 4}, pointType).to(torch::kCUDA);
    mFeaturesRest = torch::zeros({mSizeofGaussians, FeaturestDim, 3}, pointType).to(torch::kCUDA);
    // Need downsampling
    mMeans3D = torch::zeros({mImHeight*mImWidth, 3}, pointType).to(torch::kCUDA);
    mFeaturesDC = torch::zeros({mImHeight*mImWidth, 1, 3}, pointType).to(torch::kCUDA);

    std::cout << "[GaussianSplatting::OptimizeMonoGS] cx, cy, fx, fy, height, width: " << cx << ", "
                                                                                        << cy << ", "
                                                                                        << fx << ", "
                                                                                        << fy << ", "
                                                                                        << mImHeight << ", "
                                                                                        << mImWidth << ", "
                                                                                        << std::endl;

    {
        torch::NoGradGuard no_grad;
        
        int GaussianIndex = 0;

        // std::cout << "[GaussianSplatting::OptimizeMonoGS] mInitalDepthTensor: " << mInitalDepthTensor  << std::endl;

        // Need CUDA version for mMeans3D and mFeaturesDC
        for(int i = 0; i < mImWidth; i++)
        {
            for(int j = 0; j < mImHeight; j++)
            {
                
                // Get Gaussian Pos through pixels [image coord to camera coord * camera coord to world coord]
                torch::Tensor depth = mInitalDepthTensor[0][j][i];
                torch::Tensor CameraCoord = torch::ones({3}, pointType).to(torch::kCUDA);
                
                CameraCoord[0] = depth * (i-cx)/fx;
                CameraCoord[1] = depth * (j-cy)/fy;
                CameraCoord[2] = depth;

                torch::Tensor Point3D = torch::ones({1, 3}, pointType).to(torch::kCUDA);
                torch::Tensor View2WorldMatrix = mViewMatrix.inverse();
                Point3D[0][0] = View2WorldMatrix[0][0]*CameraCoord[0] + View2WorldMatrix[0][1]*CameraCoord[1] + View2WorldMatrix[0][2]*CameraCoord[2] + View2WorldMatrix[3][0];
                Point3D[0][1] = View2WorldMatrix[1][0]*CameraCoord[0] + View2WorldMatrix[1][1]*CameraCoord[1] + View2WorldMatrix[1][2]*CameraCoord[2] + View2WorldMatrix[3][1];
                Point3D[0][2] = View2WorldMatrix[2][0]*CameraCoord[0] + View2WorldMatrix[2][1]*CameraCoord[1] + View2WorldMatrix[2][2]*CameraCoord[2] + View2WorldMatrix[3][2];
                

                mMeans3D.index_put_({GaussianIndex, torch::indexing::Slice()},  Point3D);

                // Get Gaussian colors
                torch::Tensor GauColor = torch::ones({1, 1, 3}, pointType).to(torch::kCUDA);
                GauColor[0][0][0] = mTrainedImageTensor[0][j][i];
                GauColor[0][0][1] = mTrainedImageTensor[1][j][i];
                GauColor[0][0][2] = mTrainedImageTensor[2][j][i];
                torch::Tensor GauColorSH = ORB_SLAM3::Converter::RGB2SH(GauColor).to(torch::kCUDA);
                // std::cout << "[GaussianSplatting::OptimizeMonoGS] GauColorSH: " << GauColorSH  << std::endl;

                mFeaturesDC.index_put_({GaussianIndex, torch::indexing::Slice(), torch::indexing::Slice()}, GauColorSH);

                GaussianIndex += 1;
            }
        }

        // PointCloud Downsampling
        std::vector<int> PCIndices = GetRandomIndices(mImHeight * mImWidth);
        std::vector<int> PCIndicesSelected(PCIndices.begin(), PCIndices.begin() + mSizeofGaussians);
        torch::Tensor PointCloudIndicesTensor = torch::from_blob(PCIndicesSelected.data(), {mSizeofGaussians}, torch::kInt32).to(torch::kCUDA);
        // std::cout << "[GaussianSplatting::OptimizeMonoGS] PointCloudIndicesTensor: " << PointCloudIndicesTensor << std::endl;

        mMeans3D = mMeans3D.index_select(0, PointCloudIndicesTensor);
        mFeaturesDC = mFeaturesDC.index_select(0, PointCloudIndicesTensor);

        // Scale initialization
        torch::Tensor dist2 = torch::clamp_min(distCUDA2(mMeans3D), 0.0000001);
        mScales = torch::log(torch::sqrt(dist2)).unsqueeze(-1).repeat({1, 3});

        // Rotation initialization
        mRotation.index_put_({torch::indexing::Slice(), 0}, 1.f); 

        // std::cout << "[GaussianSplatting::OptimizeMonoGS] mMeans3D: " << mMeans3D.size(0)  << std::endl;
        // std::cout << "[GaussianSplatting::OptimizeMonoGS] mFeaturesDC: " << mFeaturesDC  << std::endl;
        // std::cout << "[GaussianSplatting::OptimizeMonoGS] mOpacity: " << mOpacity.size(0)  << std::endl;
        // std::cout << "[GaussianSplatting::OptimizeMonoGS] mScales: " << mScales.size(0)  << std::endl;
        // std::cout << "[GaussianSplatting::OptimizeMonoGS] mRotation: " << mRotation.size(0)  << std::endl;
    }

}

void GaussianOptimizer::Optimize()
{
    // std::cout << "[GaussianOptimizer::Optimize] Start" << std::endl;

    std::vector<int> CamIndices;
    int CamIndex = 0;

    for (int iter = 1; iter < mOptimParams.iterations + 1; ++iter) {
        if (CamIndices.empty()) {
            CamIndices = GetRandomIndices(mSizeofCameras);
        }

        CamIndex = CamIndices.back();
        torch::Tensor ViewMatrix = GetViewMatrixWithIndex(CamIndex);
        torch::Tensor ProjMatrix = GetProjMatrixWithIndex(CamIndex);
        torch::Tensor CamCenter = GetCamCenterWithIndex(CamIndex);
        torch::Tensor GTImg = GetGTImgTensor(CamIndex);
        CamIndices.pop_back(); // remove last element to iterate over all cameras randomly

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
            .sh_degree = mSHDegree,
            .camera_center = CamCenter,
            .prefiltered = mPrefiltered};

        // Tensors setup
        torch::Tensor Means2D = torch::zeros_like(mMeans3D).to(torch::kCUDA).requires_grad_(true);
        Means2D.retain_grad();
        torch::Tensor Cov3DPrecomp = torch::Tensor();
        torch::Tensor ColorsPrecomp = torch::Tensor();

        // Render
        GaussianRasterizer rasterizer = GaussianRasterizer(raster_settings);
        torch::cuda::synchronize();
        auto [rendererd_image, radii] = rasterizer.forward(
            mMeans3D,
            Means2D,
            torch::sigmoid(mOpacity).to(torch::kCUDA),
            torch::cat({mFeaturesDC, mFeaturesRest}, 1).to(torch::kCUDA),
            ColorsPrecomp,
            torch::exp(mScales).to(torch::kCUDA),
            torch::nn::functional::normalize(mRotation).to(torch::kCUDA),
            Cov3DPrecomp);

        // Loss Computations
        auto L1loss =L1Loss(GTImg, rendererd_image);
        auto SSIMloss = SSIM(rendererd_image, GTImg);
        auto loss = (1.f - mOptimParams.lambda_dssim) * L1loss + mOptimParams.lambda_dssim * (1.f - SSIMloss);
        loss.backward(); 
        // std::cout << "[GaussianSplatting::Optimize] Loss: " << loss << std::endl;

        { 
            torch::NoGradGuard no_grad;

            // Image debug
            if(iter == mOptimParams.iterations)
            {
                torch::NoGradGuard no_grad;

                cv::Mat RenderImg = TensorToCVMat(rendererd_image);
                cv::imwrite("LocalBA_InitialImage.png", RenderImg);
                // std::cout << "[GaussianOptimizer::OptimizeMonoGS] Debug;RenderImg: " << rendererd_image.index({"...", 0, "..."}) << std::endl;

                cv::Mat TrianedImg = TensorToCVMat(GTImg);
                cv::imwrite("LocalBA_InitialTrainedImage.png", TrianedImg);

                float psnr = PSNR(rendererd_image, GTImg);
                std::cout << "[GaussianOptimizer::Optimize] Debug;psnr: " << psnr << std::endl;
            }
            
            // Densification
            if (iter < mOptimParams.densify_until_iter) {
                torch::Tensor VisibilityFilter = radii > 0;
                AddDensificationStats(Means2D, VisibilityFilter);
                if (iter > mOptimParams.densify_from_iter && iter % mOptimParams.densification_interval == 0) {
                    float size_threshold = iter > mOptimParams.opacity_reset_interval ? 20.f : -1.f;
                    DensifyAndPrune(mOptimParams.densify_grad_threshold, mOptimParams.min_opacity, size_threshold);
                }

                if (iter % mOptimParams.opacity_reset_interval == 0 || (mWhiteBackground && iter == mOptimParams.densify_from_iter)) {
                    ResetOpacity();
                }
            }

            //  Optimizer step
            if (iter < mOptimParams.iterations) {
                mOptimizer->step();
                mOptimizer->zero_grad(true);
                UpdateLR(iter);
            }

            // Clear cache
            if (mOptimParams.empty_gpu_cache && iter % 100) {
                c10::cuda::CUDACachingAllocator::emptyCache();
            }
        }
    }

    std::cout << "[AfterOptimization] mSizeofGaussians: " << mSizeofGaussians << std::endl;
}

void GaussianOptimizer::OptimizeMonoGS()
{
    std::cout << "[GaussianOptimizer::OptimizeMonoGS] Start" << std::endl;

    for (int iter = 1; iter < mOptimParams.iterations + 1; ++iter) {
        
        // std::cout << ">>>>>>[GaussianOptimizer::OptimizeMonoGS] Iteration = " << iter << std::endl;
        
        // Set up rasterization configuration
        GaussianRasterizationSettings raster_settings = {
            .image_height = static_cast<int>(mImHeight),
            .image_width = static_cast<int>(mImWidth),
            .tanfovx = mTanFovx,
            .tanfovy = mTanFovy,
            .bg = mBackground,
            .scale_modifier = mScaleModifier,
            .viewmatrix = mViewMatrix,
            .projmatrix = mFullProjMatrix,
            .sh_degree = GetActiveSHDegree(),
            .camera_center = mCameraCenter,
            .prefiltered = mPrefiltered};

        // Tensors setup
        torch::Tensor Means2D = torch::zeros_like(mMeans3D).to(torch::kCUDA).requires_grad_(true);
        Means2D.retain_grad();
        torch::Tensor Cov3DPrecomp = torch::Tensor();
        torch::Tensor ColorsPrecomp = torch::Tensor();

        // Render
        GaussianRasterizer rasterizer = GaussianRasterizer(raster_settings);

        torch::cuda::synchronize();

        auto [rendererd_image, radii] = rasterizer.forward(
            mMeans3D,
            Means2D,
            torch::sigmoid(mOpacity).to(torch::kCUDA),
            torch::cat({mFeaturesDC, mFeaturesRest}, 1).to(torch::kCUDA),
            ColorsPrecomp,
            torch::exp(mScales).to(torch::kCUDA),
            torch::nn::functional::normalize(mRotation).to(torch::kCUDA),
            Cov3DPrecomp);
        
        // Image debug
        // if(iter == mOptimParams.iterations)
        if(iter == mOptimParams.iterations)
        {
            torch::NoGradGuard no_grad;

            cv::Mat RenderImg = TensorToCVMat(rendererd_image);
            cv::imwrite("MonoGS_InitialImage.png", RenderImg);
            // std::cout << "[GaussianOptimizer::OptimizeMonoGS] Debug;RenderImg: " << rendererd_image.index({"...", 0, "..."}) << std::endl;

            cv::Mat TrianedImg = TensorToCVMat(mTrainedImageTensor);
            cv::imwrite("MonoGS_InitialTrainedImage.png", TrianedImg);

            float psnr = PSNR(rendererd_image, mTrainedImageTensor);
            std::cout << "[GaussianOptimizer::OptimizeMonoGS] Debug;psnr: " << psnr << std::endl;
        }

        // Loss Computations
        torch::Tensor loss = L1Loss(rendererd_image, mTrainedImageTensor);
        loss.backward(); 

        // Densify, prune and reset opacity
        { 
            torch::NoGradGuard no_grad;
            // Densification
            if (iter < mOptimParams.densify_until_iter) {
                torch::Tensor VisibilityFilter = radii > 0;
                AddDensificationStats(Means2D, VisibilityFilter);
                if (iter > mOptimParams.densify_from_iter && iter % mOptimParams.densification_interval == 0) {
                    float size_threshold = iter > mOptimParams.opacity_reset_interval ? 20.f : -1.f;
                    DensifyAndPrune(mOptimParams.densify_grad_threshold, mOptimParams.min_opacity, size_threshold);
                }

                if (iter % mOptimParams.opacity_reset_interval == 0 || (mWhiteBackground && iter == mOptimParams.densify_from_iter)) {
                    ResetOpacity();
                }
            }

            //  Optimizer step
            if (iter < mOptimParams.iterations) {
                mOptimizer->step();
                mOptimizer->zero_grad(true);
                UpdateLR(iter);
            }

            // Update SH degree
            if (iter % 1000 == 0) {
                UpdateSHDegree();
            }

            // Clear cache
            if (mOptimParams.empty_gpu_cache && iter % 100) {
                c10::cuda::CUDACachingAllocator::emptyCache();
            }
        }
    }
}

void GaussianOptimizer::DensifyAndPrune(float max_grad, float min_opacity, float max_screen_size) {
    torch::Tensor grads = mPosGradientAccum / mDenom;
    grads.index_put_({grads.isnan()}, 0.0);

    DensifyAndClone(grads, max_grad);
    DensifyAndSplit(grads, max_grad, min_opacity, max_screen_size);
}

void GaussianOptimizer::DensifyAndClone(torch::Tensor& grads, float grad_threshold)
{
    // Extract points that satisfy the gradient condition
    torch::Tensor selected_pts_mask = torch::where(torch::linalg::vector_norm(grads, {2}, 1, true, torch::kFloat32) >= grad_threshold,
                                                   torch::ones_like(grads.index({torch::indexing::Slice()})).to(torch::kBool),
                                                   torch::zeros_like(grads.index({torch::indexing::Slice()})).to(torch::kBool))
                                                   .to(torch::kLong);
    selected_pts_mask = torch::logical_and(selected_pts_mask, std::get<0>(torch::exp(mScales).max(1)).unsqueeze(-1) <= mPercentDense * mNerfNormRadius);

    auto indices = torch::nonzero(selected_pts_mask.squeeze(-1) == true).index({torch::indexing::Slice(torch::indexing::None, torch::indexing::None), torch::indexing::Slice(torch::indexing::None, 1)}).squeeze(-1);
    
    // Update mvpGaussianIndiceForest before mMeans3D is expanded
    UpdateIndiceForestAfterClone(indices);

    torch::Tensor newMeans3D = mMeans3D.index_select(0, indices);
    torch::Tensor newFeaturesDC = mFeaturesDC.index_select(0, indices);
    torch::Tensor newFeaturesRest = mFeaturesRest.index_select(0, indices);
    torch::Tensor newOpacity = mOpacity.index_select(0, indices);
    torch::Tensor newScales = mScales.index_select(0, indices);
    torch::Tensor newRotation = mRotation.index_select(0, indices);

    DensificationPostfix(newMeans3D, newFeaturesDC, newFeaturesRest, newScales, newRotation, newOpacity);
    mSizeofGaussians += indices.contiguous().numel();
    assert( mMeans3D.sizes()[0] == mSizeofGaussians );

    std::cout << "[GaussianSplatting::Gaussian Numbers] [AfterClone] " << mSizeofGaussians << std::endl;
}

void GaussianOptimizer::UpdateIndiceForestAfterClone(const torch::Tensor indices)
{
    torch::Tensor Indices = indices.contiguous().cpu();
    int CloneNum = Indices.numel();
    
    if(CloneNum)
    {
        std::vector<long> IndicesVec(Indices.data_ptr<long>(), Indices.data_ptr<long>() + CloneNum);

        const int OriginalNodeNum = mvpGaussianRootIndex.size();
        const int AfterCloneNodeNum = mvpGaussianRootIndex.size() + CloneNum;
        mvpGaussianRootIndex.resize(AfterCloneNodeNum);

        for(int i = 0; i < CloneNum; i++)
            mvpGaussianRootIndex[OriginalNodeNum + i] = mvpGaussianRootIndex[IndicesVec[i]];
            
        std::cout << "[GaussianSplatting::RootIndexSize] [AfterClone] " << mvpGaussianRootIndex.size() << std::endl;
        // std::cout << "[GaussianSplatting::RootIndexSize] [AfterClone] " << mvpGaussianRootIndex << std::endl;
    }
}

void GaussianOptimizer::DensificationPostfix(torch::Tensor& newMeans3D, torch::Tensor& newFeaturesDC, torch::Tensor& newFeaturesRest,
                                            torch::Tensor& newScales, torch::Tensor& newRotation, torch::Tensor& newOpacity) {
    
    CatTensorstoOptimizer(newMeans3D, mMeans3D, 0);
    CatTensorstoOptimizer(newFeaturesDC, mFeaturesDC, 1);
    CatTensorstoOptimizer(newFeaturesRest, mFeaturesRest, 2);
    CatTensorstoOptimizer(newScales, mScales, 3);
    CatTensorstoOptimizer(newRotation, mRotation, 4);
    CatTensorstoOptimizer(newOpacity, mOpacity, 5);

    mPosGradientAccum = torch::zeros({mMeans3D.size(0), 1}).to(torch::kCUDA);
    mDenom = torch::zeros({mMeans3D.size(0), 1}).to(torch::kCUDA);
}

void GaussianOptimizer::CatTensorstoOptimizer(torch::Tensor& extension_tensor, torch::Tensor& old_tensor, int param_position) 
{
    auto adamParamStates = std::make_unique<torch::optim::AdamParamState>(static_cast<torch::optim::AdamParamState&>(
        *mOptimizer->state()[mOptimizer->param_groups()[param_position].params()[0].unsafeGetTensorImpl()]));
    mOptimizer->state().erase(mOptimizer->param_groups()[param_position].params()[0].unsafeGetTensorImpl());

    adamParamStates->exp_avg(torch::cat({adamParamStates->exp_avg(), torch::zeros_like(extension_tensor)}, 0));
    adamParamStates->exp_avg_sq(torch::cat({adamParamStates->exp_avg_sq(), torch::zeros_like(extension_tensor)}, 0));

    mOptimizer->param_groups()[param_position].params()[0] = torch::cat({old_tensor, extension_tensor}, 0).set_requires_grad(true);
    old_tensor = mOptimizer->param_groups()[param_position].params()[0];

    mOptimizer->state()[mOptimizer->param_groups()[param_position].params()[0].unsafeGetTensorImpl()] = std::move(adamParamStates);
}

void GaussianOptimizer::PruneOptimizer(torch::Tensor& old_tensor, const torch::Tensor& mask, int param_position)
{
    auto adamParamStates = std::make_unique<torch::optim::AdamParamState>(static_cast<torch::optim::AdamParamState&>(
        *mOptimizer->state()[mOptimizer->param_groups()[param_position].params()[0].unsafeGetTensorImpl()] ));
    mOptimizer->state().erase(mOptimizer->param_groups()[param_position].params()[0].unsafeGetTensorImpl());

    adamParamStates->exp_avg(adamParamStates->exp_avg().index_select(0, mask));
    adamParamStates->exp_avg_sq(adamParamStates->exp_avg_sq().index_select(0, mask));

    mOptimizer->param_groups()[param_position].params()[0] = old_tensor.index_select(0, mask).set_requires_grad(true);
    old_tensor = mOptimizer->param_groups()[param_position].params()[0]; // update old tensor
    mOptimizer->state()[mOptimizer->param_groups()[param_position].params()[0].unsafeGetTensorImpl()] = std::move(adamParamStates);
}

void GaussianOptimizer::ResetOpacity()
{
    auto new_opacity = InverseSigmoid(torch::ones_like(mOpacity, torch::TensorOptions().dtype(torch::kFloat32)) * 0.01f);

    auto adamParamStates = std::make_unique<torch::optim::AdamParamState>(static_cast<torch::optim::AdamParamState&>(
        *mOptimizer->state()[mOptimizer->param_groups()[5].params()[0].unsafeGetTensorImpl()]));

    mOptimizer->state().erase(mOptimizer->param_groups()[5].params()[0].unsafeGetTensorImpl());

    adamParamStates->exp_avg(torch::zeros_like(new_opacity));
    adamParamStates->exp_avg_sq(torch::zeros_like(new_opacity));
    // replace tensor
    mOptimizer->param_groups()[5].params()[0] = new_opacity.set_requires_grad(true);
    mOpacity = mOptimizer->param_groups()[5].params()[0];

    mOptimizer->state()[mOptimizer->param_groups()[5].params()[0].unsafeGetTensorImpl()] = std::move(adamParamStates);

}

void GaussianOptimizer::DensifyAndSplit(torch::Tensor& grads, float grad_threshold, float min_opacity, float max_screen_size)
{
    static const int N = 2;
    const int n_init_points = mMeans3D.size(0);
    // Extract points that satisfy the gradient condition
    torch::Tensor padded_grad = torch::zeros({n_init_points}).to(torch::kCUDA);
    padded_grad.slice(0, 0, grads.size(0)) = grads.squeeze();
    torch::Tensor selected_pts_mask = torch::where(padded_grad >= grad_threshold, torch::ones_like(padded_grad).to(torch::kBool), torch::zeros_like(padded_grad).to(torch::kBool));

    selected_pts_mask = torch::logical_and(selected_pts_mask, std::get<0>(torch::exp(mScales).max(1)) > mPercentDense * mNerfNormRadius);
    auto indices = torch::nonzero(selected_pts_mask.squeeze(-1) == true).index({torch::indexing::Slice(torch::indexing::None, torch::indexing::None), torch::indexing::Slice(torch::indexing::None, 1)}).squeeze(-1);

    torch::Tensor stds = torch::exp(mScales).index_select(0, indices).repeat({N, 1});
    torch::Tensor means = torch::zeros({stds.size(0), 3}).to(torch::kCUDA);
    torch::Tensor samples = torch::randn({stds.size(0), stds.size(1)}).to(torch::kCUDA) * stds + means;
    torch::Tensor rots = RotQuaToMatrix(mRotation.index_select(0, indices)).repeat({N, 1, 1});

    torch::Tensor newMeans3D = torch::bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + mMeans3D.index_select(0, indices).repeat({N, 1});
    torch::Tensor newScales = torch::log(torch::exp(mScales).index_select(0, indices).repeat({N, 1}) / (0.8 * N));
    torch::Tensor newRotation = mRotation.index_select(0, indices).repeat({N, 1});
    torch::Tensor newFeaturesDC = mFeaturesDC.index_select(0, indices).repeat({N, 1, 1});
    torch::Tensor newFeaturesRest = mFeaturesRest.index_select(0, indices).repeat({N, 1, 1});
    torch::Tensor newOpacity = mOpacity.index_select(0, indices).repeat({N, 1});

    // Update mvpGaussianIndiceForest before mMeans3D ia expanded
    UpdateIndiceForestAfterSplit(indices);

    DensificationPostfix(newMeans3D, newFeaturesDC, newFeaturesRest, newScales, newRotation, newOpacity);
    mSizeofGaussians = mMeans3D.sizes()[0];

    std::cout << "[GaussianSplatting::Gaussian Numbers] [AfterSplit] " << mSizeofGaussians << std::endl;

    torch::Tensor prune_filter = torch::cat({selected_pts_mask.squeeze(-1), torch::zeros({N * selected_pts_mask.sum().item<int>()}).to(torch::kBool).to(torch::kCUDA)});
    prune_filter = torch::logical_or(prune_filter, (torch::sigmoid(mOpacity) < min_opacity).squeeze(-1));
    PrunePoints(prune_filter);

    std::cout << "[GaussianSplatting::Gaussian Numbers] [AfterPrune] " << mSizeofGaussians << std::endl;
}

void GaussianOptimizer::UpdateIndiceForestAfterSplit(const torch::Tensor indices)
{
    torch::Tensor Indices = indices.contiguous().cpu();
    int SpltNum = Indices.numel();
    
    if(SpltNum)
    {
        std::vector<long> IndicesVec(Indices.data_ptr<long>(), Indices.data_ptr<long>() + SpltNum);

        const int OriginalNodeNum = mvpGaussianRootIndex.size();
        const int AfterSplitNodeNum = mvpGaussianRootIndex.size() + SpltNum * 2;
        mvpGaussianRootIndex.resize(AfterSplitNodeNum);

        for(int i = 0; i < SpltNum; i++)
        {
            mvpGaussianRootIndex[OriginalNodeNum + i] = mvpGaussianRootIndex[IndicesVec[i]];
            mvpGaussianRootIndex[OriginalNodeNum + SpltNum + i] = mvpGaussianRootIndex[IndicesVec[i]];
        }

        std::cout << "[GaussianSplatting::RootIndexSize] [AfterSplit] " << mvpGaussianRootIndex.size() << std::endl;
        // std::cout << "[GaussianSplatting::RootIndexSize] [AfterSplit] " << mvpGaussianRootIndex<< std::endl;
    }

}

void GaussianOptimizer::PrunePoints(torch::Tensor mask)
{
    // reverse to keep points
    auto valid_point_mask = ~mask;
    int true_count = valid_point_mask.sum().item<int>();
    auto indices = torch::nonzero(valid_point_mask == true).index({torch::indexing::Slice(torch::indexing::None, torch::indexing::None), torch::indexing::Slice(torch::indexing::None, 1)}).squeeze(-1);
    
    // Update mvpGaussianIndiceForest
    UpdateIndiceForestAfterPrune(indices);

    PruneOptimizer(mMeans3D, indices, 0);
    PruneOptimizer(mFeaturesDC, indices, 1);
    PruneOptimizer(mFeaturesRest, indices, 2);
    PruneOptimizer(mScales, indices, 3);
    PruneOptimizer(mRotation, indices, 4);
    PruneOptimizer(mOpacity, indices, 5);

    // std::cout << "[GaussianSplatting::Densification Prune] indices: " << indices << std::endl;

    mPosGradientAccum = mPosGradientAccum.index_select(0, indices);
    mDenom = mDenom.index_select(0, indices);
    // _max_radii2D = _max_radii2D.index_select(0, indices);
    mSizeofGaussians = mMeans3D.sizes()[0];
}

void GaussianOptimizer::UpdateIndiceForestAfterPrune(const torch::Tensor indices)
{
    torch::Tensor Indices = indices.contiguous().cpu();
    int AfterPruneNum = Indices.numel();
    
    if(AfterPruneNum)
    {
        std::vector<long> GaussianRootIndex(AfterPruneNum, -1);
        std::vector<long> IndicesVec(Indices.data_ptr<long>(), Indices.data_ptr<long>() + AfterPruneNum);
        
        for(int i = 0; i < AfterPruneNum; i++)
            GaussianRootIndex[i] = mvpGaussianRootIndex[IndicesVec[i]];

        mvpGaussianRootIndex = GaussianRootIndex;

        // for(int i = 0; i < AfterPruneNum; i++)
        //     std::cout << "[GaussianSplatting::RootIndex] [AfterPrune] " << mvpGaussianRootIndex[i] << std::endl;
    
        std::cout << "[GaussianSplatting::RootIndexSize] [AfterPrune] " << mvpGaussianRootIndex.size() << std::endl;
        // std::cout << "[GaussianSplatting::RootIndexSize] [AfterPrune] " << mvpGaussianRootIndex << std::endl;
    }

}

void GaussianOptimizer::AddDensificationStats(torch::Tensor& viewspace_point_tensor, torch::Tensor& update_filter) {
    mPosGradientAccum.index_put_({update_filter}, mPosGradientAccum.index_select(0, update_filter.nonzero().squeeze()) + viewspace_point_tensor.grad().index_select(0, update_filter.nonzero().squeeze()).slice(1, 0, 2).norm(2, -1, true));
    mDenom.index_put_({update_filter}, mDenom.index_select(0, update_filter.nonzero().squeeze()) + 1);
}

void GaussianOptimizer::UpdateLR(float iteration) {
    // This is hacky because you cant change in libtorch individual parameter learning rate
    // xyz is added first, since _optimizer->param_groups() return a vector, we assume that xyz stays first
    auto lr = mPosSchedulerArgs(iteration);
    static_cast<torch::optim::AdamOptions&>(mOptimizer->param_groups()[0].options()).set_lr(lr);
}

std::vector<int> GaussianOptimizer::GetRandomIndices(const int &max_index) 
{
    std::vector<int> indices(max_index);
    std::iota(indices.begin(), indices.end(), 0);
    // Shuffle the vector
    std::shuffle(indices.begin(), indices.end(), std::default_random_engine());
    std::reverse(indices.begin(), indices.end());
    return indices;
}

torch::Tensor GaussianOptimizer::GetViewMatrixWithIndex(const int &CamIndex)
{
    return mViewMatrices[CamIndex];
}

torch::Tensor GaussianOptimizer::GetProjMatrixWithIndex(const int &CamIndex)
{
    return mProjMatrices[CamIndex];
}

torch::Tensor GaussianOptimizer::GetCamCenterWithIndex(const int &CamIndex)
{
    return mCameraCenters[CamIndex];
}

torch::Tensor GaussianOptimizer::GetGTImgTensor(const int &CamIndex)
{
    return mTrainedImagesTensor[CamIndex];
}

std::vector<long> GaussianOptimizer::GetGaussianRootIndex()
{
    return mvpGaussianRootIndex;
}

torch::Tensor GaussianOptimizer::GetWorldPos()
{
    return mMeans3D.clone();
}

torch::Tensor GaussianOptimizer::GetWorldRot()
{
    return mRotation.clone();
}

torch::Tensor GaussianOptimizer::GetScale()
{
    return mScales.clone();
}

torch::Tensor GaussianOptimizer::GetOpacity()
{
    return mOpacity.clone();
}

torch::Tensor GaussianOptimizer::GetFeaturest()
{
    return mFeaturesRest.clone();
}

torch::Tensor GaussianOptimizer::GetFeatureDC()
{
    return mFeaturesDC.clone();
}

torch::Tensor GaussianOptimizer::GetWorldPos(const torch::Tensor indices)
{
    torch::Tensor SelectTensor = mMeans3D.index_select(0, indices);
    return SelectTensor.clone();
}

torch::Tensor GaussianOptimizer::GetWorldRot(const torch::Tensor indices)
{
    torch::Tensor SelectTensor = mRotation.index_select(0, indices);
    return SelectTensor.clone();
}

torch::Tensor GaussianOptimizer::GetScale(const torch::Tensor indices)
{
    torch::Tensor SelectTensor = mScales.index_select(0, indices);
    return SelectTensor.clone();
}

torch::Tensor GaussianOptimizer::GetOpacity(const torch::Tensor indices)
{
    torch::Tensor SelectTensor = mOpacity.index_select(0, indices);
    return SelectTensor.clone();
}

torch::Tensor GaussianOptimizer::GetFeaturest(const torch::Tensor indices)
{
    torch::Tensor SelectTensor = mFeaturesRest.index_select(0, indices);
    return SelectTensor.clone();
}

torch::Tensor GaussianOptimizer::GetFeatureDC(const torch::Tensor indices)
{
    torch::Tensor SelectTensor = mFeaturesDC.index_select(0, indices);
    return SelectTensor.clone();
}

torch::Tensor GaussianOptimizer::L1Loss(const torch::Tensor& network_output, const torch::Tensor& gt) {
        return torch::abs((network_output - gt)).mean();
    }

torch::Tensor GaussianOptimizer::SSIM(const torch::Tensor& img1, const torch::Tensor& img2) {
auto mu1 = torch::nn::functional::conv2d(img1, mSSIMWindow, torch::nn::functional::Conv2dFuncOptions().padding(mWindowSize / 2).groups(mChannel));
    auto mu1_sq = mu1.pow(2);
    auto sigma1_sq = torch::nn::functional::conv2d(img1 * img1, mSSIMWindow, torch::nn::functional::Conv2dFuncOptions().padding(mWindowSize / 2).groups(mChannel)) - mu1_sq;

    auto mu2 = torch::nn::functional::conv2d(img2, mSSIMWindow, torch::nn::functional::Conv2dFuncOptions().padding(mWindowSize / 2).groups(mChannel));
    auto mu2_sq = mu2.pow(2);
    auto sigma2_sq = torch::nn::functional::conv2d(img2 * img2, mSSIMWindow, torch::nn::functional::Conv2dFuncOptions().padding(mWindowSize / 2).groups(mChannel)) - mu2_sq;

    auto mu1_mu2 = mu1 * mu2;
    auto sigma12 = torch::nn::functional::conv2d(img1 * img2, mSSIMWindow, torch::nn::functional::Conv2dFuncOptions().padding(mWindowSize / 2).groups(mChannel)) - mu1_mu2;
    auto ssim_map = ((2.f * mu1_mu2 + mC1) * (2.f * sigma12 + mC2)) / ((mu1_sq + mu2_sq + mC1) * (sigma1_sq + sigma2_sq + mC2));

    return ssim_map.mean();
}

cv::Mat GaussianOptimizer::TensorToCVMat(torch::Tensor tensor)
{
    torch::Tensor detached_tensor = tensor.detach().clamp(0.f, 1.f) * 255;
    detached_tensor = detached_tensor.to(torch::kUInt8);
    detached_tensor = detached_tensor.permute({1, 2, 0}).contiguous().to(torch::kCPU);

    int64_t height = detached_tensor.size(0);
    int64_t width = detached_tensor.size(1);
    cv::Mat mat = cv::Mat(height, width, CV_8UC3, detached_tensor.data_ptr());

    return mat.clone();
}

torch::Tensor GaussianOptimizer::CVMatToTensor(cv::Mat mat)
{
    cv::Mat matFloat;
    mat.convertTo(matFloat, CV_32F, 1.f / 255.f);
    
    auto size = matFloat.size();
    auto nChannels = matFloat.channels();
    auto tensor = torch::from_blob(matFloat.data, {size.height, size.width, nChannels}, torch::kFloat32);

    return tensor.clamp(0.f, 1.f).permute({2, 0, 1}).clone();
}

void GaussianOptimizer::TrainingSetup()
{
    mPercentDense = mOptimParams.percent_dense;
    mPosGradientAccum = torch::zeros({mMeans3D.size(0), 1}).to(torch::kCUDA);
    mDenom = torch::zeros({mMeans3D.size(0), 1}).to(torch::kCUDA);
    mPosSchedulerArgs = GaussianSplatting::Expon_lr_func(mOptimParams.position_lr_init * mSpatialLRScale,
                                                         mOptimParams.position_lr_final * mSpatialLRScale,
                                                         mOptimParams.position_lr_delay_mult,
                                                         mOptimParams.position_lr_max_steps  );

    mMeans3D.set_requires_grad(true);
    mOpacity.set_requires_grad(true);
    mScales.set_requires_grad(true);
    mRotation.set_requires_grad(true);
    mFeaturesDC.set_requires_grad(true);
    mFeaturesRest.set_requires_grad(true);

    std::vector<torch::optim::OptimizerParamGroup> OptimizerParamsGroups;
    OptimizerParamsGroups.reserve(6);
    OptimizerParamsGroups.push_back(torch::optim::OptimizerParamGroup({mMeans3D}, std::make_unique<torch::optim::AdamOptions>(mOptimParams.position_lr_init * mSpatialLRScale)));
    OptimizerParamsGroups.push_back(torch::optim::OptimizerParamGroup({mFeaturesDC}, std::make_unique<torch::optim::AdamOptions>(mOptimParams.feature_lr)));
    OptimizerParamsGroups.push_back(torch::optim::OptimizerParamGroup({mFeaturesRest}, std::make_unique<torch::optim::AdamOptions>(mOptimParams.feature_lr / 20.)));
    OptimizerParamsGroups.push_back(torch::optim::OptimizerParamGroup({mScales}, std::make_unique<torch::optim::AdamOptions>(mOptimParams.scaling_lr * mSpatialLRScale)));
    OptimizerParamsGroups.push_back(torch::optim::OptimizerParamGroup({mRotation}, std::make_unique<torch::optim::AdamOptions>(mOptimParams.rotation_lr)));
    OptimizerParamsGroups.push_back(torch::optim::OptimizerParamGroup({mOpacity}, std::make_unique<torch::optim::AdamOptions>(mOptimParams.opacity_lr)));

    static_cast<torch::optim::AdamOptions&>(OptimizerParamsGroups[0].options()).eps(1e-15);
    static_cast<torch::optim::AdamOptions&>(OptimizerParamsGroups[1].options()).eps(1e-15);
    static_cast<torch::optim::AdamOptions&>(OptimizerParamsGroups[2].options()).eps(1e-15);
    static_cast<torch::optim::AdamOptions&>(OptimizerParamsGroups[3].options()).eps(1e-15);
    static_cast<torch::optim::AdamOptions&>(OptimizerParamsGroups[4].options()).eps(1e-15);
    static_cast<torch::optim::AdamOptions&>(OptimizerParamsGroups[5].options()).eps(1e-15);

    // std::cout << "[GaussianSplatting::Optimize] OptimizerParamsGroups mMeans3D Check " << mMeans3D.is_leaf() << std::endl;
    // std::cout << "[GaussianSplatting::Optimize] OptimizerParamsGroups mFeaturesDC Check " << mFeaturesDC.is_leaf() << std::endl;
    // std::cout << "[GaussianSplatting::Optimize] OptimizerParamsGroups mFeaturesRest Check " << mFeaturesRest.is_leaf() << std::endl;
    // std::cout << "[GaussianSplatting::Optimize] OptimizerParamsGroups mScales Check " << mScales.is_leaf() << std::endl;
    // std::cout << "[GaussianSplatting::Optimize] OptimizerParamsGroups mRotation Check " << mRotation.is_leaf() << std::endl;
    // std::cout << "[GaussianSplatting::Optimize] OptimizerParamsGroups mOpacity Check " << mOpacity.is_leaf() << std::endl;

    mOptimizer = std::make_unique<torch::optim::Adam>(OptimizerParamsGroups, torch::optim::AdamOptions(0.f).eps(1e-15));  
}

void GaussianOptimizer::TrainingSetupMonoGS()
{
    mPercentDense = mMonoGSOptimParams.percent_dense;
    mPosGradientAccum = torch::zeros({mMeans3D.size(0), 1}).to(torch::kCUDA);
    mDenom = torch::zeros({mMeans3D.size(0), 1}).to(torch::kCUDA);
    mPosSchedulerArgs = GaussianSplatting::Expon_lr_func(mMonoGSOptimParams.position_lr_init * mSpatialLRScale,
                                                         mMonoGSOptimParams.position_lr_final * mSpatialLRScale,
                                                         mMonoGSOptimParams.position_lr_delay_mult,
                                                         mMonoGSOptimParams.position_lr_max_steps  );
    
    mMeans3D.set_requires_grad(true);
    mOpacity.set_requires_grad(true);
    mScales.set_requires_grad(true);
    mRotation.set_requires_grad(true);
    mFeaturesDC.set_requires_grad(true);
    mFeaturesRest.set_requires_grad(true);

    std::vector<torch::optim::OptimizerParamGroup> OptimizerParamsGroups;
    OptimizerParamsGroups.reserve(6);
    OptimizerParamsGroups.push_back(torch::optim::OptimizerParamGroup({mMeans3D}, std::make_unique<torch::optim::AdamOptions>(mMonoGSOptimParams.position_lr_init * mSpatialLRScale)));
    OptimizerParamsGroups.push_back(torch::optim::OptimizerParamGroup({mFeaturesDC}, std::make_unique<torch::optim::AdamOptions>(mMonoGSOptimParams.feature_lr)));
    OptimizerParamsGroups.push_back(torch::optim::OptimizerParamGroup({mFeaturesRest}, std::make_unique<torch::optim::AdamOptions>(mMonoGSOptimParams.feature_lr / 20.)));
    OptimizerParamsGroups.push_back(torch::optim::OptimizerParamGroup({mScales}, std::make_unique<torch::optim::AdamOptions>(mMonoGSOptimParams.scaling_lr * mSpatialLRScale)));
    OptimizerParamsGroups.push_back(torch::optim::OptimizerParamGroup({mRotation}, std::make_unique<torch::optim::AdamOptions>(mMonoGSOptimParams.rotation_lr)));
    OptimizerParamsGroups.push_back(torch::optim::OptimizerParamGroup({mOpacity}, std::make_unique<torch::optim::AdamOptions>(mMonoGSOptimParams.opacity_lr)));

    static_cast<torch::optim::AdamOptions&>(OptimizerParamsGroups[0].options()).eps(1e-15);
    static_cast<torch::optim::AdamOptions&>(OptimizerParamsGroups[1].options()).eps(1e-15);
    static_cast<torch::optim::AdamOptions&>(OptimizerParamsGroups[2].options()).eps(1e-15);
    static_cast<torch::optim::AdamOptions&>(OptimizerParamsGroups[3].options()).eps(1e-15);
    static_cast<torch::optim::AdamOptions&>(OptimizerParamsGroups[4].options()).eps(1e-15);
    static_cast<torch::optim::AdamOptions&>(OptimizerParamsGroups[5].options()).eps(1e-15);

    // std::cout << "[GaussianSplatting::Optimize] OptimizerParamsGroups mMeans3D Check " << mMeans3D.is_leaf() << std::endl;
    // std::cout << "[GaussianSplatting::Optimize] OptimizerParamsGroups mFeaturesDC Check " << mFeaturesDC.is_leaf() << std::endl;
    // std::cout << "[GaussianSplatting::Optimize] OptimizerParamsGroups mFeaturesRest Check " << mFeaturesRest.is_leaf() << std::endl;
    // std::cout << "[GaussianSplatting::Optimize] OptimizerParamsGroups mScales Check " << mScales.is_leaf() << std::endl;
    // std::cout << "[GaussianSplatting::Optimize] OptimizerParamsGroups mRotation Check " << mRotation.is_leaf() << std::endl;
    // std::cout << "[GaussianSplatting::Optimize] OptimizerParamsGroups mOpacity Check " << mOpacity.is_leaf() << std::endl;

    mOptimizer = std::make_unique<torch::optim::Adam>(OptimizerParamsGroups, torch::optim::AdamOptions(0.f).eps(1e-15));  
}

std::pair<torch::Tensor, float> GaussianOptimizer::GetNerfppNorm() {

    auto [center, diagonal] = GetCenterAndDiag();
    return {-center,  diagonal * 1.1f};
}

std::pair<torch::Tensor, float> GaussianOptimizer::GetCenterAndDiag() {

    torch::Tensor avg_cam_center = torch::zeros({3}, torch::dtype(torch::kFloat)).to(torch::kCUDA);
    for (int i = 0; i < mSizeofCameras; i++) {
        avg_cam_center += mCameraCenters[i];
    }
    avg_cam_center /= static_cast<float>(mSizeofCameras);

    float max_dist = 0.0f;
    for (int i = 0; i < mSizeofCameras; i++) {
        torch::Tensor dist = (mCameraCenters[i] - avg_cam_center).norm().cpu();
        max_dist = std::max(max_dist, dist.data_ptr<float>()[0]);
    }
    std::cout << "[GaussianSplatting::Optimize] Camera extent max_dist: " << max_dist << std::endl;
    return {avg_cam_center, max_dist};
}

torch::Tensor GaussianOptimizer::GetViewMatrix(Sophus::SE3f &Tcw)
{  
    Sophus::SE3f Twc = Tcw.inverse();
    Eigen::Matrix<float,3,3> Rwc = Twc.rotationMatrix();
    Eigen::Matrix<float,3,1> twc = Twc.translation();

    Eigen::Matrix4f W2C = Eigen::Matrix4f::Zero();
    W2C.block<3, 3>(0, 0) = Rwc;
    W2C.block<3, 1>(0, 3) = twc;
    W2C(3, 3) = 1.0;
    // Here we create a torch::Tensor from the Eigen::Matrix
    // Note that the tensor will be on the CPU, you may want to move it to the desired device later
    auto W2CTensor = torch::from_blob(W2C.data(), {4, 4}, torch::kFloat);
    // clone the tensor to allocate new memory, as from_blob shares the same memory
    // this step is important if Rt will go out of scope and the tensor will be used later
    return W2CTensor.clone();
}

torch::Tensor GaussianOptimizer::SetProjMatrix()
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
    torch::Tensor ProjMatrix = torch::from_blob(P.data(), {4, 4}, torch::kFloat);
    return ProjMatrix.clone();
}

torch::Tensor GaussianOptimizer::SetProjMatrixMonoGS()
{
    // float top = mTanFovy * mNear;
    // float bottom = -top;
    // float right = mTanFovx * mNear;
    // float left = -right;

    float left_ = ((2 * mCx - mImWidth) / mImWidth - 1.0) * mImWidth / 2.0;
    float right_ = ((2 * mCx - mImWidth) / mImWidth + 1.0) * mImWidth / 2.0;
    float top_ = ((2 * mCy - mImHeight) / mImHeight + 1.0) * mImHeight / 2.0;
    float bottom_ = ((2 * mCy - mImHeight) / mImHeight - 1.0) * mImHeight / 2.0;

    float left = mNear / mFx * left_;
    float right = mNear / mFx * right_;
    float top = mNear / mFy * top_;
    float bottom = mNear / mFy * bottom_;

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
    torch::Tensor ProjMatrix = torch::from_blob(P.data(), {4, 4}, torch::kFloat);
    return ProjMatrix.clone();
}

/**
* @brief Builds a rotation matrix from a tensor of quaternions.
*
* @param r Tensor of quaternions with shape (N, 4).
* @return Tensor of rotation matrices with shape (N, 3, 3).
*/
torch::Tensor GaussianOptimizer::RotQuaToMatrix(torch::Tensor r) {
    torch::Tensor norm = torch::sqrt(torch::sum(r.pow(2), 1));
    torch::Tensor q = r / norm.unsqueeze(-1);

    using Slice = torch::indexing::Slice;
    torch::Tensor R = torch::zeros({q.size(0), 3, 3}, torch::device(torch::kCUDA));
    torch::Tensor r0 = q.index({Slice(), 0});
    torch::Tensor x = q.index({Slice(), 1});
    torch::Tensor y = q.index({Slice(), 2});
    torch::Tensor z = q.index({Slice(), 3});

    R.index_put_({Slice(), 0, 0}, 1 - 2 * (y * y + z * z));
    R.index_put_({Slice(), 0, 1}, 2 * (x * y - r0 * z));
    R.index_put_({Slice(), 0, 2}, 2 * (x * z + r0 * y));
    R.index_put_({Slice(), 1, 0}, 2 * (x * y + r0 * z));
    R.index_put_({Slice(), 1, 1}, 1 - 2 * (x * x + z * z));
    R.index_put_({Slice(), 1, 2}, 2 * (y * z - r0 * x));
    R.index_put_({Slice(), 2, 0}, 2 * (x * z - r0 * y));
    R.index_put_({Slice(), 2, 1}, 2 * (y * z + r0 * x));
    R.index_put_({Slice(), 2, 2}, 1 - 2 * (x * x + y * y));
    return R;
}

torch::Tensor GaussianOptimizer::GaussianKernel1D(int window_size, float sigma) {
        torch::Tensor gauss = torch::empty(window_size);
        for (int x = 0; x < window_size; ++x) {
            gauss[x] = std::exp(-(std::pow(std::floor(static_cast<float>(x - window_size) / 2.f), 2)) / (2.f * sigma * sigma));
        }
        return gauss / gauss.sum();
    }

torch::Tensor GaussianOptimizer::CreateWindow()
    {
    auto _1D_window = GaussianKernel1D(mWindowSize, 1.5).unsqueeze(1);
    auto _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0);
    return _2D_window.expand({mChannel, 1, mWindowSize, mWindowSize}).contiguous();
}

void GaussianOptimizer::UpdateSHDegree()
{
    if (mActiveSHDegree < mMaxSHDegree) {
        mActiveSHDegree++;
    }
}

int GaussianOptimizer::GetActiveSHDegree()
{
    return mActiveSHDegree;
}

// Evaluation function
float GaussianOptimizer::PSNR(const torch::Tensor& rendered_img, const torch::Tensor& gt_img) {
    torch::Tensor squared_diff = (rendered_img - gt_img).pow(2);
    torch::Tensor mse_val = squared_diff.view({rendered_img.size(0), -1}).mean(1, true);
    return (20.f * torch::log10(1.0 / mse_val.sqrt())).mean().item<float>();
}

}