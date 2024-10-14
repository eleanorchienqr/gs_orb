#include "GaussianOptimizer.h"
#include "Renderer/Rasterizer.h"

#include <c10/cuda/CUDACachingAllocator.h>

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

void GaussianOptimizer::InitializeOptimization(const std::vector<ORB_SLAM3::KeyFrame *> &vpKFs, const std::vector<ORB_SLAM3::MapGaussian *> &vpMG)
{

    std::cout << ">>>>>>>[InitializeOptimization] The numbers of Gaussians in Map: " << vpMG.size() << std::endl;
    
    // Get Gaussian Data
    mSizeofGaussians = vpMG.size();
    mMeans3D = torch::zeros({mSizeofGaussians, 3}, torch::dtype(torch::kFloat)).to(torch::kCUDA);
    mOpacity = torch::zeros({mSizeofGaussians, 1}, torch::dtype(torch::kFloat)).to(torch::kCUDA);
    mScales = torch::zeros({mSizeofGaussians, 3}, torch::dtype(torch::kFloat)).to(torch::kCUDA);
    mRotation = torch::zeros({mSizeofGaussians, 4}, torch::dtype(torch::kFloat)).to(torch::kCUDA);
    mMeans2D = torch::zeros({mSizeofGaussians, 3}, torch::dtype(torch::kFloat)).to(torch::kCUDA);

    torch::Tensor Features = torch::zeros({mSizeofGaussians, 3, static_cast<long>(std::pow((mSHDegree + 1), 2))}, torch::dtype(torch::kFloat)).to(torch::kCUDA);
    for(size_t i=0; i<vpMG.size(); i++)
    {
        ORB_SLAM3::MapGaussian* pMG = vpMG[i];
        if(pMG)
        {
            mMeans3D.index_put_({(int)i, "..."},  pMG->GetWorldPos());
            mOpacity.index_put_({(int)i, "..."},  pMG->GetOpacity());
            mScales.index_put_({(int)i, "..."},   pMG->GetScale());
            mRotation.index_put_({(int)i, "..."}, pMG->GetRotation());
            Features.index_put_({(int)i, "..."}, pMG->GetFeature());
        }
    }

    mFeaturesDC = Features.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, 1)}).transpose(1, 2).contiguous().to(torch::kCUDA);
    mFeaturesRest = Features.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None)}).transpose(1, 2).contiguous().to(torch::kCUDA);
    mFeatures = torch::cat({mFeaturesDC, mFeaturesRest}, 1).to(torch::kCUDA);

    mMeans3D.set_requires_grad(true);
    mMeans2D.set_requires_grad(true);
    mOpacity.set_requires_grad(true);
    mScales.set_requires_grad(true);
    mRotation.set_requires_grad(true);
    mFeatures.set_requires_grad(true);

    std::cout << "[GaussianSplatting::Optimize] mFeatures: " << mFeatures << std::endl;
    
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

        mTrainedImages.push_back(pKF->mIm);
        torch::Tensor TrainedImageTensor = CVMatToTensor(pKF->mIm);
        mTrainedImagesTensor.push_back(TrainedImageTensor.to(torch::kCUDA)); // 1.0 / 225.

        mViewMatrices.push_back(ViewMatrix.to(torch::kCUDA));
        mProjMatrices.push_back(FullProjMatrix.to(torch::kCUDA));
        mCameraCenters.push_back(CamCenter.to(torch::kCUDA));

        std::cout << "[GaussianSplatting::Optimize] ViewMatrix: " << i << ", " << mViewMatrices[i] << std::endl;
        std::cout << "[GaussianSplatting::Optimize] ProjMatrix: " << i << ", " << mProjMatrices[i] << std::endl;
    }

    // Setup Optimizer
    TrainingSetup();
    // Setup Loss Monitor
    mLossMonitor = new GaussianSplatting::LossMonitor(200);
    mSSIMWindow = CreateWindow().to(torch::kFloat32).to(torch::kCUDA, true);
}

void GaussianOptimizer::Optimize()
{
    std::cout << "[GaussianOptimizer::Optimize] Start" << std::endl;

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

        // Render
        GaussianRasterizer rasterizer = GaussianRasterizer(raster_settings);
        torch::cuda::synchronize();
        auto [rendererd_image, radii] = rasterizer.forward(
            mMeans3D,
            mMeans2D,
            mOpacity,
            mFeatures,
            mColorsPrecomp,
            mScales,
            mRotation,
            mCov3DPrecomp);

        // Loss Computations
        auto L1loss =L1Loss(GTImg, rendererd_image);
        auto SSIMloss = SSIM(rendererd_image, GTImg);
        auto loss = (1.f - mOptimParams.lambda_dssim) * L1loss + mOptimParams.lambda_dssim * (1.f - SSIMloss);
        loss.backward();
        std::cout << "[GaussianSplatting::Optimize] Loss: " << loss << std::endl;

        { 
            torch::NoGradGuard no_grad;

            // Densification
            if (iter < mOptimParams.densify_until_iter) {
                torch::Tensor VisibilityFilter = radii > 0;
                // std::cout << "[GaussianSplatting::Optimize] mMeans3D.grad(): " << mMeans3D.grad() << std::endl;
                // AddDensificationStats(mMeans2D, VisibilityFilter);
                if (iter > mOptimParams.densify_from_iter && iter % mOptimParams.densification_interval == 0) {
                    float size_threshold = iter > mOptimParams.opacity_reset_interval ? 20.f : -1.f;
                    // DensifyAndPrune(mOptimParams.densify_grad_threshold, mOptimParams.min_opacity, scene.Get_cameras_extent(), size_threshold);
                }

                // if (iter % mOptimParams.opacity_reset_interval == 0 || (modelParams.white_background && iter == optimParams.densify_from_iter)) {
                //     gaussians.Reset_opacity();
                // }
            }

            //  Optimizer step
            if (iter < mOptimParams.iterations) {
                mOptimizer->step();
                mOptimizer->zero_grad(true);
                UpdateLR(iter);
            }

            // Clear cache
            // if (mOptimParams.empty_gpu_cache && iter % 100) {
            //     c10::cuda::CUDACachingAllocator::emptyCache();
            // }
        }
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
    tensor = tensor.squeeze().detach().permute({1, 2, 0});
    tensor = tensor.mul(255).clamp(0, 255).to(torch::kU8);
    tensor = tensor.to(torch::kCPU);
    int64_t height = tensor.size(0);
    int64_t width = tensor.size(1);
    cv::Mat mat = cv::Mat(height, width, CV_8UC3, tensor.data_ptr());
    return mat.clone();
}

torch::Tensor GaussianOptimizer::CVMatToTensor(cv::Mat mat)
{
    cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);
    cv::Mat matFloat;
    mat.convertTo(matFloat, CV_32F, 1.0 / 255);
    auto size = matFloat.size();
    auto nChannels = matFloat.channels();
    auto tensor = torch::from_blob(matFloat.data, {size.height, size.width, nChannels});
    return tensor.permute({2, 0, 1});
}

void GaussianOptimizer::TrainingSetup()
{
    mPercentDense = mOptimParams.percent_dense;
    mPosGradientAccum = torch::zeros({mMeans3D.size(0), 1});
    mDenom = torch::zeros({mMeans3D.size(0), 1});
    mPosSchedulerArgs = GaussianSplatting::Expon_lr_func(mOptimParams.position_lr_init * mSpatialLRScale,
                                                         mOptimParams.position_lr_final * mSpatialLRScale,
                                                         mOptimParams.position_lr_delay_mult,
                                                         mOptimParams.position_lr_max_steps  );

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

    mOptimizer = std::make_unique<torch::optim::Adam>(OptimizerParamsGroups, torch::optim::AdamOptions(0.f).eps(1e-15));
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

}