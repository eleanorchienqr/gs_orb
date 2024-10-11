#include "GaussianOptimizer.h"
#include "Renderer/Rasterizer.h"

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
    mMeans3D = torch::zeros({mSizeofGaussians, 3});
    mOpacity = torch::zeros({mSizeofGaussians, 1});
    mScales = torch::zeros({mSizeofGaussians, 3});
    mRotation = torch::zeros({mSizeofGaussians, 4});
    mMeans2D = torch::zeros({mSizeofGaussians, 2});
    mFeatures = torch::zeros({mSizeofGaussians, 3, static_cast<long>(std::pow((mSHDegree + 1), 2))});

    for(size_t i=0; i<vpMG.size(); i++)
    {
        ORB_SLAM3::MapGaussian* pMG = vpMG[i];
        if(pMG)
        {
            mMeans3D.index_put_({(int)i, "..."},  pMG->GetWorldPos());
            mOpacity.index_put_({(int)i, "..."},  pMG->GetOpacity());
            mScales.index_put_({(int)i, "..."},   pMG->GetScale());
            mRotation.index_put_({(int)i, "..."}, pMG->GetRotation());
            mFeatures.index_put_({(int)i, "..."}, pMG->GetFeature());
        }
    }

    mFeaturesDC = mFeatures.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, 1)}).transpose(1, 2);
    mFeaturesRest = mFeatures.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None)}).transpose(1, 2);

    // Get Camera and Image Data
    mSizeofCameras = vpKFs.size();
    vpKFs[0]->GetGaussianRenderParams(mImHeight, mImWidth, mTanFovx, mTanFovy);
    SetProjMatrix();
    mTrainedImages.resize(mSizeofCameras);
    mTrainedImagesTensor.resize(mSizeofCameras);
    mViewMatrices.resize(mSizeofCameras);
    mProjMatrices.resize(mSizeofCameras);
    mCameraCenters.resize(mSizeofCameras);

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
        mTrainedImagesTensor.push_back(CVMatToTensor(pKF->mIm)); // 1.0 / 225.

        mViewMatrices.push_back(ViewMatrix);
        mProjMatrices.push_back(FullProjMatrix);
        mCameraCenters.push_back(CamCenter);
    }

    // Setup Optimizer
    TrainingSetup();
    // Setup Loss Monitor
    mLossMonitor = new GaussianSplatting::LossMonitor(200);
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
        // auto [rendererd_image, radii] = rasterizer.forward(
        //     mMeans3D,
        //     mMeans2D,
        //     mOpacity,
        //     mFeatures,
        //     mColorsPrecomp,
        //     mScales,
        //     mRotation,
        //     mCov3DPrecomp);

        // Loss Computations
        // auto l1l = gaussian_splatting::l1_loss(image, gt_image);
        // auto ssim_loss = gaussian_splatting::ssim(image, gt_image, conv_window, window_size, channel);
        // auto loss = (1.f - optimParams.lambda_dssim) * l1l + optimParams.lambda_dssim * (1.f - ssim_loss);
    }
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

void GaussianOptimizer::SetProjMatrix()
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
    mProjMatrix = torch::from_blob(P.data(), {4, 4}, torch::kFloat);
}


}