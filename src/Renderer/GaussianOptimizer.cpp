#include "GaussianOptimizer.h"

namespace GaussianSplatting{

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

    // Get Camera and Image Data
    mSizeofCameras = vpKFs.size();
    vpKFs[0]->GetGaussianRenderParams(mImHeight, mImWidth, mTanFovx, mTanFovy);
    SetProjMatrix();
    mTrainedImages.resize(mSizeofCameras);
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
        mViewMatrices.push_back(ViewMatrix);
        mProjMatrices.push_back(FullProjMatrix);
        mCameraCenters.push_back(CamCenter);
    }

    // Setup Optimizer
    TrainingSetup();

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

    // std::vector<torch::optim::OptimizerParamGroup> optimizer_params_groups;
    // optimizer_params_groups.reserve(6);
    // optimizer_params_groups.push_back(torch::optim::OptimizerParamGroup({_xyz}, std::make_unique<torch::optim::AdamOptions>(params.position_lr_init * this->_spatial_lr_scale)));
    // optimizer_params_groups.push_back(torch::optim::OptimizerParamGroup({_features_dc}, std::make_unique<torch::optim::AdamOptions>(params.feature_lr)));
    // optimizer_params_groups.push_back(torch::optim::OptimizerParamGroup({_features_rest}, std::make_unique<torch::optim::AdamOptions>(params.feature_lr / 20.)));
    // optimizer_params_groups.push_back(torch::optim::OptimizerParamGroup({_scaling}, std::make_unique<torch::optim::AdamOptions>(params.scaling_lr * this->_spatial_lr_scale)));
    // optimizer_params_groups.push_back(torch::optim::OptimizerParamGroup({_rotation}, std::make_unique<torch::optim::AdamOptions>(params.rotation_lr)));
    // optimizer_params_groups.push_back(torch::optim::OptimizerParamGroup({_opacity}, std::make_unique<torch::optim::AdamOptions>(params.opacity_lr)));

    // static_cast<torch::optim::AdamOptions&>(optimizer_params_groups[0].options()).eps(1e-15);
    // static_cast<torch::optim::AdamOptions&>(optimizer_params_groups[1].options()).eps(1e-15);
    // static_cast<torch::optim::AdamOptions&>(optimizer_params_groups[2].options()).eps(1e-15);
    // static_cast<torch::optim::AdamOptions&>(optimizer_params_groups[3].options()).eps(1e-15);
    // static_cast<torch::optim::AdamOptions&>(optimizer_params_groups[4].options()).eps(1e-15);
    // static_cast<torch::optim::AdamOptions&>(optimizer_params_groups[5].options()).eps(1e-15);

    // _optimizer = std::make_unique<torch::optim::Adam>(optimizer_params_groups, torch::optim::AdamOptions(0.f).eps(1e-15));
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