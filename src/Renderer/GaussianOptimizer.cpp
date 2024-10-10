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
    mFeatures = torch::zeros({mSizeofGaussians, 3, static_cast<long>(std::pow((10 + 1), 2))});

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
    // std::vector<cv::Mat> mTrainedImages;
    // std::vector<torch::Tensor> mViewmatrices;
    // std::vector<torch::Tensor> mProjmatrices;
    // std::vector<torch::Tensor> mVameraCenters;

    for(size_t i=0; i<vpKFs.size(); i++)
    {
        ORB_SLAM3::KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;


    }



}


}