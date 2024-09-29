/**
* This file is part of ORB-SLAM3
*/

#include "MapGaussian.h"

#include <cmath>
#include<mutex>

namespace ORB_SLAM3
{

long unsigned int MapGaussian::nNextId=0;
mutex MapGaussian::mGlobalMutex;

MapGaussian::MapGaussian():
    mnFirstKFid(0), mnFirstFrame(0), nObs(0), mnTrackReferenceForFrame(0),
    mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopGaussianForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mnVisible(1), mnFound(1), mbBad(false),
    mpReplaced(static_cast<MapGaussian*>(NULL))
{
    mpReplaced = static_cast<MapGaussian*>(NULL);
}

// MapGaussian::MapGaussian(const long unsigned int SHDegree):
//     mSHDegree(mSHDegree), mnFirstKFid(0), mnFirstFrame(0), nObs(0), mnTrackReferenceForFrame(0),
//     mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopGaussianForKF(0), mnCorrectedByKF(0),
//     mnCorrectedReference(0), mnBAGlobalForKF(0), mnVisible(1), mnFound(1), mbBad(false),
//     mpReplaced(static_cast<MapGaussian*>(NULL))
// {
//     mpReplaced = static_cast<MapGaussian*>(NULL);
// }

MapGaussian::MapGaussian(const Eigen::Vector3f &Pos, KeyFrame *pRefKF, Map* pMap):
    mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), nObs(0), mnTrackReferenceForFrame(0),
    mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopGaussianForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(pRefKF), mnVisible(1), mnFound(1), mbBad(false),
    mpReplaced(static_cast<MapGaussian*>(NULL)), mfMinDistance(0), mfMaxDistance(0), mpMap(pMap),
    mnOriginMapId(pMap->GetId())
{
    // std::cout << "-------------Create single MapGaussian------------" << std::endl;

    const int FeaturestDim = std::pow(mSHDegree + 1, 2) - 1;
    const auto pointType = torch::TensorOptions().dtype(torch::kFloat32);

    Eigen::Vector3f PosTranspose = Pos.transpose();
    mWorldPos = torch::from_blob(PosTranspose.data(), {1, 3}, pointType).to(torch::kCUDA);
    mWorldRot = torch::zeros({1, 4}).to(torch::kCUDA, true);
    mOpacity = inverse_sigmoid(0.5 * torch::ones({1, 1})).to(torch::kCUDA, true);
    mFeatureDC = torch::ones({1, 3}).to(torch::kCUDA, true);
    mFeaturest = torch::zeros({1, FeaturestDim}).to(torch::kCUDA, true);

    // std::cout << "WorldPos of MapPoint: " << PosTranspose << std::endl;
    // std::cout << "WorldPos of Gaussian: " << mWorldPos << std::endl;
    // std::cout << "mOpacity of Gaussian: " << mOpacity << std::endl;
    // std::cout << std:: endl;

    mbTrackInViewR = false;
    mbTrackInView = false;

    // MapGaussians can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexGaussianCreation);
    mnId=nNextId++;
}

// MapGaussian::MapGaussian(const Eigen::Vector3f &Pos, const Eigen::Vector4f &Rot, const Eigen::Vector3f &Scale, 
//                          const float &Opacity,const Eigen::Vector3f &FeatureDC, KeyFrame *pRefKF, Map* pMap):
//     mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), nObs(0), mnTrackReferenceForFrame(0),
//     mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopGaussianForKF(0), mnCorrectedByKF(0),
//     mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(pRefKF), mnVisible(1), mnFound(1), mbBad(false),
//     mpReplaced(static_cast<MapGaussian*>(NULL)), mfMinDistance(0), mfMaxDistance(0), mpMap(pMap),
//     mnOriginMapId(pMap->GetId())
// {
//     SetGaussianParam(Pos, Rot, Scale, Opacity, FeatureDC);

//     mbTrackInViewR = false;
//     mbTrackInView = false;

//     // MapGaussians can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
//     unique_lock<mutex> lock(mpMap->mMutexGaussianCreation);
//     mnId=nNextId++;
// }

// void MapGaussian::SetGaussianParam(const Eigen::Vector3f &Pos, const Eigen::Vector4f &Rot, const Eigen::Vector3f &Scale, 
//                                    const float &Opacity,const Eigen::Vector3f &FeatureDC) {
//     unique_lock<mutex> lock2(mGlobalMutex);
//     unique_lock<mutex> lock(mMutexParam);
//     mWorldPos = Pos;
//     mWorldRot = Rot;
//     mScale = Scale;
//     mOpacity = Opacity;
//     mFeatureDC = FeatureDC;
// }

torch::Tensor MapGaussian::GetWorldPos()
{
    unique_lock<mutex> lock(mMutexPos);
    return mWorldPos;
}

Map* MapGaussian::GetMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mpMap;
}

void MapGaussian::SetScale(const torch::Tensor &Scale)
{
    unique_lock<mutex> lock2(mGlobalMutex);
    unique_lock<mutex> lock(mMutexScale);
    mScale = Scale;
}


} //namespace ORB_SLAM