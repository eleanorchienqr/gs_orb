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

MapGaussian::MapGaussian(const long unsigned int SHDegree):
    mSHDegree(mSHDegree), mnFirstKFid(0), mnFirstFrame(0), nObs(0), mnTrackReferenceForFrame(0),
    mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopGaussianForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mnVisible(1), mnFound(1), mbBad(false),
    mpReplaced(static_cast<MapGaussian*>(NULL))
{
    mpReplaced = static_cast<MapGaussian*>(NULL);
}

MapGaussian::MapGaussian(const Eigen::Vector3f &Pos, KeyFrame *pRefKF, Map* pMap):
    mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), nObs(0), mnTrackReferenceForFrame(0),
    mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopGaussianForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(pRefKF), mnVisible(1), mnFound(1), mbBad(false),
    mpReplaced(static_cast<MapGaussian*>(NULL)), mfMinDistance(0), mfMaxDistance(0), mpMap(pMap),
    mnOriginMapId(pMap->GetId())
{
    // std::cout << "-------------Create single MapGaussian------------" << std::endl;
    mWorldPos = Pos;
    mWorldRot.setZero();
    mOpacity = 0.0;
    mFeatureDC << 1.0, 1.0, 1.0;

    const int FeaturestDim = std::pow(mSHDegree + 1, 2) - 1;
    mFeaturest.setZero(FeaturestDim, 1);

    mbTrackInViewR = false;
    mbTrackInView = false;

    // MapGaussians can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexGaussianCreation);
    mnId=nNextId++;
}

MapGaussian::MapGaussian(const Eigen::Vector3f &Pos, const Eigen::Vector4f &Rot, const Eigen::Vector3f &Scale, 
                         const float &Opacity,const Eigen::Vector3f &FeatureDC, KeyFrame *pRefKF, Map* pMap):
    mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), nObs(0), mnTrackReferenceForFrame(0),
    mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopGaussianForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(pRefKF), mnVisible(1), mnFound(1), mbBad(false),
    mpReplaced(static_cast<MapGaussian*>(NULL)), mfMinDistance(0), mfMaxDistance(0), mpMap(pMap),
    mnOriginMapId(pMap->GetId())
{
    SetGaussianParam(Pos, Rot, Scale, Opacity, FeatureDC);

    mbTrackInViewR = false;
    mbTrackInView = false;

    // MapGaussians can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexGaussianCreation);
    mnId=nNextId++;
}

void MapGaussian::SetGaussianParam(const Eigen::Vector3f &Pos, const Eigen::Vector4f &Rot, const Eigen::Vector3f &Scale, 
                                   const float &Opacity,const Eigen::Vector3f &FeatureDC) {
    unique_lock<mutex> lock2(mGlobalMutex);
    unique_lock<mutex> lock(mMutexParam);
    mWorldPos = Pos;
    mWorldRot = Rot;
    mScale = Scale;
    mOpacity = Opacity;
    mFeatureDC = FeatureDC;
}

Eigen::Vector3f MapGaussian::GetWorldPos()
{
    unique_lock<mutex> lock(mMutexPos);
    return mWorldPos;
}

Map* MapGaussian::GetMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mpMap;
}


} //namespace ORB_SLAM