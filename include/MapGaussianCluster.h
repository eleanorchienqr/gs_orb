/**
* This file is part of ORB-SLAM3
*/

#ifndef MAPGAUSSIANCLUSTER_H
#define MAPGAUSSIANCLUSTER_H

#include "MapGaussian.h"
#include "KeyFrame.h"
#include "Frame.h"
#include "Map.h"

#include "SerializationUtils.h"
#include <torch/torch.h>


#define ChildNum 10

namespace GaussianSplatting 
{

class MapGaussianCluster
{

    friend class boost::serialization::access;
    template<class Archive> 
    void serialize(Archive & ar, const unsigned int version)
    {

    }

    public:
        // Initialization from associated MapPoints
        // MapGaussianNode(const Eigen::Vector3f &Pos, ORB_SLAM3::KeyFrame* pRefKF, ORB_SLAM3::Map* pMap); 

        // ORB_SLAM3::Map* GetMap();
    
    public:
        // long unsigned int mnId;
        // static long unsigned int nNextId;
        // long int mnFirstKFid;
        // long int mnFirstFrame;
        // int nObs;

        // // Variables used by the tracking
        // float mTrackProjX;
        // float mTrackProjY;
        // float mTrackDepth;
        // float mTrackDepthR;
        // float mTrackProjXR;
        // float mTrackProjYR;
        // bool mbTrackInView, mbTrackInViewR;
        // int mnTrackScaleLevel, mnTrackScaleLevelR;
        // float mTrackViewCos, mTrackViewCosR;
        // long unsigned int mnTrackReferenceForFrame;
        // long unsigned int mnLastFrameSeen;

        //  // Variables used by local mapping
        // long unsigned int mnBALocalForKF;
        // long unsigned int mnFuseCandidateForKF;

        // ORB_SLAM3::KeyFrame* mpHostKF;

        // static std::mutex mGlobalMutex;

        // unsigned int mnOriginMapId;

    protected:

        long unsigned int mChildrenNum = 0;
        long unsigned int mSHDegree = 3;

        torch::Tensor mWorldPos;            // {1 + mChildrenNum, 3}
        torch::Tensor mWorldRot;            // {1 + mChildrenNum, 4}
        torch::Tensor mScale;               // {1 + mChildrenNum, 3}
        torch::Tensor mOpacity;             // {1 + mChildrenNum, 1}
        torch::Tensor mFeatureDC;           // {1 + mChildrenNum, 3}
        torch::Tensor mFeaturest;           // {1 + mChildrenNum, (mSHDegree+1)**2 - 1}

        // Reference KeyFrame
        ORB_SLAM3::KeyFrame* mpRefKF;

        // Tracking counters
        // int mnVisible;
        // int mnFound;

        // Bad flag (we do not currently erase MapGaussian from memory)
        // bool mbBad;

        ORB_SLAM3::Map* mpMap;

        // Mutex
        // std::mutex mMutexParam;
};


}//namespace GAUSSIANSPLATTING

#endif // MAPGAUSSIAN_H

