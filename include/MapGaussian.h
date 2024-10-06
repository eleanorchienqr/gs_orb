/**
* This file is part of ORB-SLAM3
*/

#ifndef MAPGAUSSIAN_H
#define MAPGAUSSIAN_H

#include "KeyFrame.h"
#include "Frame.h"
#include "Map.h"

#include "SerializationUtils.h"
#include <vector>
#include <torch/torch.h>

#define ChildNum 10

namespace ORB_SLAM3 {

class KeyFrame;
class Map;
class Frame;

class MapGaussian
{
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & mnId;
        ar & mnFirstKFid;
        ar & mnFirstFrame;
        ar & nObs;

        // Protected variables
        // ar & boost::serialization::make_array(mWorldPos.data(), mWorldPos.size());
        // ar & boost::serialization::make_array(mWorldRot.data(), mWorldRot.size());
        // ar & boost::serialization::make_array(mScale.data(), mScale.size());
        // ar & boost::serialization::make_array(mFeatureDC.data(), mFeatureDC.size());
        
        //ar & BOOST_SERIALIZATION_NVP(mBackupObservationsId);
        //ar & mObservations;
        // ar & mOpacity;
        ar & mBackupObservationsId1;
        ar & mBackupObservationsId2;
        ar & mBackupRefKFId;
        ar & mnVisible;
        ar & mnFound;

        ar & mbBad;
        ar & mBackupReplacedId;

        ar & mfMinDistance;
        ar & mfMaxDistance;
    }

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        MapGaussian();

        // Initialization from associated MapPoints
        MapGaussian(const Eigen::Vector3f &Pos, KeyFrame* pRefKF, Map* pMap); 

        //utils
        inline torch::Tensor inverse_sigmoid(torch::Tensor x) {
            return torch::log(x / (1 - x));
        }

        
        //Setters
        void SetWorldPos(const Eigen::Vector3f &Pos);
        void SetScale(const torch::Tensor &Scale);
        // void SetWorldPos(const Eigen::Vector3f &Pos);
        
        // Getters
        torch::Tensor GetWorldPos();
        torch::Tensor GetOpacity();
        torch::Tensor GetScale();
        torch::Tensor GetRotation();
        torch::Tensor GetFeature();
        // void SetCov(const Eigen::Vector4f &Rot, const Eigen::Vector3f &Scale);
        // Eigen::Matrix3d GetCov();

        // void SetSHParam(const Eigen::Vector3f &FeatureDC);
        // Eigen::Vector3f GetSHParam();

        // KeyFrame* GetReferenceKeyFrame();

        // std::map<KeyFrame*,std::tuple<int,int>> GetObservations();
        // int Observations();

        // void AddObservation(KeyFrame* pKF,int idx);
        // void EraseObservation(KeyFrame* pKF);

        // std::tuple<int,int> GetIndexInKeyFrame(KeyFrame* pKF);
        // bool IsInKeyFrame(KeyFrame* pKF);

        // void SetBadFlag();
        // bool isBad();

        // void Replace(MapGaussian* pMG);    
        // MapGaussian* GetReplaced();

        // void IncreaseVisible(int n=1);
        // void IncreaseFound(int n=1);
        // float GetFoundRatio();
        inline int GetFound(){
            return mnFound;
        }

        // void UpdateNormalAndDepth();

        // float GetMinDistanceInvariance();
        // float GetMaxDistanceInvariance();
        // int PredictScale(const float &currentDist, KeyFrame*pKF);
        // int PredictScale(const float &currentDist, Frame* pF);

        Map* GetMap();
        // void UpdateMap(Map* pMap);

        // void PrintObservations();

        // void PreSave(set<KeyFrame*>& spKF,set<MapGaussian*>& spMG);
        // void PostLoad(map<long unsigned int, KeyFrame*>& mpKFid, map<long unsigned int, MapGaussian*>& mpMGid);

    public:
        long unsigned int mnId;
        static long unsigned int nNextId;
        long int mnFirstKFid;
        long int mnFirstFrame;
        int nObs;

        // Variables used by the tracking
        float mTrackProjX;
        float mTrackProjY;
        float mTrackDepth;
        float mTrackDepthR;
        float mTrackProjXR;
        float mTrackProjYR;
        bool mbTrackInView, mbTrackInViewR;
        int mnTrackScaleLevel, mnTrackScaleLevelR;
        float mTrackViewCos, mTrackViewCosR;
        long unsigned int mnTrackReferenceForFrame;
        long unsigned int mnLastFrameSeen;

         // Variables used by local mapping
        long unsigned int mnBALocalForKF;
        long unsigned int mnFuseCandidateForKF;

        // Variables used by loop closing
        long unsigned int mnLoopGaussianForKF;
        long unsigned int mnCorrectedByKF;
        long unsigned int mnCorrectedReference;    
        // Eigen::Vector3f mPosGBA;
        long unsigned int mnBAGlobalForKF;
        long unsigned int mnBALocalForMerge;

        // Variable used by merging
        // Eigen::Vector3f mPosMerge;
        // Eigen::Vector3f mNormalVectorMerge;

        // For inverse depth optimization
        double mInvDepth;
        double mInitU;
        double mInitV;
        KeyFrame* mpHostKF;

        static std::mutex mGlobalMutex;

        unsigned int mnOriginMapId;

    
    protected:
        // Gaussian representation
        // Eigen::Vector3f mWorldPos;
        // Eigen::Vector4f mWorldRot;
        // Eigen::Vector3f mScale;
        // float mOpacity;
        // long unsigned int mSHDegree = 10;
        // Eigen::Vector3f mFeatureDC;
        // Eigen::MatrixXf mFeaturest;
        long unsigned int mSHDegree = 10;

        torch::Tensor mWorldPos;            // {1, 3}
        torch::Tensor mWorldRot;            // {1, 4}
        torch::Tensor mScale;               // {1, 3}
        torch::Tensor mOpacity;             // {1, 1}
        torch::Tensor mFeatureDC;           // {1, 3}
        torch::Tensor mFeaturest;           // {1, (mSHDegree+1)**2 - 1}

        // For save relation without pointer, this is necessary for save/load function
        std::map<long unsigned int, int> mBackupObservationsId1;
        std::map<long unsigned int, int> mBackupObservationsId2;

        // Reference KeyFrame
        KeyFrame* mpRefKF;
        long unsigned int mBackupRefKFId;

        // Tracking counters
        int mnVisible;
        int mnFound;

        // Bad flag (we do not currently erase MapGaussian from memory)
        bool mbBad;
        MapGaussian* mpReplaced;
        // For save relation without pointer, this is necessary for save/load function
        long long int mBackupReplacedId;

        // Scale invariance distances
        float mfMinDistance;
        float mfMaxDistance;

        Map* mpMap;

        // Mutex
        std::mutex mMutexParam;
        std::mutex mMutexPos;
        std::mutex mMutexScale;
        std::mutex mMutexFeatures;
        std::mutex mMutexMap;

};

struct MapGaussianNode{

    MapGaussian* data;
    std::vector<MapGaussianNode*> children;

    MapGaussianNode(MapGaussian* newData):data(newData){
        children = std::vector<MapGaussianNode*>(ChildNum,static_cast<MapGaussianNode*>(NULL)); 
    }

    MapGaussianNode* AddChild(MapGaussian* newData){
        MapGaussianNode*  newNode = new MapGaussianNode(newData);
        for(int i = 0; i<ChildNum; i++){
            if(children[i] == NULL){
                children[i] = newNode;
                break;
            }
        }
        return newNode;
    }
};

class MapGaussianTree
{
    private:
        MapGaussianNode* root;
    
    public:
        MapGaussianTree():root(NULL){}
        MapGaussianTree(MapGaussian* data):root(new MapGaussianNode(data)){}
        ~MapGaussianTree(){
            delete root;
        }
        
        MapGaussianNode* GetRoot(){
            return root;
        }
        // std::vector<MapGaussian*> GetAllMapGaussians();
};

}//namespace ORB_SLAM

#endif // MAPGAUSSIAN_H

