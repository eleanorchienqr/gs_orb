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
        ar & boost::serialization::make_array(mWorldPos.data(), mWorldPos.size());
        ar & boost::serialization::make_array(mWorldRot.data(), mWorldRot.size());
        ar & boost::serialization::make_array(mScale.data(), mScale.size());
        ar & boost::serialization::make_array(mFeatureDC.data(), mFeatureDC.size());
        
        //ar & BOOST_SERIALIZATION_NVP(mBackupObservationsId);
        //ar & mObservations;
        ar & mOpacity;
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
        MapGaussian(const Eigen::Vector3f &Pos, const Eigen::Vector4f &Rot, const Eigen::Vector3f &Scale, 
                    const float &Opacity,const Eigen::Vector3f &FeatureDC, KeyFrame* pRefKF, Map* pMap);

        // MapPoint(const double invDepth, cv::Point2f uv_init, KeyFrame* pRefKF, KeyFrame* pHostKF, Map* pMap);
        // MapPoint(const Eigen::Vector3f &Pos,  Map* pMap, Frame* pFrame, const int &idxF);

        void SetGaussianParam(const Eigen::Vector3f &Pos, const Eigen::Vector4f &Rot, const Eigen::Vector3f &Scale, 
                              const float &Opacity,const Eigen::Vector3f &FeatureDC);
        // void SetWorldPos(const Eigen::Vector3f &Pos);
        Eigen::Vector3f GetWorldPos();

        // void SetCov(const Eigen::Vector4f &Rot, const Eigen::Vector3f &Scale);
        // Eigen::Matrix3d GetCov();

        // void SetOpacity(const float &Opacity);
        // float GetOpacity();

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
        Eigen::Vector3f mWorldPos;
        Eigen::Vector4f mWorldRot;
        Eigen::Vector3f mScale;
        float mOpacity;
        long unsigned int mSHDegree = 10;
        Eigen::Vector3f mFeatureDC;
        // Eigen::Matrix3Xf mFeaturest;

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
        std::vector<MapGaussian*> GetAllGaussians();
};

}//namespace ORB_SLAM

#endif // MAPGAUSSIAN_H

