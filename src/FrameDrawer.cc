/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include "FrameDrawer.h"
#include "Tracking.h"
#include "Renderer/Rasterizer.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include<mutex>

namespace ORB_SLAM3
{

FrameDrawer::FrameDrawer(Atlas* pAtlas):both(false),mpAtlas(pAtlas)
{
    mState=Tracking::SYSTEM_NOT_READY;
    mImOrigin = cv::Mat(480,640,CV_8UC3, cv::Scalar(0,0,0));
    mIm = cv::Mat(480,640,CV_8UC3, cv::Scalar(0,0,0));
    mImRight = cv::Mat(480,640,CV_8UC3, cv::Scalar(0,0,0));
}

cv::Mat FrameDrawer::DrawFrame(float imageScale)
{
    cv::Mat im;
    vector<cv::KeyPoint> vIniKeys; // Initialization: KeyPoints in reference frame
    vector<int> vMatches; // Initialization: correspondeces with reference keypoints
    vector<cv::KeyPoint> vCurrentKeys; // KeyPoints in current frame
    vector<bool> vbVO, vbMap; // Tracked MapPoints in current frame
    vector<pair<cv::Point2f, cv::Point2f> > vTracks;
    int state; // Tracking state
    vector<float> vCurrentDepth;
    float thDepth;

    Frame currentFrame;
    vector<MapPoint*> vpLocalMap;
    vector<cv::KeyPoint> vMatchesKeys;
    vector<MapPoint*> vpMatchedMPs;
    vector<cv::KeyPoint> vOutlierKeys;
    vector<MapPoint*> vpOutlierMPs;
    map<long unsigned int, cv::Point2f> mProjectPoints;
    map<long unsigned int, cv::Point2f> mMatchedInImage;

    cv::Scalar standardColor(0,255,0);
    cv::Scalar odometryColor(255,0,0);

    //Copy variables within scoped mutex
    {
        unique_lock<mutex> lock(mMutex);
        state=mState;
        if(mState==Tracking::SYSTEM_NOT_READY)
            mState=Tracking::NO_IMAGES_YET;

        mIm.copyTo(im);

        if(mState==Tracking::NOT_INITIALIZED)
        {
            vCurrentKeys = mvCurrentKeys;
            vIniKeys = mvIniKeys;
            vMatches = mvIniMatches;
            vTracks = mvTracks;
        }
        else if(mState==Tracking::OK)
        {
            vCurrentKeys = mvCurrentKeys;
            vbVO = mvbVO;
            vbMap = mvbMap;

            currentFrame = mCurrentFrame;
            vpLocalMap = mvpLocalMap;
            vMatchesKeys = mvMatchedKeys;
            vpMatchedMPs = mvpMatchedMPs;
            vOutlierKeys = mvOutlierKeys;
            vpOutlierMPs = mvpOutlierMPs;
            mProjectPoints = mmProjectPoints;
            mMatchedInImage = mmMatchedInImage;

            vCurrentDepth = mvCurrentDepth;
            thDepth = mThDepth;

        }
        else if(mState==Tracking::LOST)
        {
            vCurrentKeys = mvCurrentKeys;
        }
    }

    if(imageScale != 1.f)
    {
        int imWidth = im.cols / imageScale;
        int imHeight = im.rows / imageScale;
        cv::resize(im, im, cv::Size(imWidth, imHeight));
    }

    if(im.channels()<3) //this should be always true
        cvtColor(im,im,cv::COLOR_GRAY2BGR);

    //Draw
    if(state==Tracking::NOT_INITIALIZED)
    {
        for(unsigned int i=0; i<vMatches.size(); i++)
        {
            if(vMatches[i]>=0)
            {
                cv::Point2f pt1,pt2;
                if(imageScale != 1.f)
                {
                    pt1 = vIniKeys[i].pt / imageScale;
                    pt2 = vCurrentKeys[vMatches[i]].pt / imageScale;
                }
                else
                {
                    pt1 = vIniKeys[i].pt;
                    pt2 = vCurrentKeys[vMatches[i]].pt;
                }
                cv::line(im,pt1,pt2,standardColor);
            }
        }
        for(vector<pair<cv::Point2f, cv::Point2f> >::iterator it=vTracks.begin(); it!=vTracks.end(); it++)
        {
            cv::Point2f pt1,pt2;
            if(imageScale != 1.f)
            {
                pt1 = (*it).first / imageScale;
                pt2 = (*it).second / imageScale;
            }
            else
            {
                pt1 = (*it).first;
                pt2 = (*it).second;
            }
            cv::line(im,pt1,pt2, standardColor,5);
        }

    }
    else if(state==Tracking::OK) //TRACKING
    {
        mnTracked=0;
        mnTrackedVO=0;
        const float r = 5;
        int n = vCurrentKeys.size();
        for(int i=0;i<n;i++)
        {
            if(vbVO[i] || vbMap[i])
            {
                cv::Point2f pt1,pt2;
                cv::Point2f point;
                if(imageScale != 1.f)
                {
                    point = vCurrentKeys[i].pt / imageScale;
                    float px = vCurrentKeys[i].pt.x / imageScale;
                    float py = vCurrentKeys[i].pt.y / imageScale;
                    pt1.x=px-r;
                    pt1.y=py-r;
                    pt2.x=px+r;
                    pt2.y=py+r;
                }
                else
                {
                    point = vCurrentKeys[i].pt;
                    pt1.x=vCurrentKeys[i].pt.x-r;
                    pt1.y=vCurrentKeys[i].pt.y-r;
                    pt2.x=vCurrentKeys[i].pt.x+r;
                    pt2.y=vCurrentKeys[i].pt.y+r;
                }

                // This is a match to a MapPoint in the map
                if(vbMap[i])
                {
                    cv::rectangle(im,pt1,pt2,standardColor);
                    cv::circle(im,point,2,standardColor,-1);
                    mnTracked++;
                }
                else // This is match to a "visual odometry" MapPoint created in the last frame
                {
                    cv::rectangle(im,pt1,pt2,odometryColor);
                    cv::circle(im,point,2,odometryColor,-1);
                    mnTrackedVO++;
                }
            }
        }
    }

    cv::Mat imWithInfo;
    DrawTextInfo(im,state, imWithInfo);

    return imWithInfo;
}

cv::Mat FrameDrawer::DrawRightFrame(float imageScale)
{
    cv::Mat im;
    vector<cv::KeyPoint> vIniKeys; // Initialization: KeyPoints in reference frame
    vector<int> vMatches; // Initialization: correspondeces with reference keypoints
    vector<cv::KeyPoint> vCurrentKeys; // KeyPoints in current frame
    vector<bool> vbVO, vbMap; // Tracked MapPoints in current frame
    int state; // Tracking state

    //Copy variables within scoped mutex
    {
        unique_lock<mutex> lock(mMutex);
        state=mState;
        if(mState==Tracking::SYSTEM_NOT_READY)
            mState=Tracking::NO_IMAGES_YET;

        mImRight.copyTo(im);

        if(mState==Tracking::NOT_INITIALIZED)
        {
            vCurrentKeys = mvCurrentKeysRight;
            vIniKeys = mvIniKeys;
            vMatches = mvIniMatches;
        }
        else if(mState==Tracking::OK)
        {
            vCurrentKeys = mvCurrentKeysRight;
            vbVO = mvbVO;
            vbMap = mvbMap;
        }
        else if(mState==Tracking::LOST)
        {
            vCurrentKeys = mvCurrentKeysRight;
        }
    } // destroy scoped mutex -> release mutex

    if(imageScale != 1.f)
    {
        int imWidth = im.cols / imageScale;
        int imHeight = im.rows / imageScale;
        cv::resize(im, im, cv::Size(imWidth, imHeight));
    }

    if(im.channels()<3) //this should be always true
        cvtColor(im,im,cv::COLOR_GRAY2BGR);

    //Draw
    if(state==Tracking::NOT_INITIALIZED) //INITIALIZING
    {
        for(unsigned int i=0; i<vMatches.size(); i++)
        {
            if(vMatches[i]>=0)
            {
                cv::Point2f pt1,pt2;
                if(imageScale != 1.f)
                {
                    pt1 = vIniKeys[i].pt / imageScale;
                    pt2 = vCurrentKeys[vMatches[i]].pt / imageScale;
                }
                else
                {
                    pt1 = vIniKeys[i].pt;
                    pt2 = vCurrentKeys[vMatches[i]].pt;
                }

                cv::line(im,pt1,pt2,cv::Scalar(0,255,0));
            }
        }
    }
    else if(state==Tracking::OK) //TRACKING
    {
        mnTracked=0;
        mnTrackedVO=0;
        const float r = 5;
        const int n = mvCurrentKeysRight.size();
        const int Nleft = mvCurrentKeys.size();

        for(int i=0;i<n;i++)
        {
            if(vbVO[i + Nleft] || vbMap[i + Nleft])
            {
                cv::Point2f pt1,pt2;
                cv::Point2f point;
                if(imageScale != 1.f)
                {
                    point = mvCurrentKeysRight[i].pt / imageScale;
                    float px = mvCurrentKeysRight[i].pt.x / imageScale;
                    float py = mvCurrentKeysRight[i].pt.y / imageScale;
                    pt1.x=px-r;
                    pt1.y=py-r;
                    pt2.x=px+r;
                    pt2.y=py+r;
                }
                else
                {
                    point = mvCurrentKeysRight[i].pt;
                    pt1.x=mvCurrentKeysRight[i].pt.x-r;
                    pt1.y=mvCurrentKeysRight[i].pt.y-r;
                    pt2.x=mvCurrentKeysRight[i].pt.x+r;
                    pt2.y=mvCurrentKeysRight[i].pt.y+r;
                }

                // This is a match to a MapPoint in the map
                if(vbMap[i + Nleft])
                {
                    cv::rectangle(im,pt1,pt2,cv::Scalar(0,255,0));
                    cv::circle(im,point,2,cv::Scalar(0,255,0),-1);
                    mnTracked++;
                }
                else // This is match to a "visual odometry" MapPoint created in the last frame
                {
                    cv::rectangle(im,pt1,pt2,cv::Scalar(255,0,0));
                    cv::circle(im,point,2,cv::Scalar(255,0,0),-1);
                    mnTrackedVO++;
                }
            }
        }
    }

    cv::Mat imWithInfo;
    DrawTextInfo(im,state, imWithInfo);

    return imWithInfo;
}

cv::Mat FrameDrawer::DrawGaussianFrame()
{
    cv::Mat im;
    cv::Mat CombinedImg;

    // Gaussian Info
    int GauSHDegree;
    torch::Tensor Means3D;
    torch::Tensor Opacity;
    torch::Tensor Scales;
    torch::Tensor Rotation;
    torch::Tensor FeaturesDC;
    torch::Tensor FeaturesRest;

    //Camera Info
    int ImHeight, ImWidth;
    float TanFovx, TanFovy;
    torch::Tensor ViewMatrix;
    torch::Tensor ProjMatrix;
    torch::Tensor CamCenter;

    int state = mState;

    //Copy variables within scoped mutex
    {
        unique_lock<mutex> lock(mMutex);

        if(mState==Tracking::OK)
        {
            GetGaussianRenderData(GauSHDegree, Means3D, Opacity, Scales, Rotation, FeaturesDC, FeaturesRest);
            GetCamParams(ImHeight, ImWidth, TanFovx, TanFovy, ViewMatrix, ProjMatrix, CamCenter);
        }

        mImOrigin.copyTo(im);
        CombinedImg = CombindImages(im, im);
    }

    //Draw
    if(state==Tracking::OK) //TRACKING
    {
        torch::NoGradGuard no_grad;

        GaussianRasterizationSettings raster_settings = {
            .image_height = static_cast<int>(ImHeight),
            .image_width = static_cast<int>(ImWidth),
            .tanfovx = TanFovx,
            .tanfovy = TanFovy,
            .bg = torch::tensor({1.f, 1.f, 1.f}).to(torch::kCUDA),
            .scale_modifier = 1.f,
            .viewmatrix = ViewMatrix,
            .projmatrix = ProjMatrix,
            .sh_degree = GauSHDegree,
            .camera_center = CamCenter,
            .prefiltered = false};

        torch::Tensor Means2D = torch::zeros_like(Means3D).to(torch::kCUDA);
        torch::Tensor Cov3DPrecomp = torch::Tensor();
        torch::Tensor ColorsPrecomp = torch::Tensor();

        // Render
        GaussianRasterizer rasterizer = GaussianRasterizer(raster_settings);
        torch::cuda::synchronize();
        auto [rendererd_image, radii] = rasterizer.forward(
            Means3D,
            Means2D,
            torch::sigmoid(Opacity).to(torch::kCUDA),
            torch::cat({FeaturesDC, FeaturesRest}, 1).to(torch::kCUDA),
            ColorsPrecomp,
            torch::exp(Scales).to(torch::kCUDA),
            torch::nn::functional::normalize(Rotation).to(torch::kCUDA),
            Cov3DPrecomp);

        // Post Processing
        cv::Mat RenderImg = TensorToCVMat(rendererd_image);
        CombinedImg = CombindImages(im, RenderImg);
    }

    cv::Mat imWithInfo;
    DrawTextInfo(CombinedImg,state, imWithInfo);
    return imWithInfo;
}

void FrameDrawer::DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText)
{
    stringstream s;
    if(nState==Tracking::NO_IMAGES_YET)
        s << " WAITING FOR IMAGES";
    else if(nState==Tracking::NOT_INITIALIZED)
        s << " TRYING TO INITIALIZE ";
    else if(nState==Tracking::OK)
    {
        if(!mbOnlyTracking)
            s << "SLAM MODE |  ";
        else
            s << "LOCALIZATION | ";
        int nMaps = mpAtlas->CountMaps();
        int nKFs = mpAtlas->KeyFramesInMap();
        int nMPs = mpAtlas->MapPointsInMap();
        s << "Maps: " << nMaps << ", KFs: " << nKFs << ", MPs: " << nMPs << ", Matches: " << mnTracked;
        if(mnTrackedVO>0)
            s << ", + VO matches: " << mnTrackedVO;
    }
    else if(nState==Tracking::LOST)
    {
        s << " TRACK LOST. TRYING TO RELOCALIZE ";
    }
    else if(nState==Tracking::SYSTEM_NOT_READY)
    {
        s << " LOADING ORB VOCABULARY. PLEASE WAIT...";
    }

    int baseline=0;
    cv::Size textSize = cv::getTextSize(s.str(),cv::FONT_HERSHEY_PLAIN,1,1,&baseline);

    imText = cv::Mat(im.rows+textSize.height+10,im.cols,im.type());
    im.copyTo(imText.rowRange(0,im.rows).colRange(0,im.cols));
    imText.rowRange(im.rows,imText.rows) = cv::Mat::zeros(textSize.height+10,im.cols,im.type());
    cv::putText(imText,s.str(),cv::Point(5,imText.rows-5),cv::FONT_HERSHEY_PLAIN,1,cv::Scalar(255,255,255),1,8);

}

void FrameDrawer::Update(Tracking *pTracker)
{
    unique_lock<mutex> lock(mMutex);
    pTracker->mImOrigin.copyTo(mImOrigin);
    pTracker->mImGray.copyTo(mIm);
    mvCurrentKeys=pTracker->mCurrentFrame.mvKeys;
    mThDepth = pTracker->mCurrentFrame.mThDepth;
    mvCurrentDepth = pTracker->mCurrentFrame.mvDepth;

    if(both){
        mvCurrentKeysRight = pTracker->mCurrentFrame.mvKeysRight;
        pTracker->mImRight.copyTo(mImRight);
        N = mvCurrentKeys.size() + mvCurrentKeysRight.size();
    }
    else{
        N = mvCurrentKeys.size();
    }

    mvbVO = vector<bool>(N,false);
    mvbMap = vector<bool>(N,false);
    mbOnlyTracking = pTracker->mbOnlyTracking;

    //Variables for the new visualization
    mCurrentFrame = pTracker->mCurrentFrame;
    mmProjectPoints = mCurrentFrame.mmProjectPoints;
    mmMatchedInImage.clear();

    mvpLocalMap = pTracker->GetLocalMapMPS();
    mvMatchedKeys.clear();
    mvMatchedKeys.reserve(N);
    mvpMatchedMPs.clear();
    mvpMatchedMPs.reserve(N);
    mvOutlierKeys.clear();
    mvOutlierKeys.reserve(N);
    mvpOutlierMPs.clear();
    mvpOutlierMPs.reserve(N);

    if(pTracker->mLastProcessedState==Tracking::NOT_INITIALIZED)
    {
        mvIniKeys=pTracker->mInitialFrame.mvKeys;
        mvIniMatches=pTracker->mvIniMatches;
    }
    else if(pTracker->mLastProcessedState==Tracking::OK)
    {
        for(int i=0;i<N;i++)
        {
            MapPoint* pMP = pTracker->mCurrentFrame.mvpMapPoints[i];
            if(pMP)
            {
                if(!pTracker->mCurrentFrame.mvbOutlier[i])
                {
                    if(pMP->Observations()>0)
                        mvbMap[i]=true;
                    else
                        mvbVO[i]=true;

                    mmMatchedInImage[pMP->mnId] = mvCurrentKeys[i].pt;
                }
                else
                {
                    mvpOutlierMPs.push_back(pMP);
                    mvOutlierKeys.push_back(mvCurrentKeys[i]);
                }
            }
        }

    }
    mState=static_cast<int>(pTracker->mLastProcessedState);
}

void FrameDrawer::GetGaussianRenderData(int &GauSHDegree, torch::Tensor &Means3D, torch::Tensor &Opacity, torch::Tensor &Scales, 
                                        torch::Tensor &Rotation, torch::Tensor &FeaturesDC, torch::Tensor &FeaturesRest)
{
    torch::NoGradGuard no_grad;

    GauSHDegree = mvpLocalMap[0]->GetGauSHDegree();

    const int FeaturestDim = std::pow(GauSHDegree + 1, 2) - 1;
    int SizeofGaussians = 0;
    for(int i = 0; i < mvpLocalMap.size(); i++){
        MapPoint* pMP = mvpLocalMap[i];
        if(pMP)
            SizeofGaussians += pMP->GetGaussianNum();
    }
    // std::cout << ">>>>>>>[FrameDrawer] The numbers of Gaussian: " << SizeofGaussians << std::endl;

    // Get Gaussians in local Map
    Means3D = torch::zeros({SizeofGaussians, 3}, torch::dtype(torch::kFloat)).to(torch::kCUDA);
    Opacity = torch::zeros({SizeofGaussians, 1}, torch::dtype(torch::kFloat)).to(torch::kCUDA);
    Scales = torch::zeros({SizeofGaussians, 3}, torch::dtype(torch::kFloat)).to(torch::kCUDA);
    Rotation = torch::zeros({SizeofGaussians, 4}, torch::dtype(torch::kFloat)).to(torch::kCUDA);
    FeaturesDC = torch::zeros({SizeofGaussians, 1, 3}, torch::dtype(torch::kFloat)).to(torch::kCUDA);
    FeaturesRest = torch::zeros({SizeofGaussians, FeaturestDim, 3}, torch::dtype(torch::kFloat)).to(torch::kCUDA);

    int GaussianClusterIndex = 0;
    for(int i = 0; i < mvpLocalMap.size(); i++)
    {
        MapPoint* pMP = mvpLocalMap[i];
        if(pMP)
        {
            int GaussianClusterNum = pMP->GetGaussianNum();
            if(GaussianClusterNum)
            {
                // std::cout << "[FrameDrawer] GaussianClusterNum Check: [ " << GaussianClusterIndex << ", " << GaussianClusterIndex + GaussianClusterNum << " ]" << std::endl;
                // std::cout << "[FrameDrawer] Means3D Check " << pMP->GetGauOpacity() << std::endl;
                // std::cout << "[InitializeOptimization] The numbers of Gaussian Cluster in Map: [ " << GaussianClusterIndex << ", " << GaussianClusterIndex + GaussianClusterNum << " ]" << std::endl;
                // std::cout << "[InitializeOptimization] The numbers of Gaussian Cluster in Map: " << pMP->GetGauWorldPos() << std::endl;
                Means3D.index_put_({torch::indexing::Slice(GaussianClusterIndex, GaussianClusterIndex + GaussianClusterNum), torch::indexing::Slice()},  pMP->GetGauWorldPos());
                Opacity.index_put_({torch::indexing::Slice(GaussianClusterIndex, GaussianClusterIndex + GaussianClusterNum), torch::indexing::Slice()},  pMP->GetGauOpacity());
                Scales.index_put_({torch::indexing::Slice(GaussianClusterIndex, GaussianClusterIndex + GaussianClusterNum), torch::indexing::Slice()},   pMP->GetGauScale());
                Rotation.index_put_({torch::indexing::Slice(GaussianClusterIndex, GaussianClusterIndex + GaussianClusterNum), torch::indexing::Slice()}, pMP->GetGauWorldRot());
                FeaturesDC.index_put_({torch::indexing::Slice(GaussianClusterIndex, GaussianClusterIndex + GaussianClusterNum), torch::indexing::Slice(), torch::indexing::Slice()}, pMP->GetGauFeatureDC());
                FeaturesRest.index_put_({torch::indexing::Slice(GaussianClusterIndex, GaussianClusterIndex + GaussianClusterNum), torch::indexing::Slice(), torch::indexing::Slice()}, pMP->GetGauFeaturest());
                GaussianClusterIndex += GaussianClusterNum;
            }
        }
    }
}

void FrameDrawer::GetCamParams(int &ImHeight, int &ImWidth, float &TanFovx, float &TanFovy, torch::Tensor &ViewMatrix, torch::Tensor &ProjMatrix, torch::Tensor &CamCenter)
{
    mCurrentFrame.GetGaussianRenderParams(ImHeight, ImWidth, TanFovx, TanFovy);
    
    // std::cout << "[FrameDrawer] mImHeight: " << ImHeight << std::endl;
    // std::cout << "[FrameDrawer] mImWidth: " << ImWidth << std::endl;
    // std::cout << "[FrameDrawer] mTanFovx: " << TanFovx << std::endl;
    // std::cout << "[FrameDrawer] mTanFovy: " << TanFovy << std::endl;
    ProjMatrix = GetFrameProjMatrix(TanFovx, TanFovy, 0.01f, 100.0f);

    Sophus::SE3<float> Tcw = mCurrentFrame.GetPose();
    ViewMatrix = GetViewMatrix(Tcw);
    CamCenter = ViewMatrix.inverse()[3].slice(0, 0, 3);

    // std::cout << "[FrameDrawer] ProjMatrix: "<< ProjMatrix << std::endl;
    // std::cout << "[FrameDrawer] ViewMatrix: "<< ViewMatrix << std::endl;
}

torch::Tensor FrameDrawer::GetFrameProjMatrix(const float TanFovx, const float TanFovy, const float Near, const float Far)
{
    float top = TanFovy * Near;
    float bottom = -top;
    float right = TanFovx * Near;
    float left = -right;

    Eigen::Matrix4f P = Eigen::Matrix4f::Zero();

    float z_sign = 1.f;

    P(0, 0) = 2.f * Near / (right - left);
    P(1, 1) = 2.f * Near / (top - bottom);
    P(0, 2) = (right + left) / (right - left);
    P(1, 2) = (top + bottom) / (top - bottom);
    P(3, 2) = z_sign;
    P(2, 2) = z_sign * Far / (Far - Near);
    P(2, 3) = -(Far * Near) / (Far - Near);

    // create torch::Tensor from Eigen::Matrix
    torch::Tensor ProjMatrix = torch::from_blob(P.data(), {4, 4}, torch::kFloat);
    return ProjMatrix.clone().to(torch::kCUDA);
}

torch::Tensor FrameDrawer::GetViewMatrix(Sophus::SE3<float> &Tcw)
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
    return W2CTensor.clone().to(torch::kCUDA);
}

cv::Mat FrameDrawer::TensorToCVMat(torch::Tensor tensor)
{
    tensor = tensor.squeeze().detach().permute({1, 2, 0});
    tensor = tensor.mul(255).clamp(0, 255).to(torch::kU8);
    tensor = tensor.to(torch::kCPU);
    int64_t height = tensor.size(0);
    int64_t width = tensor.size(1);
    cv::Mat mat = cv::Mat(height, width, CV_8UC3, tensor.data_ptr());
    return mat.clone();
}

cv::Mat FrameDrawer::CombindImages(const cv::Mat Img1, const cv::Mat Img2)
{
    cv::Mat CombinedImg = cv::Mat(480,640,CV_8UC3, cv::Scalar(0,0,0));
    if (Img1.cols == Img2.cols && Img1.rows == Img2.rows)
    {
        int rows = Img1.rows;
        int cols = Img1.cols;
        CombinedImg = cv::Mat(rows * 2, cols, CV_8UC3, cv::Scalar(0,0,0));

        // Copy images in correct position
        Img1.copyTo(CombinedImg(cv::Rect(0, 0, cols, rows)));
        Img2.copyTo(CombinedImg(cv::Rect(0, rows, cols, rows)));
    }    
    else
    {
        std::cout << "[FraweDrawer]: THe rendered image has incorrected size !!!" << std::endl;
    }
    return CombinedImg;
}
} //namespace ORB_SLAM
