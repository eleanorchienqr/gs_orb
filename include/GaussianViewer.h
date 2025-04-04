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


#ifndef GaussianViewer_H
#define GaussianViewer_H

#include "KeyFrame.h"
#include "Atlas.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "Tracking.h"
#include "MapDrawer.h"
#include "FrameDrawer.h"
#include "KeyFrameDatabase.h"
#include "Settings.h"
#include "Config.h"

#include <torch/torch.h>
#include <mutex>

#include <Thirdparty/imgui/imgui.h>
#include <Thirdparty/imgui/backends/imgui_impl_glfw.h>
#include <Thirdparty/imgui/backends/imgui_impl_opengl3.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <GL/gl.h>


namespace ORB_SLAM3
{

class System;
class Tracking;
class LoopClosing;
class LocalMapping;
class Atlas;
class FrameDrawer;
class MapDrawer;

class GaussianViewer
{
public:
    GaussianViewer(System* pSystem, FrameDrawer* pFrameDrawer, MapDrawer* pMapDrawer, Tracking *pTracking);

    void SetLoopCloser(LoopClosing* pLoopCloser);
    void SetTracker(Tracking* pTracker);
    void SetLocalMapper(LocalMapping* pLocalMapper);

    // void newParameterLoader(Settings* settings);

    // Main thread function for rendering scenes from Gaussians.
    // Drawing is refreshed according to the camera fps. We use ImGUI.
    void Run();

    // void RequestFinish();

    // void RequestStop();

    // bool isFinished();

    // bool isStopped();

    // bool isStepByStep();

    // void Release();

    // For debugging (erase in normal mode)
    int mInitFr;

    // not consider far points (clouds)
    bool mbFarPoints;
    float mThFarPoints;

protected:

    System *mpSystem;
    Atlas* mpAtlas;
    FrameDrawer* mpFrameDrawer;
    MapDrawer* mpMapDrawer;

    bool mbMonocular;
    bool mbInertial;

    LoopClosing* mpLoopCloser;
    Tracking* mpTracker;
    LocalMapping* mpLocalMapper;

    // bool ParseViewerParamFile(cv::FileStorage &fSettings);

    // bool Stop();

    // 1/fps in ms
    double mT = 1e3;
    float mImageWidth, mImageHeight;
    float mImageViewerScale;

    float mViewpointX, mViewpointY, mViewpointZ, mViewpointF;

    // Window settings
    const int mWindowSizeWidth = 1600;
    const int mWindowSizeHeight = 1200;

    GLFWwindow* mGLFWindow = nullptr;

    bool mGUIRedraw = true;
    bool mRenderWindow = false; // open in InitializeWindow; control render or not

    // ImGUI widgets
    bool mToolActive = true;
    bool mFollowCamera = true;
    bool mFromBehind = true;
    bool mShowCameraObjects = false;
    bool mShowActiveWindow = false;
    bool mShowAxis = false;
    bool mRenderDepth = false;
    bool mRenderOpacity = false;
    bool mRenderTimeShader = false;
    bool mRenderElpsoidShader = false;

    // Thread management
    bool mbFinishRequested;
    bool mbFinished;
    bool mbStopped;
    bool mbStopRequested;

    // std::mutex mMutexFinish;
    // std::mutex mMutexStop;
    // bool mbStopTrack;

protected:
    void ImGUIWindowTest();

    void InitializeGLFW();
    void InitializeImGUI();

    void ShowMenuBar();
    void ShowWidgets();

    // Thread Functions
    // bool CheckFinish();
    // void SetFinish();

};


} //namespace ORB_SLAM

#endif // GaussianViewer_H
