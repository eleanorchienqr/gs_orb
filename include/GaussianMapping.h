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


#ifndef GAUSSIANMAPPING_H
#define GAUSSIANMAPPING_H

#include "KeyFrame.h"
#include "Atlas.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "Tracking.h"
#include "KeyFrameDatabase.h"
#include "Settings.h"
#include "RenderGUI.h"
#include "Config.h"

#include <torch/torch.h>
#include <mutex>


namespace ORB_SLAM3
{

class System;
class Tracking;
class LoopClosing;
class LocalMapping;
class Atlas;
class RenderGUI;

class GaussianMapping
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    GaussianMapping(System* pSys, Atlas* pAtlas, const float bMonocular, bool bInertial, const string &_strSeqName=std::string());

    void SetLoopCloser(LoopClosing* pLoopCloser);
    void SetTracker(Tracking* pTracker);
    void SetLocalMapper(LocalMapping* pLocalMapper);

    // Main function
    void Run();

    // For debugging (erase in normal mode)
    int mInitFr;

    // not consider far points (clouds)
    bool mbFarPoints;
    float mThFarPoints;

protected:

    System *mpSystem;
    Atlas* mpAtlas;

    bool mbMonocular;
    bool mbInertial;

    LoopClosing* mpLoopCloser;
    Tracking* mpTracker;
    LocalMapping* mpLocalMapper;

    // RenderGUI
    RenderGUI* mpGUI;
    // std::list<MapPoint*> mlpRecentAddedMapPoints;

};

} //namespace ORB_SLAM

#endif // GAUSSIANMAPPING_H
