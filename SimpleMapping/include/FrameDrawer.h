/**
* This file is part of SimpleMapping. Most of the code is modified based on ORB-SLAM3.
*
* Copyright (C) 2023 Yingye Xin, Xingxing Zuo, Dongyue Lu and Stefan Leutenegger, Technical University of Munich.
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* SimpleMapping is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* SimpleMapping is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/


#ifndef FRAMEDRAWER_H
#define FRAMEDRAWER_H

#include "Tracking.h"
#include "MapPoint.h"
#include "LocalMapping.h"
#include "Atlas.h"
#include "utils/MinimalImage.h"
#include "Settings.h"

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include<mutex>
#include <unordered_set>


namespace ORB_SLAM3
{

class Tracking;
class Viewer;
class LocalMapping;

class FrameDrawer
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    FrameDrawer(Atlas* pAtlas, Settings* settings);

    // Update info from the last processed frame.
    void Update(Tracking *pTracker);

    // Draw last processed frame.
    cv::Mat DrawFrame(float imageScale=1.f);
    cv::Mat DrawRightFrame(float imageScale=1.f);

    bool both;
    
    // Visualize the MVSNet and reconstruction results
    void pushDrKfImage(unsigned char * bgr, const double timestamp);
    void pushDrKfDepth(float * image, float depth_min, float depth_max, const double timestamp);
    void pushDrMesh(size_t num, float const* vert, float const* cols);
    bool bFrameReady = false;
    bool bMeshChanged = false;
    size_t dr_mesh_num = 0;
    cv::Mat DrawMvsFrame(){return mImBGR;};
    cv::Mat DrawMvsDepth(){return mInternalDepth;};
    float* getDrMeshVert(){return dr_mesh_vert;};
    float* getDrMeshCols(){return dr_mesh_cols;};

protected:

    void DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText);

    // For MVSNet visualization 
    int mWidth;
    int mHeight;
    bool mbMvsDepth=false;
    MinimalImageB3* internalDrKfImage;
    MinimalImageB3* internalDrKfDepth;

    // Mesh Stuff
    const size_t dr_mesh_num_max = 60000000;
    float* dr_mesh_vert;
    float* dr_mesh_cols;

    // Info of the visualization frame
    cv::Mat mIm, mImRight, mImBGR, mInternalDepth;
    int N;
    vector<cv::KeyPoint> mvCurrentKeys,mvCurrentKeysRight;
    vector<bool> mvbMap, mvbVO;
    bool mbOnlyTracking;
    int mnTracked, mnTrackedVO;
    vector<cv::KeyPoint> mvIniKeys;
    vector<int> mvIniMatches;
    int mState;
    std::vector<float> mvCurrentDepth;
    float mThDepth;

    Atlas* mpAtlas;

    std::mutex mMutex;
    vector<pair<cv::Point2f, cv::Point2f> > mvTracks;

    Frame mCurrentFrame;
    vector<MapPoint*> mvpLocalMap;
    vector<cv::KeyPoint> mvMatchedKeys;
    vector<MapPoint*> mvpMatchedMPs;
    vector<cv::KeyPoint> mvOutlierKeys;
    vector<MapPoint*> mvpOutlierMPs;

    map<long unsigned int, cv::Point2f> mmProjectPoints;
    map<long unsigned int, cv::Point2f> mmMatchedInImage;

};

} //namespace ORB_SLAM

#endif // FRAMEDRAWER_H
