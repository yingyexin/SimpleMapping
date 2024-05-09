/**
* This file is part of SimpleMapping. Part of the code is modified based on TANDEM.
*
* Copyright (C) 2023 Yingye Xin, Xingxing Zuo, Dongyue Lu and Stefan Leutenegger, Technical University of Munich.
* Copyright (C) 2020 Lukas Koestler, Nan Yang, Niclas Zeller and Daniel Cremers, Technical University of Munich and Artisense.
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

// #ifndef PBA_MVSBACKEND_H
// #define PBA_MVSBACKEND_H

#pragma once

#include "utils/Timer.h"
#include "libdr/dr_fusion/src/dr_fusion/dr_fusion.h"
#include "libdr/dr_mvsnet/src/dr_mvsnet/dr_mvsnet.h"
#include "FrameDrawer.h"
#include "KeyFrame.h"

#include <memory>
#include <utility>
#include <vector>
#include <boost/thread/mutex.hpp>
#include <opencv2/opencv.hpp>

#include <boost/thread/thread.hpp>
#include <algorithm>
#include <boost/filesystem.hpp>

namespace ORB_SLAM3
{

class MVSBackendImpl;
class FrameDrawer;
class KeyFrame;
class TrackingDepthMap;

class MVSBackend {
public:
  explicit MVSBackend(
      int width, int height,
      DrMvsnet *mvsnet, DrFusion *fusion,
      Timer *dr_timer,
      const string strSeqName,
      bool bUpdateNeighDepths);

  ~MVSBackend();

  // Must check Ready() before!
  void CallAsync(
      int view_num_in,
      cv::Mat const &cur_bgr_in,
      std::vector<cv::Mat> const &src_bgr_in,
      cv::Mat const &cur_K0_in,
      cv::Mat const &cur_invK_in,
      std::vector<cv::Mat> const &src_K_in,
      std::vector<cv::Mat> const &src_c2ws_in,
      cv::Mat const &cur_c2w_in,
      cv::Mat const &cur_sparse_in,
      FrameDrawer *mpFrameDrawer,
      const int FrameId,
      KeyFrame *cur_kf
  );

  // Non-blocking
  bool Ready();

  // Blocking
  void Wait();

public:
  MVSBackendImpl *impl;
};

//#endif //PBA_MVSBACKEND_H
}//namespace ORB_SLAM3