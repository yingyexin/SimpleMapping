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

#pragma once

#include "MvsBackend.h"
#include "utils/MinimalImage.h"

namespace ORB_SLAM3
{

class MVSBackendImpl {
public:
  explicit MVSBackendImpl(
      int width, int height, 
      DrMvsnet *mvsnet, DrFusion *fusion, Timer *dr_timer,
      const string strSeqName, bool bUpdateNeighDepths) : \
      width(width), height(height), mvsnet(mvsnet), fusion(fusion), \
      dr_timer(dr_timer), mstrSeqName(strSeqName), mbUpdateNeighDepths(bUpdateNeighDepths){
    dr_timing = dr_timer != nullptr;
    get_mesh = !dr_timing;
    tracker_depth_map_A = new TrackingDepthMap(width, height); //TODO:DELETE
    tracker_depth_map_B = new TrackingDepthMap(width, height);
    tracker_depth_map_valid = nullptr;
    tracker_depth_map_use_next = tracker_depth_map_A;
    worker_thread = boost::thread(&MVSBackendImpl::Loop, this);

    internalDrKfImage = new MinimalImageB3(width, height);
    internalDrKfDepth = new MinimalImageB3(width, height);
    internalFusionKfDepth = new MinimalImageB3(width, height);
    internalFusionKfImage = new MinimalImageB3(width, height);
    internalDrKfImage->setBlack();
    internalDrKfDepth->setBlack();
    internalFusionKfDepth->setBlack();
    internalFusionKfImage->setBlack();

    path_save_img = "./result/debug/" + mstrSeqName + "/image/";
    path_save_depth = "./result/debug/" + mstrSeqName + "/depth/";
    path_save_fusionDepth = "./result/debug/" + mstrSeqName + "/fusionDepth/";

    boost::system::error_code ec;
    if(!boost::filesystem::exists(path_save_img, ec)){
        bool success_created = boost::filesystem::create_directories(path_save_img, ec);
        if(!success_created){
            cout<<"Fail to create the directory for saving image in mvs_backend: "<<path_save_img<<", msg: "<<ec.message()<<std::endl;
        }
    }

    if(!boost::filesystem::exists(path_save_depth, ec)){
        bool success_created = boost::filesystem::create_directories(path_save_depth, ec);
        if(!success_created){
            cout<<"Fail to create the directory for saving depth in mvs_backend: "<<path_save_depth<<", msg: "<<ec.message()<<std::endl;
        }
    }

    if(!boost::filesystem::exists(path_save_fusionDepth, ec)){
        bool success_created = boost::filesystem::create_directories(path_save_fusionDepth, ec);
        if(!success_created){
            cout<<"Fail to create the directory for saving depth in mvs_backend: "<<path_save_fusionDepth<<", msg: "<<ec.message()<<std::endl;
        }
    }

  };

  ~MVSBackendImpl(){
    delete tracker_depth_map_B;
    delete tracker_depth_map_A;
  };


  // Blocking for last input. Non-blocking for this input.
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
      const int FrameId,
      KeyFrame* cur_kf
  );

  // Non-blocking
  bool Ready();

  void Wait();

  // visualize depth map, mesh depth
  FrameDrawer *mpFrameDrawer = nullptr;
  void pushDrKfImage(unsigned char * bgr, const double timestamp);
  void pushDrKfDepth(float * pdepth, const double timestamp);
  void pushFusionDepth(float * image, const double timestamp);
  void pushFusionKfImage(const unsigned char *image, const double timestamp) ;

  DrMvsnetOutput* getOutput(){return output_previous;};

private:
  void CallSequential();

  void Loop();

  const bool dense_tracking = true; //TODO IS USED

  DrMvsnet *mvsnet = nullptr;
  DrFusion *fusion = nullptr;
  const string mstrSeqName;
  // used for updating neighbour keyframes(in covisibility graph) 
  // during BA (NOT USED IN PAPER VERSION)
  const bool mbUpdateNeighDepths;

  // Will run Loop.
  boost::thread worker_thread;

  // Protects all below variables
  boost::mutex mut;
  bool running = true;
  bool unprocessed_data = false;

  boost::condition_variable newInputSignal;
  boost::condition_variable dataProcessedSignal;

  // Internal
  bool setting_debugout_runquiet = true;
  bool dr_timing = false;
  int call_num = 0;
  bool get_mesh = true;

  float mesh_lower_corner[3] = {-8, -8, -4};
  float mesh_upper_corner[3] = {8, 8, 4};
  const int mesh_extraction_freq = 5;

  const int width;
  const int height;
  Timer *dr_timer;

  boost::mutex mut_tracker;
  TrackingDepthMap *tracker_depth_map_valid = nullptr;
  TrackingDepthMap *tracker_depth_map_use_next = nullptr;
  vector<TrackingDepthMap *> mvtracker_depth_map_neigh; //USE AS BA
  vector<KeyFrame *> mvKeyframe_pre_neigh;
  set<KeyFrame *> msKeyframe;
  TrackingDepthMap *tracker_depth_map_A;
  TrackingDepthMap *tracker_depth_map_B;

  // data from call
  bool has_to_wait_current = false;
  int view_num_current;
  cv::Mat cur_bgr_cur;
  std::vector<cv::Mat> src_bgr_cur;
  cv::Mat cur_K0_cur;
  cv::Mat cur_invK_cur;
  std::vector<cv::Mat> src_K_cur;
  cv::Mat cur_c2w_cur;
  std::vector<cv::Mat> src_c2ws_cur;
  cv::Mat cur_sparse_cur;
  int mFrameId_cur = 0;
  KeyFrame* mKeyframe_cur = nullptr; 

  DrMvsnetOutput *output_previous;

  // for visualize
  MinimalImageB3* internalDrKfImage;
  MinimalImageB3* internalFusionKfImage;
  MinimalImageB3* internalFusionKfDepth;
  MinimalImageB3* internalDrKfDepth;

  // data from last call
  bool has_to_wait_previous = false;
  int view_num_previous;
  cv::Mat cur_bgr_pre;
  std::vector<cv::Mat> src_bgr_pre;
  cv::Mat cur_K0_pre;
  cv::Mat cur_invK_pre;
  std::vector<cv::Mat> src_K_pre;
  cv::Mat cur_c2w_pre;
  std::vector<cv::Mat> src_c2ws_pre;
  cv::Mat cur_sparse_pre;
  int mFrameId_pre = 0;
  KeyFrame* mKeyframe_pre = nullptr; 

  std::string path_save_img = "./result/debug/seq_x/image/";
  std::string path_save_depth = "./result/debug/seq_x/depth/";
  std::string path_save_fusionDepth = "./result/debug/seq_x/fusionDepth/";

};

void MVSBackendImpl::Loop() {
  boost::unique_lock<boost::mutex> lock(mut);
  while (running) {
    if (unprocessed_data) {
      CallSequential();
      unprocessed_data = false;
      dataProcessedSignal.notify_all();
    }
    newInputSignal.wait(lock);
  }
}

void MVSBackendImpl::CallSequential() {
  int id_time;
  call_num++;

  /* --- 3.5 CURRENT: Call MVSNet Async --- */
  std::vector<unsigned char *> src_bgr_cur_ptr;
  for (auto const &e : src_bgr_cur) src_bgr_cur_ptr.push_back(e.data);
  std::vector<float *> src_K_cur_ptr;
  for (auto const &e : src_K_cur) src_K_cur_ptr.push_back((float *) e.data);
  std::vector<float *> src_c2ws_cur_ptr;
  for (auto const &e : src_c2ws_cur) src_c2ws_cur_ptr.push_back((float *) e.data);

  mvsnet->CallAsync(
      height,
      width,
      (unsigned char *) cur_bgr_cur.data,
      src_bgr_cur_ptr.data(),
      (float *) cur_K0_cur.data,
      (float *) cur_invK_cur.data,
      src_K_cur_ptr.data(),
      (float *) cur_c2w_cur.data,
      src_c2ws_cur_ptr.data(),
      (float *) cur_sparse_cur.data,
      view_num_current,
      false
  );
  has_to_wait_current = true;

  // Here we have the lock (the loop function has it)
  if (has_to_wait_previous && fusion != nullptr) {
    /* --- 3. PREVIOUS: Integrate into Fusion --- */
    if (dr_timing) id_time = dr_timer->start_timing("IntegrateScanAsync");
    fusion->IntegrateScanAsync(cur_bgr_pre.data, output_previous->depth_dense, (float *) cur_c2w_pre.data);
    if (dr_timing) dr_timer->end_timing("IntegrateScanAsync", id_time);

    /* --- 4. PREVIOUS: RenderAsync for tracker --- */
    std::vector<float const *> poses_to_render_previous;
    if (dense_tracking) { 
      poses_to_render_previous.push_back(tracker_depth_map_use_next->cam_to_world);
      if(mbUpdateNeighDepths){
          for(int i = 0; i < mvtracker_depth_map_neigh.size(); i++)
              poses_to_render_previous.push_back(mvtracker_depth_map_neigh[i]->cam_to_world);
      }
    }
    fusion->RenderAsync(poses_to_render_previous); // pass cur_c2w, obtain current depth 

    /* --- 5. PREVIOUS: Get render result --- */
    std::vector<unsigned char *> bgr_rendered_pre;
    std::vector<float *> depth_rendered_pre;
    fusion->GetRenderResult(bgr_rendered_pre, depth_rendered_pre);
    
    // if(!dr_timing){
    //   // save the fused depth and rendered image 
    //   pushFusionDepth(depth_rendered_pre[0], mKeyframe_pre->mTimeStamp);
    //   pushFusionKfImage(bgr_rendered_pre[0], mKeyframe_pre->mTimeStamp);
    // }
    
    if (dense_tracking) {
      memcpy(tracker_depth_map_use_next->depth, depth_rendered_pre[0], sizeof(float) * width * height);
      tracker_depth_map_use_next->is_valid = true; // atomic

      /* --- 5.5 PREVIOUS: Set Tracker --- */
      {
        // boost::unique_lock<boost::mutex> lock_tracker(mut_tracker);
        // Ternary will only be false on first iter
        TrackingDepthMap *tmp = tracker_depth_map_valid ? tracker_depth_map_valid : tracker_depth_map_B;
        tracker_depth_map_valid = tracker_depth_map_use_next;
        tracker_depth_map_use_next = tmp;
        mKeyframe_pre->SetTrackDepth(tracker_depth_map_valid);

        for (int i=0; i < mvtracker_depth_map_neigh.size(); i++){
          TrackingDepthMap* track_neigh = mvtracker_depth_map_neigh[i];
          if(track_neigh->is_valid){
            memcpy(track_neigh->depth, depth_rendered_pre[i+1], sizeof(float) * width * height);
            mvKeyframe_pre_neigh[i]->SetTrackDepth(track_neigh);
          }
        }
      }
    }

    /* --- 6. PREVIOUS: Get mesh and push to output_previous wrapper --- */
    if (get_mesh && (call_num % mesh_extraction_freq) == 0) {
      if (dr_timing) id_time = dr_timer->start_timing("fusion-mesh");
      fusion->ExtractMeshAsync(mesh_lower_corner, mesh_upper_corner);
      fusion->GetMeshSync();
      if (dr_timing) dr_timer->end_timing("fusion-mesh", id_time);

      if (mpFrameDrawer){
          mpFrameDrawer->pushDrMesh(fusion->dr_mesh_num, fusion->dr_mesh_vert, fusion->dr_mesh_cols);
      }

      has_to_wait_previous = false;
    }
  }

  // Now we swap *_previous <- *_current
  view_num_previous = view_num_current;
  cur_bgr_pre = cur_bgr_cur;
  src_bgr_pre = src_bgr_cur;
  cur_invK_pre = cur_invK_cur;
  src_K_pre = src_K_cur;
  cur_c2w_pre = cur_c2w_cur;
  src_c2ws_pre = src_c2ws_cur;
  has_to_wait_previous = has_to_wait_current;
  mFrameId_pre = mFrameId_cur;
  mKeyframe_pre = mKeyframe_cur;
  msKeyframe.insert(mKeyframe_cur);
  unprocessed_data = false;
}


void MVSBackendImpl::CallAsync(
      int view_num_in,
      cv::Mat const &cur_bgr_in,
      std::vector<cv::Mat> const &src_bgr_in,
      cv::Mat const &cur_K0_in,
      cv::Mat const &cur_invK_in,
      std::vector<cv::Mat> const &src_K_in,
      std::vector<cv::Mat> const &src_c2ws_in,
      cv::Mat const &cur_c2w_in,
      cv::Mat const &cur_sparse_in,
      const int FrameId,
      KeyFrame *cur_kf
) {
  using std::cerr;
  using std::cout;
  using std::endl;

  if (unprocessed_data) {
    cerr << "Wrong Call Order in ORB-MVS Backend. Will just return." << endl;
    return;
  }

  {
    boost::unique_lock<boost::mutex> lock(mut);
    {
      // Now we have the lock
      // We will process the MVSNet result for the *_previous data
      // The Loop will finish the processing of the *_previous data
      // The end of Loop will switch *_previous <- *_current

      /* --- 0. Copy input Data --- */
      view_num_current = view_num_in;
      cur_bgr_cur = cur_bgr_in;
      src_bgr_cur = src_bgr_in;
      cur_K0_cur = cur_K0_in;
      cur_invK_cur = cur_invK_in;
      src_K_cur = src_K_in;
      cur_c2w_cur = cur_c2w_in;
      src_c2ws_cur = src_c2ws_in;
      cur_sparse_cur = cur_sparse_in;
      mFrameId_cur = FrameId;
      mKeyframe_cur = cur_kf;

      if (has_to_wait_previous) {
        /* --- 0. PREVIOUS: Push previous pose for render*/
        if (dense_tracking) {
          tracker_depth_map_use_next->is_valid = false; // atomic
          tracker_depth_map_use_next->Timestamp = mKeyframe_pre->mTimeStamp;
          // cout << "keyframe number = " << to_string(1e9*mKeyframe_pre->mTimeStamp).substr(0, 19) << endl;
          memcpy(tracker_depth_map_use_next->cam_to_world, cur_c2w_pre.data, sizeof(float) * 16);

          if(mbUpdateNeighDepths){
              mvKeyframe_pre_neigh.clear();
              mvtracker_depth_map_neigh.clear();
              for(auto kf_neigh: mKeyframe_pre->GetBestCovisibilityKeyFrames(10)){
                  if(msKeyframe.count(kf_neigh))
                      mvKeyframe_pre_neigh.push_back(kf_neigh);
              }

              float c2w_default[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1};
              // printf("neighbour keyframe number = %d \n", mvKeyframe_pre_neigh.size());

              for (int i=0; i < 5; i++){
                  TrackingDepthMap* track_neigh = new TrackingDepthMap(width, height);

                  if(mvKeyframe_pre_neigh.size() > i){
                      track_neigh->is_valid = true;
                      track_neigh->Timestamp = mvKeyframe_pre_neigh[i]->mTimeStamp;
                      // cout << "init neighbour keyframe number = " << to_string(1e9*mvKeyframe_pre_neigh[i]->mTimeStamp).substr(0, 19) << endl;

                      const auto matrix = mvKeyframe_pre_neigh[i]->GetPoseInverse().matrix();
                      Eigen::Map<Eigen::MatrixXf>(track_neigh->cam_to_world, matrix.rows(), matrix.cols()) = matrix.transpose();

                      mvtracker_depth_map_neigh.push_back(track_neigh);}
                  else{
                      track_neigh->is_valid = false;
                      memcpy(track_neigh->cam_to_world, c2w_default, sizeof(float) * 16);
                      mvtracker_depth_map_neigh.push_back(track_neigh);}
              }
          }

        }

        /* --- 1. PREVIOUS: Get MVSNet result --- */
        if (!mvsnet->Ready()) {
          std::cerr << "MVSNET IS NOT READY!!! WHY" << std::endl;
          exit(EXIT_FAILURE);
        }
        output_previous = mvsnet->GetResult();
        
        /* --- 2. PREVIOUS: Push depth map to output_previous wrapper --- */
        if(mpFrameDrawer){
            // visualize the keyframe image and predicted depth in viewer
            mpFrameDrawer->pushDrKfImage(cur_bgr_pre.data, mKeyframe_pre->mTimeStamp);
            const float depth_max_value_previous = *std::max_element(output_previous->depth_dense, output_previous->depth_dense + width * height);
            mpFrameDrawer->pushDrKfDepth(output_previous->depth_dense, 0.01, depth_max_value_previous, mKeyframe_pre->mTimeStamp);
        }
        if(!dr_timing){
          // save keyframe image and corresponding predicted depth to the specific path
          pushDrKfImage(cur_bgr_pre.data, mKeyframe_pre->mTimeStamp);
          pushDrKfDepth(output_previous->depth_dense, mKeyframe_pre->mTimeStamp);
        }
      }

    }

    unprocessed_data = true;
    newInputSignal.notify_all();
  }
}

bool MVSBackendImpl::Ready() {
  return !unprocessed_data && mvsnet->Ready();
}

MVSBackend::MVSBackend(int width, int height, 
                       DrMvsnet *mvsnet, DrFusion *fusion, Timer *dr_timer,
                       const string strSeqName, bool bUpdateNeighDepths=false) {
  impl = new MVSBackendImpl(width, height, mvsnet, fusion, dr_timer, strSeqName, bUpdateNeighDepths);
}

MVSBackend::~MVSBackend() {
  delete impl;
}

bool MVSBackend::Ready() {
  return impl->Ready();
}

void MVSBackend::CallAsync(int view_num_in, cv::Mat const &cur_bgr_in, std::vector<cv::Mat> const &src_bgr_in,
      cv::Mat const &cur_K0_in, cv::Mat const &cur_invK_in, std::vector<cv::Mat> const &src_K_in, 
      std::vector<cv::Mat> const &src_c2ws_in, cv::Mat const &cur_c2w_in,
      cv::Mat const &cur_sparse_in, FrameDrawer *mpFrameDrawer, const int FrameId, KeyFrame *cur_kf) {

  if (impl->mpFrameDrawer == nullptr)
    impl->mpFrameDrawer = mpFrameDrawer;
  impl->CallAsync(
      view_num_in,
      cur_bgr_in,
      src_bgr_in,
      cur_K0_in,
      cur_invK_in,
      src_K_in,
      src_c2ws_in,
      cur_c2w_in,
      cur_sparse_in,
      FrameId,
      cur_kf
  );

}

void MVSBackend::Wait() {
  impl->Wait();
}

void MVSBackendImpl::Wait() {
  boost::unique_lock<boost::mutex> lock(mut);
  while (unprocessed_data) {
    dataProcessedSignal.wait(lock);
  }
  mvsnet->Wait();
  if (!Ready()) {
    std::cerr << "TandemBackendImpl must be Ready() after Wait(). Something went wrong." << std::endl;
    exit(EXIT_FAILURE);
  }
}

void MVSBackendImpl::pushDrKfImage(unsigned char * bgr, const double timestamp)
{
  memcpy(internalDrKfImage->data, bgr, sizeof(unsigned char) * 3 * internalDrKfImage->h * internalDrKfImage->w);
  cv::Mat internalKfImage = cv::Mat(height, width, CV_8UC3, internalDrKfImage->data);
  std::string filename = path_save_img + to_string(1e9*timestamp).substr(0, 19) + ".jpg";
  cv::imwrite(filename, internalKfImage);
}

void MVSBackendImpl::pushDrKfDepth(float * pdepth, const double timestamp)
{
  cv::Mat dense_depth = cv::Mat(height, width, CV_32FC1, pdepth);

  // Convert to 16 bit in mm to write. Some loss of accuracy but only up to a fraction of a mm
  // Perform depth_u16 = uint16(1000*depth + 0.5)
  // which converts to mm and rounds to nearest int because of the +.5
  cv::Mat depth = 1000 * dense_depth + 0.5;
  depth.convertTo(depth, CV_16UC1);

  std::string filename = path_save_depth +
      to_string(1e9*timestamp).substr(0, 19) + ".png";

  // printf("Generate dense depth written to %s\n", filename.c_str());
  cv::imwrite(filename, depth);
}

void MVSBackendImpl::pushFusionKfImage(const unsigned char *image, const double timestamp) 
{
  memcpy(internalFusionKfImage->data, image, sizeof(unsigned char) * 3 * internalDrKfImage->h * internalDrKfImage->w);
  cv::Mat internalFusionImage = cv::Mat(height, width, CV_8UC3, internalFusionKfImage->data);
  std::string filename = path_save_fusionDepth + to_string(1e9*timestamp).substr(0, 19) + ".jpg";
  cv::imwrite(filename, internalFusionImage);
}

void MVSBackendImpl::pushFusionDepth(float * pdepth, const double timestamp)
{
  cv::Mat dense_depth = cv::Mat(height, width, CV_32FC1, pdepth);

  // Convert to 16 bit in mm to write. Some loss of accuracy but only up to a fraction of a mm
  // Perform depth_u16 = uint16(1000*depth + 0.5)
  // which converts to mm and rounds to nearest int because of the +.5
  cv::Mat depth = 1000 * dense_depth + 0.5;
  depth.convertTo(depth, CV_16UC1);

  std::string filename = path_save_fusionDepth + 
      to_string(1e9*timestamp).substr(0, 19) + ".png";

  // printf("Generate dense depth written to %s\n", filename.c_str());
  cv::imwrite(filename, depth);
}
}//namespace ORB_SLAM3