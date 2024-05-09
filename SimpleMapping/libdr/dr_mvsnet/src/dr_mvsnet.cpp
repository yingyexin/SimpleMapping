/**
* This file is part of SimpleMapping.
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

#include <torch/torch.h>
#include <torch/script.h> // One-stop header.
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include <string>
#include <iostream>
#include <memory>
#include <vector>

#include <boost/thread/thread.hpp>
#include <chrono>

#include "dr_mvsnet.h"

class DrMvsnetImpl {
public:
  explicit DrMvsnetImpl(const char *filename, Timer *dr_timer) : \
      stream(at::cuda::getStreamFromPool(false)), dr_timer(dr_timer) {
    // Try to fix CUDA errors: https://github.com/pytorch/pytorch/issues/35736
    if (torch::cuda::is_available())std::cout << "DrMvsnet torch::cuda::is_vailable == true --> seems good" << std::endl;
    else std::cerr << "DrMvsnet torch::cuda::is_vailable == false --> probably this will crash" << std::endl;
    module = torch::jit::load(filename);
    dr_timing = dr_timer != nullptr;
    worker_thread = boost::thread(&DrMvsnetImpl::Loop, this);
  };

  ~DrMvsnetImpl() {
    {
      boost::unique_lock<boost::mutex> lock(mut);
      while (unprocessed_data) {
        dataProcessedSignal.wait(lock);
      }
      running = false;
      newInputSignal.notify_all();
    }
    worker_thread.join();
  }

  void CallAsync(int height,
                 int width,
                 unsigned char * cur_img,
                 unsigned char **src_img,
                 float *cur_K,
                 float *cur_invK,
                 float **src_K,
                 float *cur_c2w,
                 float **src_c2ws,
                 float *sparse_depth,
                 int view_num,
                 bool debug_print);

  DrMvsnetOutput *GetResult();

  bool Ready() { return !unprocessed_data; };

  void Wait();

private:
  void CallSequential();

  void Loop();

  // Will run Loop.
  boost::thread worker_thread;

  // Protects all below variables
  boost::mutex mut;
  bool running = true;
  bool unprocessed_data = false;

  boost::condition_variable newInputSignal;
  boost::condition_variable dataProcessedSignal;

  std::vector<torch::jit::IValue> inputs;
  int width_, height_;
  DrMvsnetOutput *output = nullptr;

  torch::jit::script::Module module;

  at::cuda::CUDAStream stream;
  bool dr_timing = false;
  Timer *dr_timer;
};

void DrMvsnetImpl::Loop() {
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

DrMvsnetOutput *DrMvsnetImpl::GetResult() {
  boost::unique_lock<boost::mutex> lock(mut);
  while (unprocessed_data) {
    dataProcessedSignal.wait(lock);
  }
  if (!output) {
    std::cerr << "Output should be valid pointer. Maybe you called GetResult more than once?" << std::endl;
    exit(EXIT_FAILURE);
  } else {
    DrMvsnetOutput *ret = output;
    output = nullptr;
    return ret;
  }
}

void DrMvsnetImpl::Wait() {
  boost::unique_lock<boost::mutex> lock(mut);
  while (unprocessed_data) {
    dataProcessedSignal.wait(lock);
  }
  if (!Ready()) {
    std::cerr << "DrMvsnetImpl must be ready after Wait" << std::endl;
    exit(EXIT_FAILURE);
  }
}

void DrMvsnet::Wait() {
  impl->Wait();
}

void DrMvsnetImpl::CallAsync(int height,
                             int width,
                             unsigned char * cur_img,
                             unsigned char **src_img,
                             float *cur_intr,
                             float *cur_inv_intr,
                             float **src_intr,
                             float *cur_c2w,
                             float **src_c2ws,
                             float *sparse_depth,
                             int view_num,
                             bool debug_print) {

  using std::cout;
  using std::endl;

  boost::unique_lock<boost::mutex> lock(mut);
  // Now we have the lock
  {
    at::cuda::CUDAStreamGuard stream_guard(stream);

    inputs.clear();
    height_ = height;
    width_ = width;

    constexpr int batch_size = 1;
    constexpr int channels = 3;

    // Check inputs
    for (int i = 0; i < view_num - 1; i++) {
      for (int j = i + 1; j < view_num; j++) {
        if (src_img[i] == src_img[j] ) {
          std::cerr << "ERROR: In Call Async passing the same data for index(imgs) " << i << " and " << j << std::endl;
          exit(EXIT_FAILURE);
        }
        if (src_c2ws[i] == src_c2ws[j]){
          std::cerr << "ERROR: In Call Async passing the same data for index(c2w) " << i << " and " << j << std::endl;
          exit(EXIT_FAILURE);
        }
      }
    }

    if (debug_print) {
      printf("--- DrMvsnetImpl::CallAsync ---\n");
      printf("W=%d, H=%d, total_view_num=%d \n", width, height, view_num + 1);
      printf("Cur Intrinsics:\n");
      for (int r = 0; r < 4; r++) 
        printf("%f %f %f %f\n", cur_intr[4 * r], cur_intr[4 * r + 1], cur_intr[4 * r + 2], cur_intr[4 * r + 3]);
      
      printf("Cur Inv Intrinsics:\n");
      for (int r = 0; r < 4; r++) 
        printf("%f %f %f %f\n", cur_inv_intr[4 * r], cur_inv_intr[4 * r + 1], cur_inv_intr[4 * r + 2], cur_inv_intr[4 * r + 3]);
      
      for (int i = 0; i < view_num; i++){
        printf("Source Intrinsics[%d]:\n", i);
        for (int r = 0; r < 4; r++) 
          printf("%f %f %f %f\n", src_intr[i][4 * r], src_intr[i][4 * r + 1], src_intr[i][4 * r + 2], src_intr[i][4 * r + 3]);
      }

      printf("\nCurrent C2W[%d]:\n");
      for (int r = 0; r < 4; r++) 
        printf("%f %f %f %f\n", cur_c2w[4 * r], cur_c2w[4 * r + 1], cur_c2w[4 * r + 2], cur_c2w[4 * r + 3]);

      for (int i = 0; i < view_num; i++) {
        printf("Source C2W[%d]:\n", i);
        float const *const c2w = src_c2ws[i];
        for (int r = 0; r < 4; r++) 
          printf("%f %f %f %f\n", c2w[4 * r], c2w[4 * r + 1], c2w[4 * r + 2], c2w[4 * r + 3]);
      }

      for (int i = 0; i < view_num; i++) printf("src_img[%d] = %p\n", i, src_img[i]);
    }

    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .layout(torch::kStrided)
        .device(torch::kCPU)
        .requires_grad(false);

    // cur_image: (B, C, H, W), cur_world_T_cam: (B, 4, 4), cur_invK: (B, 4, 4)
    auto cur_image = torch::empty({batch_size, channels, height, width}, options);
    auto cur_image_a = cur_image.accessor<float, 4>();
    auto cur_invK = torch::empty({batch_size, 4, 4}, options);
    auto cur_invK_a = cur_invK.accessor<float, 3>();
    auto cur_world_T_cam = torch::empty({batch_size, 4, 4}, options);
    auto cur_world_T_cam_a = cur_world_T_cam.accessor<float, 3>();
    for (int h = 0; h < height; h++)
      for (int w = 0; w < width; w++) {
        const int offset = channels * (width * h + w);
        cur_image_a[0][0][h][w] = ((float) cur_img[offset + 2]) / 255.0;
        cur_image_a[0][1][h][w] = ((float) cur_img[offset + 1]) / 255.0;
        cur_image_a[0][2][h][w] = ((float) cur_img[offset + 0]) / 255.0;
      }
    
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++)
        cur_invK_a[0][i][j] = cur_inv_intr[4 * i + j];

    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++)
        cur_world_T_cam_a[0][i][j] = cur_c2w[4 * i + j];
    
    auto cur_K = torch::empty({batch_size, 4, 4}, options);
    auto cur_K_a = cur_K.accessor<float, 3>();
    auto cur_sparse = torch::empty({batch_size, 1, height/2, width/2}, options);
    auto cur_sparse_a = cur_sparse.accessor<float, 4>();
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++)
        cur_K_a[0][i][j] = cur_intr[4 * i + j];
    
    for (int h = 0; h < height/2; h++)
      for (int w = 0; w < width/2; w++)
        cur_sparse_a[0][0][h][w] = ((float) sparse_depth[width/2 * h + w]);
        
    // src_image: (B, M, C, H, W), src_world_T_cam: (B, M, 4, 4), src_K: (B, M, 4, 4) 
    auto src_image = torch::empty({batch_size, view_num, channels, height, width}, options);
    auto src_image_a = src_image.accessor<float, 5>();
    auto src_K = torch::empty({batch_size, view_num, 4, 4}, options);
    auto src_K_a = src_K.accessor<float, 4>();
    auto src_world_T_cam = torch::empty({batch_size, view_num, 4, 4}, options);
    auto src_world_T_cam_a = src_world_T_cam.accessor<float, 4>();

    for (int view = 0; view < view_num; view++) {
      float const *sc2w = src_c2ws[view];
      for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
          src_world_T_cam_a[0][view][i][j] = sc2w[4 * i + j];

      float const *sk = src_intr[view];
      for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
          src_K_a[0][view][i][j] = sk[4 * i + j];

      unsigned char const *simg = src_img[view];
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          const int offset = channels * (width * h + w);
          // BGR -> RGB
          src_image_a[0][view][0][h][w] = ((float) simg[offset + 2]) / 255.0;
          src_image_a[0][view][1][h][w] = ((float) simg[offset + 1]) / 255.0;
          src_image_a[0][view][2][h][w] = ((float) simg[offset + 0]) / 255.0;
        }
      }
    }

    // image
    // TODO: This throws strange errors sometimes, nvidia-smi says (Detected Critical Xid Error) but due to async this might be from somewhere else
    inputs.emplace_back(cur_image.to(torch::kCUDA));
    inputs.emplace_back(src_image.to(torch::kCUDA));
    
    // intrinsic_matrix
    inputs.emplace_back(cur_K.to(torch::kCUDA));
    inputs.emplace_back(cur_invK.to(torch::kCUDA));
    inputs.emplace_back(src_K.to(torch::kCUDA));

    // cam_to_world
    inputs.emplace_back(cur_world_T_cam.to(torch::kCUDA));
    inputs.emplace_back(src_world_T_cam.to(torch::kCUDA));
    
    // sparse depth
    inputs.emplace_back(cur_sparse.to(torch::kCUDA));
  }
  unprocessed_data = true;
  newInputSignal.notify_all();
}

void DrMvsnetImpl::CallSequential() {
  // Inside here we are protected by a mutex
  // inputs is already set correctly
  int id_time;

  /* ---  Execute Model ---*/
  // The outputs are (depth)
  //  torch::NoGradGuard guard;
  c10::InferenceMode guard;
  at::cuda::CUDAStreamGuard stream_guard(stream);
  if (dr_timing) id_time = dr_timer->start_timing("mvsnet-forward");
  auto model_output = module.forward(inputs);
  if (dr_timing) dr_timer->end_timing("mvsnet-forward", id_time);

  /* --- Outputs --- */
  if (output) {
    std::cerr << "Output should internally be nullptr. Maybe you called CallAsync more than once?" << std::endl;
    exit(EXIT_FAILURE);
  }
  output = new DrMvsnetOutput(height_, width_);

  // if (flag_dis){ //TODO: DELETE
  //   const auto num_elements = model_output.toTuple()->elements().size();
  //   printf("num_elements = %d\n", num_elements);

  //   auto depth_tensor = model_output.toTuple()->elements()[1].toTensor().to(torch::kCPU);
  //   auto depth_a = depth_tensor.accessor<float, 4>();  
  //   auto depth_dense_tensor = model_output.toTuple()->elements()[0].toTensor().to(torch::kCPU);
  //   auto depth_dense_a = depth_dense_tensor.accessor<float, 4>();

  //   for (int h = 0; h < height_; h++)
  //     for (int w = 0; w < width_; w++){
  //       output->depth[width_ * h + w] = depth_a[0][0][h][w];
  //       output->depth_dense[width_ * h + w] = depth_dense_a[0][0][h][w];
  //     }
  
  auto depth_dense_tensor = model_output.toTensor().to(torch::kCPU);
  auto depth_dense_a = depth_dense_tensor.accessor<float, 4>();
  
  for (int h = 0; h < height_; h++) 
    for (int w = 0; w < width_; w++) {
      output->depth_dense[width_ * h + w] = depth_dense_a[0][0][h][w];
    }
  
}

DrMvsnet::DrMvsnet(const char *filename, Timer *dr_timer) {
  impl = new DrMvsnetImpl(filename, dr_timer);
}

DrMvsnet::~DrMvsnet() {
  delete impl;
}

void DrMvsnet::CallAsync(int height,
                         int width,
                         unsigned char * cur_img,
                         unsigned char **src_img,
                         float *cur_K,
                         float *cur_invK,
                         float **src_K,
                         float *cur_c2w,
                         float **src_c2ws,
                         float *sparse_depth,
                         int view_num,
                         bool debug_print) {
  impl->CallAsync(
      height,
      width,
      cur_img, 
      src_img,
      cur_K,
      cur_invK, 
      src_K,  
      cur_c2w, 
      src_c2ws,
      sparse_depth,
      view_num,
      debug_print
  );
}

DrMvsnetOutput *DrMvsnet::GetResult() {
  return impl->GetResult();
}

bool DrMvsnet::Ready() {
  return impl->Ready();
}


bool test_dr_mvsnet(DrMvsnet &model, char const *filename_inputs, bool print, int repetitions, char const *out_folder) {
    using std::cerr;
    using std::endl;
    using std::cout;

    constexpr int batch_size = 1;
    constexpr int channels = 3;

    using torch::kCPU;
    /* --- Convert Tensors to C data -- */

    /* ---  Load Input ---*/
    torch::jit::script::Module tensors = torch::jit::load(filename_inputs);

    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;

    // inputs: cur_image(b3hw), src_image(bM3hw)->RGB, cur_invK(b44), src_K(bM44), 
    // cur_world_T_cam(b44), src_world_T_cam(bM44)
    // convert image tensor to uncinged char *; cur_image(b3hw), src_image(bM3hw)
    auto cur_img_t = tensors.attr("cur_image").toTensor().to(kCPU);
    auto src_img_t = tensors.attr("src_image").toTensor().to(kCPU);
    if (cur_img_t.size(0) != batch_size || src_img_t.size(0) != batch_size) {
      cerr << "Incorrect batch size." << endl;
      return false;
    }
    if (cur_img_t.size(1) != channels || src_img_t.size(2) != channels) {
      cerr << "Incorrect channels." << endl;
      return false;
    }
    auto cur_img_a = cur_img_t.accessor<float, 4>();//b3hw
    auto src_img_a = src_img_t.accessor<float, 5>();//bM3hw
    const int height = cur_img_t.size(2);//384*512
    const int width = cur_img_t.size(3);
  
    const int src_view_num = src_img_t.size(1);
    if (print)
      cout << "View Num: " << src_view_num + 1 << endl;

    unsigned char *cur_img = (unsigned char *) malloc(sizeof(unsigned char) * height * width * channels);
    for (int h = 0; h < height; h++)
        for (int w = 0; w < width; w++) {
          // RGB -> BGR
          cur_img[channels * (width * h + w) + 0] = (unsigned char) (255.0 * cur_img_a[0][2][h][w]);
          cur_img[channels * (width * h + w) + 1] = (unsigned char) (255.0 * cur_img_a[0][1][h][w]);
          cur_img[channels * (width * h + w) + 2] = (unsigned char) (255.0 * cur_img_a[0][0][h][w]);
        }
    
    unsigned char *src_img[src_view_num];
    for (int view = 0; view < src_view_num; view++) {
      src_img[view] = (unsigned char *) malloc(sizeof(unsigned char) * height * width * channels);
      for (int h = 0; h < height; h++)
        for (int w = 0; w < width; w++) {
          src_img[view][channels * (width * h + w) + 0] = (unsigned char) (255.0 * src_img_a[0][view][2][h][w]);
          src_img[view][channels * (width * h + w) + 1] = (unsigned char) (255.0 * src_img_a[0][view][1][h][w]);
          src_img[view][channels * (width * h + w) + 2] = (unsigned char) (255.0 * src_img_a[0][view][0][h][w]);
        }
    }
    
    // convert intrinsic from tensor to float*; cur_invK(b44), src_K(bM44)
    auto cur_K_t = tensors.attr("cur_K0").toTensor().to(kCPU);
    auto cur_K_a = cur_K_t.accessor<float, 3>();
    float *cur_K = (float *) malloc(sizeof(float) * 4 * 4);
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++)
        if (i==3 && j==3)
          cur_K[i * 4 + j] = 1.0;
        else if (i==3 || j==3)
          cur_K[i * 4 + j] = 0.0;
        else
          cur_K[i * 4 + j] = cur_K_a[0][i][j];

    auto cur_invK_t = tensors.attr("cur_invK1").toTensor().to(kCPU);
    auto cur_invK_a = cur_invK_t.accessor<float, 3>();
    float *cur_invK = (float *) malloc(sizeof(float) * 4 * 4);
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++)
        if (i==3 && j==3)
          cur_invK[i * 4 + j] = 1.0;
        else if (i==3 || j==3)
          cur_invK[i * 4 + j] = 0.0;
        else
          cur_invK[i * 4 + j] = cur_invK_a[0][i][j];

    auto src_K_t = tensors.attr("src_K1").toTensor().to(kCPU);
    auto src_K_a = src_K_t.accessor<float, 4>();
    float *src_K[src_view_num];
    for (int view = 0; view < src_view_num; view++)
      src_K[view] = (float *) malloc(sizeof(float) * 4 * 4);
    for (int view = 0; view < src_view_num; view++)
      for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
          if (i==3 && j==3)
            src_K[view][i * 4 + j] = 1.0;
          else if (i==3 || j==3)
            src_K[view][i * 4 + j] = 0.0;
          else
            src_K[view][i * 4 + j] = src_K_a[0][view][i][j];
        
    // convert c2w from tensor to float*;cur_world_T_cam(b44), src_world_T_cam(bM44)
    auto cur_c2w_t = tensors.attr("cur_world_T_cam").toTensor().to(kCPU);
    auto cur_c2w_a = cur_c2w_t.accessor<float, 3>();
    float *cur_c2w = (float *) malloc(sizeof(float) * 4 * 4);
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++)
        cur_c2w[i * 4 + j] = cur_c2w_a[0][i][j];
    
    auto src_c2w_t = tensors.attr("src_world_T_cam").toTensor().to(kCPU);
    auto src_c2w_a = src_c2w_t.accessor<float, 4>();
    float **src_c2ws = (float **) malloc(sizeof(float *) * src_view_num);
    for (int view = 0; view < src_view_num; view++) {
      src_c2ws[view] = (float *) malloc(sizeof(float) * 4 * 4);
      for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
          src_c2ws[view][i * 4 + j] = src_c2w_a[0][view][i][j];
    }

    // sparse depth
    auto sparse_depth_t = tensors.attr("sparse_depth").toTensor().to(kCPU);
    auto sparse_depth_a = sparse_depth_t.accessor<float, 4>();
    float *sparse_depth = (float *) malloc(sizeof(float) * (width/2) * (height/2));
    for (int h = 0; h < height/2; h++)
        for (int w = 0; w < width/2; w++)
          sparse_depth[width/2 * h + w] = (float) sparse_depth_a[0][0][h][w];

    // get reference depth for evaluation
    auto depth_gt = tensors.attr("depth_gt").toTensor().to(kCPU);
    auto depth_ref = tensors.attr("upsampled_depth_pred").toTensor().to(kCPU);
    auto error_loaded = torch::mean(torch::abs(depth_gt - depth_ref)).item().toFloat();

    double elapsed1 = 0.0;
    double elapsed2 = 0.0;
    double elapsed3 = 0.0;

    bool correct = true;

    int warmup = (repetitions == 1) ? 0 : 5;

    for (int rep = 0; rep < repetitions + warmup; rep++) {
      if (rep == warmup) {
        elapsed1 = 0.0;
        elapsed2 = 0.0;
        elapsed3 = 0.0;
      }
      auto start = std::chrono::high_resolution_clock::now();
      model.CallAsync(
          height,
          width,
          cur_img, 
          src_img,
          cur_K,
          cur_invK, 
          src_K,  
          cur_c2w, 
          src_c2ws,
          sparse_depth,
          src_view_num,
          false
      );
      elapsed1 += std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::high_resolution_clock::now() - start).count();

      start = std::chrono::high_resolution_clock::now();
      bool ready = model.Ready();
      elapsed2 += std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::high_resolution_clock::now() - start).count();

      if (print && ready)
        std::cout << "Was ready directly. Quite unexpected. Debug. " << std::endl;

      start = std::chrono::high_resolution_clock::now();
      auto output = model.GetResult();
      elapsed3 += std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::high_resolution_clock::now() - start).count();
      
      auto depth_out = torch::from_blob(output->depth_dense, {height, width});

      auto error_depth = torch::mean(torch::abs(depth_out - depth_ref)).item().toFloat();
      auto error_depth_gt = torch::mean(torch::abs(depth_out - depth_gt)).item().toFloat();

      const double atol = 1e-2;
      auto correct_depth = error_depth < atol;
      if (print) {
        cout << "Correctness:" << endl;
        cout << "\tDepth correct     : " << correct_depth << ", error (depth_pred vs depth_loaded): " << error_depth << endl;
        cout << "\tDepth correct     : " << correct_depth << ", error (depth_pred vs depth_gt): " << error_depth_gt << endl;
        cout << "\tDepth error(depth_gt vs depth_loaded): " << error_loaded << endl;
      }

      correct &= correct_depth;

      if (out_folder && rep == 0) {
        std::string out_name = std::string(out_folder) + "pred_outputs.pt";
        cout << "Writing Result to: " << out_name << endl;

        auto bytes = torch::jit::pickle_save(depth_out);
        std::ofstream fout(out_name, std::ios::out | std::ios::binary);
        fout.write(bytes.data(), bytes.size());
        fout.close();
      }

      delete output;
    }

    if (print) {
      cout << "Performance:" << endl;
      cout << "\tCallAsync     : " << (double) elapsed1 / (1000.0 * repetitions) << " ms" << endl;
      cout << "\tReady         : " << (double) elapsed2 / (1000.0 * repetitions) << " ms" << endl;
      cout << "\tGetResult     : " << (double) elapsed3 / (1000.0 * repetitions) << " ms" << endl;
    }

    if (correct) {
      if (print)
        cout << "All looks good!" << endl;
      return true;
    } else {
      if (print)
        cout << "There has been an error. Do not use the model." << endl;
      return false;
    }
}
