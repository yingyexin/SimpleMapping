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

#ifndef DR_MVSNET_H
#define DR_MVSNET_H

#include <memory>
#include <utility>
#include "Timer.h"


class DrMvsnetImpl;

class DrMvsnetOutput {
public:
  DrMvsnetOutput(int height, int width) : height(height), width(width) {
    depth = (float *) malloc(sizeof(float) * width * height);
    depth_dense = (float *) malloc(sizeof(float) * width * height);
  };

  ~DrMvsnetOutput() {
    free(depth);
    free(depth_dense);
  }

  float *depth;
  float *depth_dense;
  const int height;
  const int width;
};

class DrMvsnet {
public:
  explicit DrMvsnet(char const *filename, Timer *dr_timer);

  ~DrMvsnet();

  // Blocking for last input. Non-blocking for this input.
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
                 bool debug_print = false);

  // Blocking
  DrMvsnetOutput *GetResult();

  // Blocking
  void Wait();

  // Non-blocking
  bool Ready();

private:
  DrMvsnetImpl *impl;
};

bool test_dr_mvsnet(DrMvsnet &model, char const *filename_inputs, 
                bool print = false, int repetitions = 1, char const *out_folder = NULL);

#endif //DR_MVSNET_H

