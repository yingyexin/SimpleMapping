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

#include <iostream>
#include <string>
#include <chrono>
#include "dr_mvsnet.h"

int main(int argc, const char *argv[]) {
    using std::cout;
    using std::endl;

    std::string module_path;
    std::string sample_path;
    int repetitions = 1;
    char const* out_folder = NULL;

    if (argc == 3){
        module_path = std::string(argv[1]);
        sample_path = std::string(argv[2]);
    }else if (argc == 4) {
        module_path = std::string(argv[1]);
        sample_path = std::string(argv[2]);
        repetitions = std::atoi(argv[3]);
    }else if (argc == 5){
        module_path = std::string(argv[1]);
        sample_path = std::string(argv[2]);
        repetitions = std::atoi(argv[3]);
        out_folder = argv[4];
    }else{
        std::cerr << "usage: ./dr_mvsnet_test <path-to-exported-script-module> <path-to-sample-input> [repetitions] [out_folder]" << endl;
        return -1;
    }

    auto start = std::chrono::high_resolution_clock::now();
    DrMvsnet mvsnet(module_path.c_str(), nullptr);
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - start).count();
    cout << "Loading Model: " << (double) elapsed / 1000000.0 << " s" << endl;

    bool correct = test_dr_mvsnet(mvsnet, sample_path.c_str(), true, repetitions, out_folder);
    
    if (!correct) {
        return -1;
    }

    return 0;
}
