/**
* This file is part of TANDEM.
* 
* Copyright 2021 Technical University of Munich and Artisense.
* Developed by Lukas Koestler <Lukas.Koestler at tum dot de>,
* for more information see <https://cvg.cit.tum.de/research/vslam/tandem>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* TANDEM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* TANDEM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with TANDEM. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef TIMER_H
#define TIMER_H

#include <chrono>
#include <string>
#include <map>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdio.h>

template <class Rep, class Period>
constexpr double cast_to_ms(const std::chrono::duration<Rep,Period>& d)
{
  return std::chrono::duration<double>(d).count() * 1e3;
}

class Timer {
  using hrc = std::chrono::high_resolution_clock;
  using time_point = hrc::time_point;
  using duration = hrc::duration;

  using string = std::string;

  class Instance {
  public:
    time_point t;
    duration dt;

    Instance(time_point const &t_in, duration const &dt_in) : t(t_in), dt(dt_in) {};

    double dt2ms() const{return cast_to_ms(dt);};
    double t2ms(time_point const& start) const{return cast_to_ms(t - start );};
  };

  using map = std::map<std::string, std::vector<Instance>>;

public:
    static time_point start(){return hrc::now();};
    static double end_ms(time_point const& start) {return cast_to_ms(hrc::now() - start); };
public:
  Timer(){
    global_start = hrc::now();
  };

  int start_timing(string const &key) {
    starts[key][counter] = hrc::now();
    counter++;
    return (counter - 1);
  };

  void end_timing(string const& key, int id, bool print=false){
    auto const& start = starts[key][id];
    instances[key].emplace_back(start, hrc::now()-start);
    starts[key].erase(id);
    if (print)
      printf(("DRMVSNET:   "+key+" %6.2f ms\n").c_str(), instances[key].back().dt2ms());
  };

  std::pair<double, int> mean_timing(string const& key){
    double sum = 0;
    int count = 0;

    for (auto const& inst: instances[key]){
      if(inst.dt2ms() < 900){
        sum += inst.dt2ms();
        count += 1;
      }
    }

    if (count == 0)
      return std::make_pair(0.0, count);
    return std::make_pair(sum / (double)count, count);
  };

  void write_to_file(string const& filename){
    std::ofstream myfile;
    myfile.open(filename, std::fstream::app);
    for (auto const& key_vec : instances){
      for (auto const& inst: key_vec.second){
        myfile << key_vec.first;
        myfile << " "<< std::setprecision (15) << inst.t2ms(global_start);
        myfile << " "<< std::setprecision (15) << inst.dt2ms() << "\n";
      }
      auto ans = mean_timing(key_vec.first);
      myfile << key_vec.first << " mean_timing: "<< std::setprecision (15) << ans.first << "\n";
      myfile << key_vec.first << " mean instance: "<< std::setprecision (15) << ans.second << "\n";
    }
    myfile.close();
  }

private:
  int counter = 0;
  std::map<std::string, std::map<int, time_point>> starts;
  std::map<std::string, std::map<int, duration>> cum_times;
  std::map<std::string, std::vector<Instance>> instances;
  time_point global_start;
};


#endif //TIMER_H
