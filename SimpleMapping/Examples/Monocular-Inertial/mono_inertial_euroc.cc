/**
* This file is part of SimpleMapping. 
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


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include <ctime>
#include <sstream>
#include <boost/filesystem.hpp>

#include<opencv2/core/core.hpp>

#include<System.h>
#include "ImuTypes.h"
#include "utils/Timer.h"

using namespace std;

void LoadImages(const string &strImagePath, const string &strPathTimes,
                vector<string> &vstrImages, vector<double> &vTimeStamps);

void LoadIMU(const string &strImuPath, vector<double> &vTimeStamps, vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro);

double ttrack_tot = 0;
int main(int argc, char *argv[])
{

    if(argc < 7)
    {
        cerr << endl << "Usage: ./mono_inertial_euroc path_to_vocabulary path_to_settings path_to_sequence_folder path_to_times_file path_to_save_mvs_result path_to_mvs_model b_viz" << endl;
        return 1;
    }

    // const int num_seq = (argc-3)/2;
    const int num_seq = 1; //TODO: bug of computing num_seq
    cout << "num_seq = " << num_seq << endl;
    bool bFileName= true;
    string result_path;
    result_path = string(argv[5]);
    const string mesh_folder = "result/debug/" + result_path +"/";
    if (bFileName)
    {
        cout << "result_path: " << result_path << endl;
        cout << "mesh_folder: " << mesh_folder << endl;

        boost::system::error_code ec;
        if(!boost::filesystem::exists(mesh_folder, ec)){
            bool success_created = boost::filesystem::create_directories(mesh_folder, ec);
            if(!success_created){
                cout<<"Fail to create the directory for saving mesh: "<<mesh_folder<<", msg: "<<ec.message()<<std::endl;
            }
        }
    }

    // Load all sequences:
    int seq;
    vector< vector<string> > vstrImageFilenames;
    vector< vector<double> > vTimestampsCam;
    vector< vector<cv::Point3f> > vAcc, vGyro;
    vector< vector<double> > vTimestampsImu;
    vector<int> nImages;
    vector<int> nImu;
    vector<int> first_imu(num_seq,0);

    vstrImageFilenames.resize(num_seq);
    vTimestampsCam.resize(num_seq);
    vAcc.resize(num_seq);
    vGyro.resize(num_seq);
    vTimestampsImu.resize(num_seq);
    nImages.resize(num_seq);
    nImu.resize(num_seq);

    int tot_images = 0;
    for (seq = 0; seq<num_seq; seq++)
    {
        cout << "Loading images for sequence " << seq << "...";

        string pathSeq(argv[(2*seq) + 3]);
        string pathTimeStamps(argv[(2*seq) + 4]);

        string pathCam0 = pathSeq + "/mav0/cam0/data";
        string pathImu = pathSeq + "/mav0/imu0/data.csv";

        LoadImages(pathCam0, pathTimeStamps, vstrImageFilenames[seq], vTimestampsCam[seq]);
        cout << "LOADED!" << endl;

        cout << "Loading IMU for sequence " << seq << "...";
        LoadIMU(pathImu, vTimestampsImu[seq], vAcc[seq], vGyro[seq]);
        cout << "LOADED!" << endl;

        nImages[seq] = vstrImageFilenames[seq].size();
        tot_images += nImages[seq];
        nImu[seq] = vTimestampsImu[seq].size();

        if((nImages[seq]<=0)||(nImu[seq]<=0))
        {
            cerr << "ERROR: Failed to load images or IMU for sequence" << seq << endl;
            return 1;
        }

        // Find first imu to be considered, supposing imu measurements start first

        while(vTimestampsImu[seq][first_imu[seq]]<=vTimestampsCam[seq][0])
            first_imu[seq]++;
        first_imu[seq]--; // first imu measurement to be considered

    }

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(tot_images);
    float totalTrack = 0.0;

    cout.precision(17);

    bool b_viz = string(argv[7]) == "True";
    cout<<"Whether to use the viewer: " << b_viz <<endl;

    // new: Create SLAM system. It initializes all system threads and gets ready to process frames. new: bUseViewer = false
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::IMU_MONOCULAR, b_viz, 0, result_path, argv[6]);
    float imageScale = SLAM.GetImageScale();

    double t_resize = 0.f;
    double t_track = 0.f;

    int proccIm=0;
    const auto timer_start = Timer::start();
    for (seq = 0; seq<num_seq; seq++)
    {

        // Main loop
        cv::Mat im;
        vector<ORB_SLAM3::IMU::Point> vImuMeas;
        proccIm = 0;
        for(int ni=0; ni<nImages[seq]; ni++, proccIm++)
        {
            // Read image from file
            im = cv::imread(vstrImageFilenames[seq][ni], cv::IMREAD_COLOR); //CV_LOAD_IMAGE_UNCHANGED);

            double tframe = vTimestampsCam[seq][ni];

            if(im.empty())
            {
                cerr << endl << "Failed to load image at: "
                     <<  vstrImageFilenames[seq][ni] << endl;
                return 1;
            }

            if(imageScale != 1.f)
            {
#ifdef REGISTER_TIMES
    #ifdef COMPILEDWITHC11
                std::chrono::steady_clock::time_point t_Start_Resize = std::chrono::steady_clock::now();
    #else
                std::chrono::monotonic_clock::time_point t_Start_Resize = std::chrono::monotonic_clock::now();
    #endif
#endif
                int width = im.cols * imageScale;
                int height = im.rows * imageScale;
                cv::resize(im, im, cv::Size(width, height));
#ifdef REGISTER_TIMES
    #ifdef COMPILEDWITHC11
                std::chrono::steady_clock::time_point t_End_Resize = std::chrono::steady_clock::now();
    #else
                std::chrono::monotonic_clock::time_point t_End_Resize = std::chrono::monotonic_clock::now();
    #endif
                t_resize = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t_End_Resize - t_Start_Resize).count();
                SLAM.InsertResizeTime(t_resize);
#endif
            }

            // Load imu measurements from previous frame
            vImuMeas.clear();

            if(ni>0)
            {
                // cout << "t_cam " << tframe << endl;

                while(vTimestampsImu[seq][first_imu[seq]]<=vTimestampsCam[seq][ni])
                {
                    vImuMeas.push_back(ORB_SLAM3::IMU::Point(vAcc[seq][first_imu[seq]].x,vAcc[seq][first_imu[seq]].y,vAcc[seq][first_imu[seq]].z,
                                                             vGyro[seq][first_imu[seq]].x,vGyro[seq][first_imu[seq]].y,vGyro[seq][first_imu[seq]].z,
                                                             vTimestampsImu[seq][first_imu[seq]]));
                    first_imu[seq]++;
                }
            }

    #ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    #else
            std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
    #endif

            // Pass the image to the SLAM system
            // cout << "tframe = " << std::to_string(1e9*tframe).substr(0, 19) << endl;
            SLAM.TrackMonocular(im,tframe,vImuMeas);

    #ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    #else
            std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
    #endif

#ifdef REGISTER_TIMES
            t_track = t_resize + std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t2 - t1).count();
            SLAM.InsertTrackTime(t_track);
#endif

            double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
            ttrack_tot += ttrack;
            // std::cout << "ttrack: " << ttrack << std::endl;

            vTimesTrack[ni]=ttrack;
            totalTrack += ttrack*1000;

            // Wait to load the next frame
            double T=0;
            if(ni<nImages[seq]-1)
                T = vTimestampsCam[seq][ni+1]-tframe;
            else if(ni>0)
                T = tframe-vTimestampsCam[seq][ni-1];

            // new: enlarge track waiting time: real track time < 2.0 *(frame interval)
            float timescale = b_viz ? 1.0 : 1.0;
            if(ttrack < timescale*T)
                usleep((timescale*T-ttrack)*1e6); // 1e6
        }
        if(seq < num_seq - 1)
        {
            cout << "Changing the dataset" << endl;

            SLAM.ChangeDataset();
        }
    }
    double MilliSecondsTakenHRC = Timer::end_ms(timer_start);
    printf("\nTOTAL TIMING: =================="
           "\n%d Frames (%.1f fps)"
           "\n%.2fms Tracking perframe"
           "\n%.2fms per frame; "
           "\n%.2fs total time; "
           "\n======================\n\n",
           proccIm, 1000 * proccIm / MilliSecondsTakenHRC,
           totalTrack / proccIm, 
           MilliSecondsTakenHRC / proccIm,
           MilliSecondsTakenHRC / 1000.0);

    // new: save mesh
    if (bFileName){
        SLAM.SaveRunTime(mesh_folder, proccIm, totalTrack, MilliSecondsTakenHRC);
        SLAM.SaveMesh(mesh_folder);
    }

    // Stop all threads
    SLAM.Shutdown();

    // Save camera trajectory
    if (bFileName)
    {
        // const string kf_file =  "kf_" + string(argv[argc-2]) + ".txt";
        // const string f_file =  "f_" + string(argv[argc-2]) + ".txt";
        const string f_file = "./result/debug/" + result_path + "/CameraTrajectory.txt";
        const string kf_file = "./result/debug/" + result_path + "/KeyFrameTrajectory.txt";
        const string fw_file = "./result/debug/" + result_path + "/CameraWorldTrajectory.txt";
        SLAM.SaveTrajectoryEuRoC(f_file);
        SLAM.SaveKeyFrameTrajectoryEuRoC(kf_file);
        SLAM.SaveWorldTrajectoryEuRoC(fw_file);
    }
    else
    {
        // new: save trajectory
        const string f_file = "./result/debug/V202/CameraTrajectory.txt";
        const string kf_file = "./result/debug/V202/KeyFrameTrajectory.txt";
        const string f_w_file = "./result/debug/V202/CameraWorldTrajectory.txt";
        SLAM.SaveTrajectoryEuRoC(f_file);
        SLAM.SaveKeyFrameTrajectoryEuRoC(kf_file);
        SLAM.SaveWorldTrajectoryEuRoC(f_w_file);
    }

    return 0;
}

void LoadImages(const string &strImagePath, const string &strPathTimes,
                vector<string> &vstrImages, vector<double> &vTimeStamps)
{
    ifstream fTimes;
    fTimes.open(strPathTimes.c_str());
    vTimeStamps.reserve(5000);
    vstrImages.reserve(5000);
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            vstrImages.push_back(strImagePath + "/" + ss.str() + ".png");
            double t;
            ss >> t;
            vTimeStamps.push_back(t/1e9);

        }
    }
}

void LoadIMU(const string &strImuPath, vector<double> &vTimeStamps, vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro)
{
    ifstream fImu;
    fImu.open(strImuPath.c_str());
    vTimeStamps.reserve(5000);
    vAcc.reserve(5000);
    vGyro.reserve(5000);

    while(!fImu.eof())
    {
        string s;
        getline(fImu,s);
        if (s[0] == '#')
            continue;

        if(!s.empty())
        {
            string item;
            size_t pos = 0;
            double data[7];
            int count = 0;
            while ((pos = s.find(',')) != string::npos) {
                item = s.substr(0, pos);
                data[count++] = stod(item);
                s.erase(0, pos + 1);
            }
            item = s.substr(0, pos);
            data[6] = stod(item);

            vTimeStamps.push_back(data[0]/1e9);
            vAcc.push_back(cv::Point3f(data[4],data[5],data[6]));
            vGyro.push_back(cv::Point3f(data[1],data[2],data[3]));
        }
    }
}
