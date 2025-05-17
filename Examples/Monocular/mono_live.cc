#include <iostream>
#include <iomanip>
#include <chrono>
#include <Eigen/Core>                 // Eigen önce!
#include <opencv2/opencv.hpp>
#include <System.h>
#include <sophus/se3.hpp>            // Sophus SE3 kullanımı

using namespace std;

// cv::Mat → Sophus::SE3f dönüşümü
Sophus::SE3f cvMatToSophus(const cv::Mat& Tcw)
{
    Eigen::Matrix4f eig_T;
    for(int i = 0; i < 4; ++i)
        for(int j = 0; j < 4; ++j)
            eig_T(i, j) = Tcw.at<float>(i, j);
    return Sophus::SE3f(Eigen::Matrix3f(eig_T.block<3,3>(0,0)), eig_T.block<3,1>(0,3));
}

int main(int argc, char **argv)
{
    if(argc < 3)
    {
        cerr << endl << "Usage: ./mono_live path_to_vocabulary path_to_settings [trajectory_file_name]" << endl;
        return 1;
    }

    string vocab_path = argv[1];
    string settings_path = argv[2];

    bool save_traj = (argc == 4);
    string traj_filename = save_traj ? argv[3] : "";

    ORB_SLAM3::System SLAM(vocab_path, settings_path, ORB_SLAM3::System::MONOCULAR, true);

    float imageScale = SLAM.GetImageScale();  // önemli

    cv::VideoCapture cap(0);
    if(!cap.isOpened())
    {
        cerr << "Kamera açılamadı!" << endl;
        return -1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FPS, 30);

    cout << "ORB-SLAM3 başlatıldı. Çıkmak için ESC." << endl;

    vector<float> vTimesTrack;
    int frame_id = 0;

    while(true)
    {
        cv::Mat frame, gray;
        cap >> frame;
        if(frame.empty()) break;

        if(imageScale != 1.f)
        {
            int width = frame.cols * imageScale;
            int height = frame.rows * imageScale;
            cv::resize(frame, frame, cv::Size(width, height));
        }

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        double tframe = chrono::duration_cast<chrono::duration<double>>(
                            chrono::steady_clock::now().time_since_epoch()).count();

        auto t1 = chrono::steady_clock::now();
        cv::Mat Tcw_cv = SLAM.TrackMonocular(frame, tframe);
        auto t2 = chrono::steady_clock::now();

        float ttrack = chrono::duration_cast<chrono::duration<float>>(t2 - t1).count();
        float fps = 1.0f / ttrack;
        cout << "FPS: " << fixed << setprecision(2) << fps << endl;
        vTimesTrack.push_back(ttrack);

        if (!Tcw_cv.empty())
        {
            Sophus::SE3f Tcw = cvMatToSophus(Tcw_cv);
            Eigen::Vector3f trans = Tcw.translation();

            cout << fixed << setprecision(6);
            cout << "Frame " << frame_id << " | Pozisyon [x y z]: "
                 << trans.x() << " "
                 << trans.y() << " "
                 << trans.z() << endl;
        }

        cv::imshow("ORB-SLAM3 Live", frame);
        if(cv::waitKey(1) == 27) break;

        frame_id++;
    }

    SLAM.Shutdown();

    if(save_traj)
    {
        SLAM.SaveTrajectoryEuRoC(traj_filename + "_trajectory.txt");
        SLAM.SaveKeyFrameTrajectoryEuRoC(traj_filename + "_keyframes.txt");
    }
    else
    {
        SLAM.SaveTrajectoryEuRoC("CameraTrajectory.txt");
        SLAM.SaveKeyFrameTrajectoryEuRoC("KeyFrameTrajectory.txt");
    }

    return 0;
}
