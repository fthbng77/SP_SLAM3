#include <iostream>
#include <iomanip>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <System.h>

using namespace std;

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
    string traj_filename;
    if(save_traj)
        traj_filename = argv[3];

    // ORB-SLAM3 sistemini başlat
    ORB_SLAM3::System SLAM(vocab_path, settings_path, ORB_SLAM3::System::MONOCULAR, true);

    // Kamerayı başlat
    cv::VideoCapture cap(0);
    if(!cap.isOpened())
    {
        cerr << "Kamera açılamadı!" << endl;
        return -1;
    }

    // Kamera çözünürlük ve FPS ayarı
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FPS, 30);

    cout << "Gerçek zamanlı ORB-SLAM3 çalışıyor. Çıkmak için ESC'ye basın." << endl;

    vector<float> vTimesTrack;
    int frame_id = 0;

    while(true)
    {
        cv::Mat frame, gray;
        cap >> frame;
        if(frame.empty()) break;

        // Görüntüyü griye çevir
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Opsiyonel: Histogram eşitleme (görüntü çok karanlıksa aktif edebilirsin)
        // cv::equalizeHist(gray, gray);

        // Zaman etiketi oluştur
        double tframe = chrono::duration_cast<chrono::duration<double>>(
                            chrono::steady_clock::now().time_since_epoch()).count();

        auto t1 = chrono::steady_clock::now();
        cv::Mat Tcw = SLAM.TrackMonocular(frame, tframe);
        auto t2 = chrono::steady_clock::now();

        float ttrack = chrono::duration_cast<chrono::duration<float>>(t2 - t1).count();
        vTimesTrack.push_back(ttrack);

        float fps = 1.0f / ttrack;
        cout << "FPS: " << fixed << setprecision(2) << fps << endl;
        vTimesTrack.push_back(ttrack);

        // Pozisyon bilgisi terminale yazdır
        if (!Tcw.empty())
        {
            cv::Mat t = Tcw.rowRange(0, 3).col(3);
            cout << fixed << setprecision(6);
            cout << "Frame " << frame_id << " | Pozisyon [x y z]: "
                 << t.at<float>(0) << " "
                 << t.at<float>(1) << " "
                 << t.at<float>(2) << endl;
        }

        cv::imshow("ORB-SLAM3 Live", frame);  // Orijinal renkli görüntü gösterilir
        if(cv::waitKey(1) == 27) break; // ESC tuşu ile çık

        frame_id++;
    }

    // SLAM'i durdur
    SLAM.Shutdown();

    // Trajectory kaydet (isteğe bağlı)
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
