#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <opencv2/opencv.hpp>
#include <System.h>

using namespace std;

// Ayrı thread'de kameradan sürekli kare yakala
class FrameGrabber {
public:
    FrameGrabber(cv::VideoCapture &cap) : mCap(cap), mNewFrame(false), mStop(false) {}

    void Run() {
        cv::Mat frame;
        while(!mStop) {
            mCap >> frame;
            if(frame.empty()) {
                mStop = true;
                break;
            }
            {
                lock_guard<mutex> lock(mMutex);
                frame.copyTo(mFrame);
                mNewFrame = true;
            }
        }
    }

    bool GetFrame(cv::Mat &frame) {
        lock_guard<mutex> lock(mMutex);
        if(!mNewFrame) return false;
        mFrame.copyTo(frame);
        mNewFrame = false;
        return true;
    }

    void Stop() { mStop = true; }
    bool IsStopped() { return mStop; }

private:
    cv::VideoCapture &mCap;
    cv::Mat mFrame;
    mutex mMutex;
    atomic<bool> mNewFrame;
    atomic<bool> mStop;
};

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

    // Kamera yakalama thread'ini başlat
    FrameGrabber grabber(cap);
    thread grabThread(&FrameGrabber::Run, &grabber);

    vector<float> vTimesTrack;
    int frame_id = 0;

    while(!grabber.IsStopped())
    {
        cv::Mat frame;
        if(!grabber.GetFrame(frame))
        {
            // Henüz yeni kare yok, pencereleri canlı tut
            if(cv::waitKey(1) == 27) break;
            continue;
        }

        // Zaman etiketi oluştur
        double tframe = chrono::duration_cast<chrono::duration<double>>(
                            chrono::steady_clock::now().time_since_epoch()).count();

        auto t1 = chrono::steady_clock::now();
        cv::Mat Tcw = SLAM.TrackMonocular(frame, tframe);
        auto t2 = chrono::steady_clock::now();

        float ttrack = chrono::duration_cast<chrono::duration<float>>(t2 - t1).count();
        vTimesTrack.push_back(ttrack);

        float fps = 1.0f / ttrack;

        // Pozisyon bilgisi terminale yazdır
        if (!Tcw.empty())
        {
            cv::Mat t = Tcw.rowRange(0, 3).col(3);
            cout << fixed << setprecision(6);
        }

        if(cv::waitKey(1) == 27) break;

        frame_id++;
    }

    // Kamera thread'ini durdur
    grabber.Stop();
    if(grabThread.joinable())
        grabThread.join();

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
