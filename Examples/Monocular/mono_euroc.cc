#include <iostream>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <System.h>

using namespace std;

void LoadImages(const string &strImagePath, const string &strTimestampPath,
                vector<string> &vstrImages, vector<double> &vTimestamps);

int main(int argc, char **argv)
{
    if(argc < 4)
    {
        cerr << endl << "Usage: ./mono_euroc path_to_vocabulary path_to_settings path_to_sequence_folder [path_to_timestamps]" << endl;
        return 1;
    }

    string vocab_path = argv[1];
    string settings_path = argv[2];
    string sequence_path = argv[3];

    // Timestamp dosyası verilmişse onu kullan, yoksa data.csv'den oku
    string timestamp_path;
    if(argc >= 5)
        timestamp_path = argv[4];

    // Görüntüleri yükle
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;

    if(!timestamp_path.empty())
    {
        // Harici timestamp dosyasından oku
        ifstream fTimes(timestamp_path);
        if(!fTimes.is_open())
        {
            cerr << "Timestamp dosyası açılamadı: " << timestamp_path << endl;
            return 1;
        }
        string s;
        while(getline(fTimes, s))
        {
            if(s.empty()) continue;
            long long t = stoll(s);
            vTimestamps.push_back(t / 1e9);
            vstrImageFilenames.push_back(sequence_path + "/mav0/cam0/data/" + s + ".png");
        }
    }
    else
    {
        // data.csv'den oku
        string csv_path = sequence_path + "/mav0/cam0/data.csv";
        ifstream fCSV(csv_path);
        if(!fCSV.is_open())
        {
            cerr << "data.csv açılamadı: " << csv_path << endl;
            return 1;
        }
        string line;
        while(getline(fCSV, line))
        {
            if(line.empty() || line[0] == '#') continue;
            size_t comma = line.find(',');
            if(comma == string::npos) continue;
            string ts = line.substr(0, comma);
            string filename = line.substr(comma + 1);
            // Boşlukları temizle
            filename.erase(remove(filename.begin(), filename.end(), ' '), filename.end());
            vTimestamps.push_back(stoll(ts) / 1e9);
            vstrImageFilenames.push_back(sequence_path + "/mav0/cam0/data/" + filename);
        }
    }

    int nImages = vstrImageFilenames.size();
    if(nImages == 0)
    {
        cerr << "Görüntü bulunamadı!" << endl;
        return 1;
    }
    cout << "Toplam " << nImages << " görüntü yüklendi." << endl;

    // SLAM sistemini başlat
    // Viewer disabled for benchmarking (set to true for interactive use)
    bool bUseViewer = (getenv("SLAM_NO_VIEWER") == nullptr);
    ORB_SLAM3::System SLAM(vocab_path, settings_path, ORB_SLAM3::System::MONOCULAR, bUseViewer);

    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << "Görüntüler işleniyor..." << endl;

    for(int ni = 0; ni < nImages; ni++)
    {
        // Görüntüyü oku
        cv::Mat im = cv::imread(vstrImageFilenames[ni], cv::IMREAD_UNCHANGED);
        if(im.empty())
        {
            cerr << "Görüntü okunamadı: " << vstrImageFilenames[ni] << endl;
            continue;
        }

        double tframe = vTimestamps[ni];

        auto t1 = chrono::steady_clock::now();
        SLAM.TrackMonocular(im, tframe);
        auto t2 = chrono::steady_clock::now();

        double ttrack = chrono::duration_cast<chrono::duration<double>>(t2 - t1).count();
        vTimesTrack[ni] = ttrack;

        // Dataset hızında çalışması için bekle
        double T = 0;
        if(ni < nImages - 1)
            T = vTimestamps[ni + 1] - tframe;
        else if(ni > 0)
            T = tframe - vTimestamps[ni - 1];

        if(ttrack < T)
            usleep((T - ttrack) * 1e6);
    }

    // SLAM'i durdur
    SLAM.Shutdown();

    // Tracking sürelerini hesapla
    sort(vTimesTrack.begin(), vTimesTrack.end());
    float totaltime = 0;
    for(int ni = 0; ni < nImages; ni++)
        totaltime += vTimesTrack[ni];

    cout << "-------" << endl;
    cout << "median tracking time: " << vTimesTrack[nImages / 2] << " s" << endl;
    cout << "mean tracking time: " << totaltime / nImages << " s" << endl;
    cout << "-------" << endl;

    // Trajectory kaydet
    SLAM.SaveTrajectoryEuRoC("CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryEuRoC("KeyFrameTrajectory.txt");

    return 0;
}
