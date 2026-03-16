#ifndef PLACE_RECOGNITION_H
#define PLACE_RECOGNITION_H

#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <mutex>
#include <set>

#ifdef EIGEN_MPL2_ONLY
#undef EIGEN_MPL2_ONLY
#endif

namespace ORB_SLAM3
{

class KeyFrame;

class PlaceRecognition {
public:
    PlaceRecognition(const std::string &model_path, bool use_cuda = true, bool use_fp16 = false);

    // Extract global descriptor from an image
    torch::Tensor extractDescriptor(const cv::Mat &image);

    // Add a keyframe to the database
    void add(KeyFrame *pKF, const torch::Tensor &descriptor);

    // Query N most similar keyframes, excluding connected ones
    std::vector<KeyFrame*> query(KeyFrame *pKF,
                                  const torch::Tensor &descriptor,
                                  int nCandidates,
                                  const std::set<KeyFrame*> &spConnectedKFs);

    // Remove a keyframe from the database
    void erase(KeyFrame *pKF);

    bool isLoaded() const { return mbLoaded; }

private:
    torch::jit::script::Module mModel;
    torch::Device mDevice;
    bool mbLoaded;
    bool mbFP16;

    // Database: stores descriptors for cosine similarity search
    struct Entry {
        KeyFrame* pKF;
        torch::Tensor descriptor;  // [D] normalized
    };
    std::vector<Entry> mvDatabase;
    std::mutex mMutex;

    torch::Tensor preprocessImage(const cv::Mat &image);
};

}  // namespace ORB_SLAM3

#endif // PLACE_RECOGNITION_H
