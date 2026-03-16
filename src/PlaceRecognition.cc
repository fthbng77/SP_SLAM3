#include "PlaceRecognition.h"
#include "KeyFrame.h"
#include <iostream>
#include <algorithm>

namespace ORB_SLAM3
{

PlaceRecognition::PlaceRecognition(const std::string &model_path, bool use_cuda, bool use_fp16)
    : mDevice(torch::kCPU), mbLoaded(false), mbFP16(use_fp16)
{
    if (use_cuda && torch::cuda::is_available())
        mDevice = torch::Device(torch::kCUDA);

    try {
        mModel = torch::jit::load(model_path, mDevice);
        mModel.eval();

        if (mbFP16 && mDevice.is_cuda())
            mModel.to(torch::kFloat16);

        mbLoaded = true;
        std::cout << "PlaceRecognition model loaded from: " << model_path
                  << " (device: " << (mDevice.is_cuda() ? "CUDA" : "CPU")
                  << ", FP16: " << (mbFP16 && mDevice.is_cuda() ? "ON" : "OFF")
                  << ")" << std::endl;
    } catch (const c10::Error &e) {
        std::cerr << "Failed to load PlaceRecognition model: " << e.what() << std::endl;
        mbLoaded = false;
    }
}

torch::Tensor PlaceRecognition::preprocessImage(const cv::Mat &image)
{
    cv::Mat img;

    // Convert to RGB float and resize to model input size (typically 320x320 or 480x640)
    if (image.channels() == 1)
        cv::cvtColor(image, img, cv::COLOR_GRAY2RGB);
    else if (image.channels() == 3)
        img = image;
    else
        cv::cvtColor(image, img, cv::COLOR_BGRA2RGB);

    cv::resize(img, img, cv::Size(320, 320));
    img.convertTo(img, CV_32FC3, 1.0 / 255.0);

    // Normalize with ImageNet mean/std
    cv::Mat channels[3];
    cv::split(img, channels);
    channels[0] = (channels[0] - 0.485f) / 0.229f;
    channels[1] = (channels[1] - 0.456f) / 0.224f;
    channels[2] = (channels[2] - 0.406f) / 0.225f;
    cv::merge(channels, 3, img);

    // HWC -> CHW -> NCHW
    auto tensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kFloat32).clone();
    tensor = tensor.permute({2, 0, 1}).unsqueeze(0).contiguous();

    return tensor;
}

torch::Tensor PlaceRecognition::extractDescriptor(const cv::Mat &image)
{
    if (!mbLoaded)
        return torch::Tensor();

    torch::NoGradGuard no_grad;

    auto input = preprocessImage(image).to(mDevice);
    if (mbFP16 && mDevice.is_cuda())
        input = input.to(torch::kFloat16);

    auto output = mModel.forward({input}).toTensor();
    output = output.to(torch::kFloat32).squeeze(0);  // [D]

    // L2 normalize
    output = output / output.norm();

    return output.to(torch::kCPU).contiguous();
}

void PlaceRecognition::add(KeyFrame *pKF, const torch::Tensor &descriptor)
{
    if (descriptor.numel() == 0) return;

    std::unique_lock<std::mutex> lock(mMutex);
    mvDatabase.push_back({pKF, descriptor});
}

void PlaceRecognition::erase(KeyFrame *pKF)
{
    std::unique_lock<std::mutex> lock(mMutex);
    mvDatabase.erase(
        std::remove_if(mvDatabase.begin(), mvDatabase.end(),
            [pKF](const Entry &e) { return e.pKF == pKF; }),
        mvDatabase.end());
}

std::vector<KeyFrame*> PlaceRecognition::query(
    KeyFrame *pKF,
    const torch::Tensor &descriptor,
    int nCandidates,
    const std::set<KeyFrame*> &spConnectedKFs)
{
    std::vector<KeyFrame*> result;
    if (descriptor.numel() == 0) return result;

    std::unique_lock<std::mutex> lock(mMutex);

    if (mvDatabase.empty()) return result;

    // Stack all database descriptors into a matrix for batch cosine similarity
    std::vector<torch::Tensor> descList;
    std::vector<KeyFrame*> kfList;

    for (auto &entry : mvDatabase) {
        if (entry.pKF == pKF || entry.pKF->isBad())
            continue;
        if (spConnectedKFs.count(entry.pKF))
            continue;
        // Only search within the same map
        if (entry.pKF->GetMap() != pKF->GetMap())
            continue;

        descList.push_back(entry.descriptor);
        kfList.push_back(entry.pKF);
    }

    if (descList.empty()) return result;

    // [N, D] x [D, 1] -> [N] cosine similarities (descriptors are already normalized)
    auto dbMatrix = torch::stack(descList);          // [N, D]
    auto similarities = torch::mv(dbMatrix, descriptor);  // [N]

    // Get top-K
    int k = std::min(nCandidates, (int)kfList.size());
    auto [values, indices] = similarities.topk(k);

    auto idx_acc = indices.accessor<int64_t, 1>();
    result.reserve(k);
    for (int i = 0; i < k; i++) {
        result.push_back(kfList[idx_acc[i]]);
    }

    return result;
}

}  // namespace ORB_SLAM3
