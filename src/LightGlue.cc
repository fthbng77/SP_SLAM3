#include "LightGlue.h"
#include <iostream>

namespace ORB_SLAM3
{

LightGlue::LightGlue(const std::string &model_path, bool use_cuda, bool use_fp16)
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
        std::cout << "LightGlue model loaded from: " << model_path
                  << " (device: " << (mDevice.is_cuda() ? "CUDA" : "CPU")
                  << ", FP16: " << (mbFP16 && mDevice.is_cuda() ? "ON" : "OFF")
                  << ")" << std::endl;
    } catch (const c10::Error &e) {
        std::cerr << "Failed to load LightGlue model: " << e.what() << std::endl;
        mbLoaded = false;
    }
}

torch::Tensor LightGlue::normalizeKeypoints(const std::vector<cv::KeyPoint> &kpts,
                                              const cv::Size &image_size)
{
    auto opts = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor tensor = torch::zeros({(long)kpts.size(), 2}, opts);
    auto accessor = tensor.accessor<float, 2>();

    float w = (float)image_size.width;
    float h = (float)image_size.height;

    for (size_t i = 0; i < kpts.size(); i++) {
        // Normalize to [-1, 1]
        accessor[i][0] = 2.0f * kpts[i].pt.x / w - 1.0f;
        accessor[i][1] = 2.0f * kpts[i].pt.y / h - 1.0f;
    }

    return tensor;
}

torch::Tensor LightGlue::descriptorsToTensor(const cv::Mat &desc)
{
    auto tensor = torch::from_blob(
        (void*)desc.data,
        {desc.rows, desc.cols},
        torch::kFloat32
    ).clone();

    return tensor;
}

std::vector<LightGlueMatch> LightGlue::match(
    const std::vector<cv::KeyPoint> &kpts0,
    const cv::Mat &desc0,
    const std::vector<cv::KeyPoint> &kpts1,
    const cv::Mat &desc1,
    const cv::Size &image_size)
{
    std::vector<LightGlueMatch> matches;

    if (!mbLoaded || kpts0.empty() || kpts1.empty())
        return matches;

    // Serialize all GPU inference calls across threads
    std::lock_guard<std::mutex> lock(getInferenceMutex());

    torch::NoGradGuard no_grad;

    // Prepare inputs
    auto kp0_tensor = normalizeKeypoints(kpts0, image_size).unsqueeze(0).to(mDevice);  // [1, N, 2]
    auto kp1_tensor = normalizeKeypoints(kpts1, image_size).unsqueeze(0).to(mDevice);  // [1, M, 2]
    auto desc0_tensor = descriptorsToTensor(desc0).unsqueeze(0).to(mDevice);            // [1, N, 256]
    auto desc1_tensor = descriptorsToTensor(desc1).unsqueeze(0).to(mDevice);            // [1, M, 256]

    if (mbFP16 && mDevice.is_cuda()) {
        kp0_tensor = kp0_tensor.to(torch::kFloat16);
        kp1_tensor = kp1_tensor.to(torch::kFloat16);
        desc0_tensor = desc0_tensor.to(torch::kFloat16);
        desc1_tensor = desc1_tensor.to(torch::kFloat16);
    }

    // Forward pass
    // Expected TorchScript interface: forward(kpts0, kpts1, desc0, desc1) -> (matches, scores)
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(kp0_tensor);
    inputs.push_back(kp1_tensor);
    inputs.push_back(desc0_tensor);
    inputs.push_back(desc1_tensor);

    try {
        auto output = mModel.forward(inputs);

        // Parse output: expect a tuple of (match_indices [K, 2], match_scores [K])
        auto output_tuple = output.toTuple();
        auto match_indices = output_tuple->elements()[0].toTensor().to(torch::kCPU).to(torch::kInt32);
        auto match_scores = output_tuple->elements()[1].toTensor().to(torch::kCPU).to(torch::kFloat32);

        int num_matches = match_indices.size(0);
        matches.reserve(num_matches);

        auto idx_acc = match_indices.accessor<int, 2>();
        auto score_acc = match_scores.accessor<float, 1>();

        for (int i = 0; i < num_matches; i++) {
            LightGlueMatch m;
            m.idx0 = idx_acc[i][0];
            m.idx1 = idx_acc[i][1];
            m.score = score_acc[i];
            matches.push_back(m);
        }
    } catch (const c10::Error &e) {
        std::cerr << "LightGlue inference failed: " << e.what() << std::endl;
    }

    // Ensure all GPU operations complete before releasing mutex
    if (mDevice.is_cuda())
        torch::cuda::synchronize();

    return matches;
}

}  // namespace ORB_SLAM3
