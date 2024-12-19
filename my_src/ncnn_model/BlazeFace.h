#pragma once

#include <MNN/ImageProcess.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/Executor.hpp>
#include <cv/cv.hpp>

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>

constexpr auto num_featuremap = 4;

namespace CS
{
    enum class NMSType { /* mix nms was been proposaled in paper blaze face, aims to minimize the temporal jitter*/
        HardNMS = 1,
        BlendingNMS = 2
    };
    typedef struct FaceInfo {
        float x1;
        float y1;
        float x2;
        float y2;
        float score;

    } FaceInfo;

    class UltraFace {
    public:
        UltraFace(const std::string& mnn_path, int input_width, int input_length, int num_thread_ = 4, float score_threshold_ = 0.7, float iou_threshold_ = 0.3, int topk_ = -1, bool use_gpu = true);

        ~UltraFace();

        cv::Rect_<float> detectLargest(cv::Mat& raw_image);
        int detectAll(cv::Mat& img, std::vector<FaceInfo>& face_list);

    private:
        void generateBBox(std::vector<FaceInfo>& bbox_collection, MNN::Tensor* scores, MNN::Tensor* boxes);

        void nms(std::vector<FaceInfo>& input, std::vector<FaceInfo>& output, NMSType type = NMSType::BlendingNMS);

    private:

        std::shared_ptr<MNN::Interpreter> ultraface_interpreter;
        MNN::Session* ultraface_session = nullptr;
        MNN::Tensor* input_tensor = nullptr;

        int num_thread;
        int image_w;
        int image_h;

        int in_w;
        int in_h;
        int num_anchors;

        float score_threshold;
        float iou_threshold;


        const float mean_vals[3] = { 127.0f, 127.0f, 127.0f };
        const float norm_vals[3] = { 1.0f / 128.0f, 1.0f / 128.0f, 1.0f / 128.0f };

        const float center_variance = 0.1f;
        const float size_variance = 0.2f;
        const std::vector<std::vector<float>> min_boxes = {
                {10.0f,  16.0f,  24.0f},
                {32.0f,  48.0f},
                {64.0f,  96.0f},
                {128.0f, 192.0f, 256.0f} };
        const std::vector<float> strides = { 8.0f, 16.0f, 32.0f, 64.0f };
        std::vector<std::vector<float>> featuremap_size;
        std::vector<std::vector<float>> shrinkage_size;
        std::vector<int> w_h_list;

        std::vector<std::vector<float>> priors = {};

        std::shared_ptr<MNN::CV::ImageProcess> pretreat;
    };
}
