#pragma once

#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>

#include <MNN/ImageProcess.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/Executor.hpp>
#include <cv/cv.hpp>

#include "KalmanFilter.h" // 假设 KalmanFilter 类在这个头文件中定义

namespace CS {
    struct ImagePaddingInfo {
        uint16_t width;   // 缩放后的宽度
        uint16_t height;  // 缩放后的高度
        uint16_t widthPad; // 填充宽度
        uint16_t heightPad; // 填充高度
    };

    class DepthAnythingInferenceMNN {
    public:
        // 构造函数
        DepthAnythingInferenceMNN(const std::string& modeltype, bool restore_size, bool use_kalman=false, bool use_gpu=true);
        ~DepthAnythingInferenceMNN();
        // 推理函数
        int infer(const cv::Mat& image, cv::Mat& depth, bool use_color_map, const std::string& output_path = "");

    private:
        std::shared_ptr<MNN::CV::ImageProcess> pretreat;
        bool use_pad = false;  // 前后处理 等比例缩放
        bool m_use_kalman;    // 是否使用卡尔曼滤波
        bool m_restore_size;  // 是否恢复原始大小
        MNN::Tensor* input_tensor = nullptr;
        std::vector<int> m_inputSize; // 获取模型输入尺寸
        std::shared_ptr<MNN::Interpreter> net; // MNN 解释器
        MNN::Session* session; // MNN 会话
        std::shared_ptr<KalmanFilter> kalmanFilter; // 卡尔曼滤波器
        std::unique_ptr<float[]> skin_mask_result;  // 皮肤掩膜结果

        // 归一化参数
        const std::array<float, 3> mean_vals = { 0.485f, 0.456f, 0.406f }; // 均值
        const std::array<float, 3> norm_vals = { 0.01712475f, 0.0175f, 0.01742919f }; // 标准差
        
        bool load(const std::string& modeltype, bool use_gpu, std::vector<int>& in_shape); // 加载模型
        void resizeAndPadImage(const cv::Mat& image, cv::Mat& output, uint16_t targetSize, ImagePaddingInfo& paddingInfo);
    };

} // namespace CS
