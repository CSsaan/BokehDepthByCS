#pragma once

#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>

#include <MNN/ImageProcess.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/Executor.hpp>
#include <cv/cv.hpp>

#include "KalmanFilter.h" // ���� KalmanFilter �������ͷ�ļ��ж���

namespace CS {
    struct ImagePaddingInfo {
        uint16_t width;   // ���ź�Ŀ��
        uint16_t height;  // ���ź�ĸ߶�
        uint16_t widthPad; // �����
        uint16_t heightPad; // ���߶�
    };

    class DepthAnythingInferenceMNN {
    public:
        // ���캯��
        DepthAnythingInferenceMNN(const std::string& modeltype, uint32_t inputSize, bool restore_size, bool use_kalman=false, bool use_gpu=true);
        ~DepthAnythingInferenceMNN();
        // ������
        int infer(const cv::Mat& image, cv::Mat& depth, bool use_color_map, const std::string& output_path = "");

    private:
        std::shared_ptr<MNN::CV::ImageProcess> pretreat;
        bool use_pad = false;  // ǰ���� �ȱ�������
        bool m_use_kalman;    // �Ƿ�ʹ�ÿ������˲�
        uint16_t m_inputSize; // �߶ȺͿ��
        bool m_restore_size;  // �Ƿ�ָ�ԭʼ��С
        MNN::Tensor* input_tensor = nullptr;
        std::shared_ptr<MNN::Interpreter> net; // MNN ������
        MNN::Session* session; // MNN �Ự
        std::shared_ptr<KalmanFilter> kalmanFilter; // �������˲���
        std::unique_ptr<float[]> skin_mask_result;  // Ƥ����Ĥ���

        // ��һ������
        const std::array<float, 3> mean_vals = { 0.485f, 0.456f, 0.406f }; // ��ֵ
        const std::array<float, 3> norm_vals = { 0.01712475f, 0.0175f, 0.01742919f }; // ��׼��
        
        int load(const std::string& modeltype, bool use_gpu); // ����ģ��
        void resizeAndPadImage(const cv::Mat& image, cv::Mat& output, uint16_t targetSize, ImagePaddingInfo& paddingInfo);
    };

} // namespace CS
