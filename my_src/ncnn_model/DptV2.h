#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include "net.h"
#include "cpu.h"

#include "KalmanFilter.h"

#define NCNN_VULKAN 1

namespace CS
{
    class DepthAnythingInferenceNCNN {
    public:
        DepthAnythingInferenceNCNN(const char* modeltype, int heightWidth, bool restore_size, bool use_gpu);
        
        int infer(const cv::Mat& image, cv::Mat& depth, bool use_color_map, const std::string& output_path = "");

    private:
        ncnn::Net net{};
        bool use_gpu{ true };
        const char* modeltype{ "v2_s" };
        ncnn::UnlockedPoolAllocator blob_pool_allocator{};
        ncnn::PoolAllocator workspace_pool_allocator{};
        std::shared_ptr<KalmanFilter> kalmanFilter;
        std::unique_ptr<float[]> skin_mask_result;

        cv::Mat color_map;
        int m_heightWidth{ 518 };
        bool m_restore_size{ false };
        const float mean_vals[3] = { 0.485f, 0.456f, 0.406f }; // mean=[0.485, 0.456, 0.406] 
        const float norm_vals[3] = { 0.01712475f, 0.0175f, 0.01742919f }; // std=[0.229, 0.224, 0.225]

        int load(const char* modeltype, int target_size, bool use_gpu);
    };
}