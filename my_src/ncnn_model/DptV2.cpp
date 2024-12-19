#include "DptV2.h"

#include <iostream>
#include <opencv2/opencv.hpp>
#include "net.h"

namespace CS
{
DepthAnythingInferenceNCNN::DepthAnythingInferenceNCNN(const char* modeltype, int heightWidth, bool restore_size, bool use_gpu):
    m_heightWidth(heightWidth), m_restore_size(restore_size)
{
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);

    kalmanFilter = std::make_shared<KalmanFilter>(heightWidth, 1e-5, 5e-4, 0.02);
    skin_mask_result = std::make_unique<float[]>(heightWidth * heightWidth);

    load(modeltype, m_heightWidth, use_gpu);
}

int DepthAnythingInferenceNCNN::load(const char* modeltype, int _target_size, bool use_gpu)
{
    net.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    net.opt = ncnn::Option();

#if NCNN_VULKAN
    net.opt.use_vulkan_compute = use_gpu;
    std::cout << "USE_GPU: " << use_gpu << std::endl;
#endif

    net.opt.num_threads = ncnn::get_big_cpu_count();
    net.opt.blob_allocator = &blob_pool_allocator;
    net.opt.workspace_allocator = &workspace_pool_allocator;

    net.opt.use_winograd_convolution = true;
    net.opt.use_sgemm_convolution = true;
    net.opt.use_packing_layout = true;
    net.opt.use_shader_pack8 = false;
    net.opt.use_image_storage = false;
    // net.opt.use_int8_inference = true;

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s/dpt/dpt%s-fp16.param", MODEL_PATH, modeltype);
    sprintf(modelpath, "%s/dpt/dpt%s-fp16.bin", MODEL_PATH, modeltype);

    net.load_param(parampath);
    net.load_model(modelpath);

    color_map = cv::Mat(_target_size, _target_size, CV_8UC3);

    return 0;
}

int DepthAnythingInferenceNCNN::infer(const cv::Mat& image, cv::Mat& depth, bool use_color_map, const std::string& output_path) {
    // std::cout << "---------------- infer --------------------" << std::endl;
    // std::cout << "origin image size: " << image.size() << std::endl;
    int img_h = image.rows;
    int img_w = image.cols;

    // pad to multiple of 32
    int w = img_w;
    int h = img_h;
    float scale = 1.0f;
    if (w > h)
    {
        scale = (float)m_heightWidth / w;
        w = m_heightWidth;
        h = h * scale;
    }
    else
    {
        scale = (float)m_heightWidth / h;
        h = m_heightWidth;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);
    // std::cout << "preResize in size: [" << in.w << " x " << in.h << " x " << in.c << "]" << std::endl;
    // pad to target_size rectangle
    int wpad = m_heightWidth - w;
    int hpad = m_heightWidth - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.0f);
    // std::cout << "prePad in_pad size: [" << in_pad.w << " x " << in_pad.h << " x " << in_pad.c << "]" << std::endl;
    in_pad.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("image", in_pad);
    ncnn::Mat out;
    ex.extract("depth", out);

    kalmanFilter->processMaskArray((const float*)out.data, skin_mask_result.get());
    // 反归一化至 0-255
    cv::Mat cv_pha(out.h, out.w, CV_32FC1, skin_mask_result.get()); // (void*)out.data
    // std::cout << "model out size: " << out.h << ", " << out.w << std::endl;
    cv::normalize(cv_pha, depth, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    if (use_color_map)
    {
        cv::applyColorMap(depth, depth, cv::ColormapTypes::COLORMAP_INFERNO);
        
    } 
    // 去 pad
    depth = depth(cv::Rect(wpad / 2, hpad / 2, w, h));

    if (m_restore_size) 
    {
        cv::resize(depth, depth, cv::Size(image.cols, image.rows));
    }
    if (depth.channels() > 1) 
    {
        cv::cvtColor(depth, depth, cv::COLOR_BGR2GRAY);
    }

    // std::cout << "postProcess depth size: " << depth.size() << std::endl;

    if (!output_path.empty()) {
        cv::imwrite(output_path, cv_pha);
    }
 
    return 0;
}
}