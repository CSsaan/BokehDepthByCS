#include "DptV2MNN.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <memory>
#include <optional>

using namespace MNN;
using namespace MNN::Express;

namespace CS
{
    DepthAnythingInferenceMNN::DepthAnythingInferenceMNN(const std::string& modeltype, bool restore_size, bool use_kalman, bool use_gpu) :
        m_restore_size(restore_size), m_use_kalman(use_kalman)
    {
        if(load(modeltype, use_gpu, m_inputSize))
        {
            uint16_t w_size = m_inputSize[2];
            uint16_t h_size = m_inputSize[3];
            if (m_use_kalman)
            {
                std::cout << "[DepthAnythingInferenceMNN] [ USE Kalman ]" << std::endl;
                kalmanFilter = std::make_shared<KalmanFilter>(w_size, h_size, 1e-5, 5e-4, 0.02);
            }
            skin_mask_result = std::make_unique<float[]>(w_size * h_size);
        }

        pretreat.reset(MNN::CV::ImageProcess::create(MNN::CV::BGR, MNN::CV::RGB, mean_vals.data(), 3, norm_vals.data(), 3), MNN::CV::ImageProcess::destroy);
    }

    DepthAnythingInferenceMNN::~DepthAnythingInferenceMNN() {
        if (net) {
            net->releaseModel();
            net->releaseSession(session);
        }
    }

    bool DepthAnythingInferenceMNN::load(const std::string& modeltype, bool use_gpu, std::vector<int> &in_shape)
    {
        std::string modelPath = std::string(MODEL_PATH) + "/dpt/dpt" + modeltype + ".mnn";
        net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(modelPath.c_str()));
        if (!net) {
            std::cerr << "Failed to create interpreter from model file." << std::endl;
            return -1;
        }

        MNN::ScheduleConfig config;
        config.numThread = std::max(1u, std::thread::hardware_concurrency()); // 设置线程数
        config.type = use_gpu ? MNN_FORWARD_OPENCL : MNN_FORWARD_CPU;
        std::cout << "[DepthAnythingInferenceMNN] use: " << (use_gpu ? "GPU" : "CPU") << std::endl;

        //MNN::BackendConfig backendConfig;
        //backendConfig.precision = MNN::BackendConfig::PrecisionMode::Precision_Normal;
        //backendConfig.power = MNN::BackendConfig::PowerMode::Power_Normal;
        //backendConfig.memory = MNN::BackendConfig::MemoryMode::Memory_Normal;
        //config.backendConfig = &backendConfig;

        session = net->createSession(config);
        input_tensor = net->getSessionInput(session, nullptr);

        // print info
        float memoryUsage = 0.0f;
        net->getSessionInfo(session, MNN::Interpreter::MEMORY, &memoryUsage);
        float flops = 0.0f;
        net->getSessionInfo(session, MNN::Interpreter::FLOPS, &flops);
        int backendType[2];
        net->getSessionInfo(session, MNN::Interpreter::BACKENDS, backendType);
        printf("[DepthAnythingInferenceMNN] Session Info: memory use %f MB, flops is %f M, backendType is %d \n", memoryUsage, flops, backendType[0]);

        in_shape = input_tensor->shape();
        std::cout << "[DepthAnythingInferenceMNN] m_inputSize: [";
        for (int dim : in_shape)
        {
            std::cout << dim << ", ";
        }
        std::cout << "]" << std::endl;
        return input_tensor ? true : false;
    }

    void DepthAnythingInferenceMNN::resizeAndPadImage(const cv::Mat& image, cv::Mat& output, uint16_t targetSize, ImagePaddingInfo& paddingInfo) {
        int img_h = image.rows;
        int img_w = image.cols;
        // Determine scaling factor and calculate new dimensions
        float scale = static_cast<float>(targetSize) / std::max(img_w, img_h);
        paddingInfo.width = static_cast<uint16_t>(img_w * scale);
        paddingInfo.height = static_cast<uint16_t>(img_h * scale);
        // Resize image
        cv::resize(image, output, cv::Size(paddingInfo.width, paddingInfo.height)); // 0.358ms
        // Calculate padding sizes
        paddingInfo.widthPad = targetSize - paddingInfo.width;
        paddingInfo.heightPad = targetSize - paddingInfo.height;
        // Add padding
        cv::copyMakeBorder(output, output, // 0.08ms
            paddingInfo.heightPad / 2, paddingInfo.heightPad - paddingInfo.heightPad / 2,
            paddingInfo.widthPad / 2, paddingInfo.widthPad - paddingInfo.widthPad / 2,
            cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    }

    int DepthAnythingInferenceMNN::infer(const cv::Mat& raw_image, cv::Mat& depth, bool use_color_map, const std::string& output_path) {
        if (raw_image.empty()) {
            std::cerr << "Image is empty, please check!" << std::endl;
            return -1;
        }

        if (!net || !session || !input_tensor) {
            std::cerr << "Net, session, or input_tensor is null" << std::endl;
            return -1;
        }
        net->resizeTensor(input_tensor, m_inputSize);
        net->resizeSession(session);

        // Padding and Resize
        ImagePaddingInfo paddingInfo;
        if (use_pad) {
            cv::Mat image;
            resizeAndPadImage(raw_image, image, m_inputSize[2], paddingInfo);
            // 原始使用opencv resize的pretreat
            pretreat->convert(image.data, m_inputSize[2], m_inputSize[3], image.step[0], input_tensor);
        }
        else {
            //cv::resize(raw_image, image, cv::Size(m_inputSize[2], m_inputSize[3])); // abandoned
            //TODO: MNN resize 的 pretreat
            MNN::CV::Matrix trans;
            trans.setScale((float)(raw_image.cols - 1) / (m_inputSize[2] - 1), (float)(raw_image.rows - 1) / (m_inputSize[3] - 1));
            pretreat->setMatrix(trans);
            pretreat->convert(raw_image.data, raw_image.cols, raw_image.rows, 0, input_tensor);
        }

        // 推理
        auto start = std::chrono::high_resolution_clock::now();
        net->runSession(session);
        MNN::Tensor* tensor_depth = net->getSessionOutput(session, "depth");
        if (!tensor_depth) {
            std::cerr << "tensor_depth is null" << std::endl;
            return -1;
        }
        // 读取模型输出
        MNN::Tensor tensor_depth_host(tensor_depth, tensor_depth->getDimensionType());
        auto flag1 = tensor_depth->copyToHostTensor(&tensor_depth_host);
        float* tensor_data = nullptr;
        if (flag1) {
            tensor_data = tensor_depth_host.host<float>(); // 如果 tensor_scores 在 GPU 上，使用 tensor_scores_host
        }
        else {
            tensor_data = tensor_depth->host<float>(); // 如果 tensor_scores 在 CPU 上，使用 tensor_scores
        }
        // 卡尔曼滤波 & 反归一化 0.2ms
        if (m_use_kalman) {
            kalmanFilter->processMaskArray(static_cast<const float*>(tensor_data), skin_mask_result.get());
            cv::Mat cv_pha(m_inputSize[2], m_inputSize[3], CV_32FC1, skin_mask_result.get());
            cv::normalize(cv_pha, depth, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        }
        else {
            cv::Mat cv_pha(m_inputSize[2], m_inputSize[3], CV_32FC1, static_cast<float*>(tensor_data));
            cv::normalize(cv_pha, depth, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        }
        // 伪彩色显示
        if (use_color_map) {
            cv::applyColorMap(depth, depth, cv::COLORMAP_INFERNO);
        }
        // 去除 padding
        if (use_pad) {
            depth = depth(cv::Rect(paddingInfo.widthPad / 2, paddingInfo.heightPad / 2, paddingInfo.width, paddingInfo.height));
        }
        if (m_restore_size) {
            cv::resize(depth, depth, cv::Size(raw_image.cols, raw_image.rows));
        }
        if (depth.channels() > 1) {
            cv::cvtColor(depth, depth, cv::COLOR_BGR2GRAY);
        }
        //if (!output_path.empty()) {
        //    cv::imwrite(output_path, cv_pha);
        //}

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        std::cout << "[DepthAnythingInferenceMNN] inference time: " << duration.count() << " ms" << std::endl;

        return 0;
    }
}