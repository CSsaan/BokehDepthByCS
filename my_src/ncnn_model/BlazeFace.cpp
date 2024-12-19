#include "BlazeFace.h"

using namespace std;
using namespace MNN;
using namespace MNN::Express;

#define clip(x, y) (x < 0 ? 0 : (x > y ? y : x))

namespace CS
{
    UltraFace::UltraFace(const std::string& mnn_path, int input_width, int input_length, int num_thread_, float score_threshold_, float iou_threshold_, int topk_, bool use_gpu) {
        num_thread = num_thread_;
        score_threshold = score_threshold_;
        iou_threshold = iou_threshold_;
        in_w = input_width;
        in_h = input_length;
        w_h_list = { in_w, in_h };

        for (auto size : w_h_list) {
            std::vector<float> fm_item;
            for (float stride : strides) {
                fm_item.push_back(ceil(size / stride));
            }
            featuremap_size.push_back(fm_item);
        }

        for (auto size : w_h_list) {
            shrinkage_size.push_back(strides);
        }
        /* generate prior anchors */
        for (int index = 0; index < num_featuremap; index++) {
            float scale_w = in_w / shrinkage_size[0][index];
            float scale_h = in_h / shrinkage_size[1][index];
            for (int j = 0; j < featuremap_size[1][index]; j++) {
                for (int i = 0; i < featuremap_size[0][index]; i++) {
                    float x_center = (i + 0.5) / scale_w;
                    float y_center = (j + 0.5) / scale_h;

                    for (float k : min_boxes[index]) {
                        float w = k / in_w;
                        float h = k / in_h;
                        priors.push_back({ clip(x_center, 1), clip(y_center, 1), clip(w, 1), clip(h, 1) });
                    }
                }
            }
        }
        /* generate prior anchors finished */

        num_anchors = priors.size();

        ultraface_interpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(mnn_path.c_str()));
        MNN::ScheduleConfig config;
        config.numThread = num_thread;
        if (use_gpu) {
            std::cout << "UltraFace use: GPU" << std::endl;
            config.type = static_cast<MNNForwardType>(MNN_FORWARD_OPENCL);
        }
        else {
            std::cout << "UltraFace use: CPU" << std::endl;
        }

        MNN::BackendConfig backendConfig;
        backendConfig.precision = MNN::BackendConfig::PrecisionMode::Precision_Normal;
        backendConfig.power = MNN::BackendConfig::PowerMode::Power_Normal;
        backendConfig.memory = MNN::BackendConfig::MemoryMode::Memory_Normal;
        config.backendConfig = &backendConfig;

        ultraface_session = ultraface_interpreter->createSession(config);
        input_tensor = ultraface_interpreter->getSessionInput(ultraface_session, nullptr);

        pretreat.reset(MNN::CV::ImageProcess::create(MNN::CV::BGR, MNN::CV::RGB, mean_vals, 3, norm_vals, 3));
    }

    UltraFace::~UltraFace() {
        ultraface_interpreter->releaseModel();
        ultraface_interpreter->releaseSession(ultraface_session);
    }

    cv::Rect_<float> UltraFace::detectLargest(cv::Mat& raw_image)
    {
        vector<FaceInfo> face_list;
        detectAll(raw_image, face_list);
        cv::Rect_<float> largestFace;
        float maxArea = 0.0f;

        for (const auto& face : face_list) {
            // 将坐标归一化为 0.0-1.0 的范围
            float x1_norm = static_cast<float>(face.x1) / raw_image.cols;
            float y1_norm = static_cast<float>(face.y1) / raw_image.rows;
            float x2_norm = static_cast<float>(face.x2) / raw_image.cols;
            float y2_norm = static_cast<float>(face.y2) / raw_image.rows;
            cv::Rect_<float> currentFace(x1_norm, y1_norm, x2_norm - x1_norm, y2_norm - y1_norm);
            // cv::Rect currentFace(face.x1, face.y1, face.x2 - face.x1, face.y2 - face.y1);

            float area = currentFace.width * currentFace.height;
            if (area > maxArea) {
                maxArea = area;
                largestFace = currentFace;
            }
        }

        return largestFace;
    }

    int UltraFace::detectAll(cv::Mat& raw_image, std::vector<FaceInfo>& face_list) {
        if (raw_image.empty()) {
            std::cout << "image is empty ,please check!" << std::endl;
            return -1;
        }

        image_h = raw_image.rows;
        image_w = raw_image.cols;
        cv::Mat image;
        cv::resize(raw_image, image, cv::Size(in_w, in_h));

        ultraface_interpreter->resizeTensor(input_tensor, { 1, 3, in_h, in_w });
        ultraface_interpreter->resizeSession(ultraface_session);
        
        pretreat->convert(image.data, in_w, in_h, image.step[0], input_tensor);

        auto start = std::chrono::high_resolution_clock::now();


        // run network
        ultraface_interpreter->runSession(ultraface_session);

        // get output data

        string scores = "scores";
        string boxes = "boxes";
        MNN::Tensor* tensor_scores = ultraface_interpreter->getSessionOutput(ultraface_session, scores.c_str());
        MNN::Tensor* tensor_boxes = ultraface_interpreter->getSessionOutput(ultraface_session, boxes.c_str());

        MNN::Tensor tensor_scores_host(tensor_scores, tensor_scores->getDimensionType());
        auto flag1 = tensor_scores->copyToHostTensor(&tensor_scores_host);
        MNN::Tensor tensor_boxes_host(tensor_boxes, tensor_boxes->getDimensionType());
        auto flag2 = tensor_boxes->copyToHostTensor(&tensor_boxes_host);
        // std::cout << "flag1, flag2(true for GPU, false for CPU): " << flag1 << ", " << flag2 << std::endl;

        std::vector<FaceInfo> bbox_collection;

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        std::cout << "UltraFace inference time: " << duration.count() << " ms" << std::endl;

        // generateBBox(bbox_collection, tensor_scores, tensor_boxes);
        // 根据 flag1 和 flag2 决定传入的张量
        if (flag1) {
            generateBBox(bbox_collection, &tensor_scores_host, &tensor_boxes_host); // 如果 tensor_scores 在 GPU 上，使用 tensor_scores_host
        }
        else {
            generateBBox(bbox_collection, tensor_scores, tensor_boxes); // 如果 tensor_scores 在 CPU 上，使用 tensor_scores
        }
        nms(bbox_collection, face_list);
        return 0;
    }

    void UltraFace::generateBBox(std::vector<FaceInfo>& bbox_collection, MNN::Tensor* scores, MNN::Tensor* boxes) {
        if (scores == nullptr || boxes == nullptr) {
            std::cerr << "Error: scores or boxes tensor is null." << std::endl;
            return;
        }
        for (int i = 0; i < num_anchors; i++) {
            if (scores->host<float>()[i * 2 + 1] > score_threshold) {
                FaceInfo rects;
                float x_center = boxes->host<float>()[i * 4] * center_variance * priors[i][2] + priors[i][0];
                float y_center = boxes->host<float>()[i * 4 + 1] * center_variance * priors[i][3] + priors[i][1];
                float w = exp(boxes->host<float>()[i * 4 + 2] * size_variance) * priors[i][2];
                float h = exp(boxes->host<float>()[i * 4 + 3] * size_variance) * priors[i][3];

                rects.x1 = clip(x_center - w / 2.0, 1) * image_w;
                rects.y1 = clip(y_center - h / 2.0, 1) * image_h;
                rects.x2 = clip(x_center + w / 2.0, 1) * image_w;
                rects.y2 = clip(y_center + h / 2.0, 1) * image_h;
                rects.score = clip(scores->host<float>()[i * 2 + 1], 1);
                bbox_collection.push_back(rects);
            }
        }
    }

    void UltraFace::nms(std::vector<FaceInfo>& input, std::vector<FaceInfo>& output, NMSType type) {
        std::sort(input.begin(), input.end(), [](const FaceInfo& a, const FaceInfo& b) { return a.score > b.score; });

        int box_num = input.size();

        std::vector<int> merged(box_num, 0);

        for (int i = 0; i < box_num; i++) {
            if (merged[i])
                continue;
            std::vector<FaceInfo> buf;

            buf.push_back(input[i]);
            merged[i] = 1;

            float h0 = input[i].y2 - input[i].y1 + 1;
            float w0 = input[i].x2 - input[i].x1 + 1;

            float area0 = h0 * w0;

            for (int j = i + 1; j < box_num; j++) {
                if (merged[j])
                    continue;

                float inner_x0 = input[i].x1 > input[j].x1 ? input[i].x1 : input[j].x1;
                float inner_y0 = input[i].y1 > input[j].y1 ? input[i].y1 : input[j].y1;

                float inner_x1 = input[i].x2 < input[j].x2 ? input[i].x2 : input[j].x2;
                float inner_y1 = input[i].y2 < input[j].y2 ? input[i].y2 : input[j].y2;

                float inner_h = inner_y1 - inner_y0 + 1;
                float inner_w = inner_x1 - inner_x0 + 1;

                if (inner_h <= 0 || inner_w <= 0)
                    continue;

                float inner_area = inner_h * inner_w;

                float h1 = input[j].y2 - input[j].y1 + 1;
                float w1 = input[j].x2 - input[j].x1 + 1;

                float area1 = h1 * w1;

                float score;

                score = inner_area / (area0 + area1 - inner_area);

                if (score > iou_threshold) {
                    merged[j] = 1;
                    buf.push_back(input[j]);
                }
            }
            switch (type) {
            case NMSType::HardNMS: {
                output.push_back(buf[0]);
                break;
            }
            case NMSType::BlendingNMS: {
                float total = 0;
                for (int i = 0; i < buf.size(); i++) {
                    total += exp(buf[i].score);
                }
                FaceInfo rects;
                memset(&rects, 0, sizeof(rects));
                for (int i = 0; i < buf.size(); i++) {
                    float rate = exp(buf[i].score) / total;
                    rects.x1 += buf[i].x1 * rate;
                    rects.y1 += buf[i].y1 * rate;
                    rects.x2 += buf[i].x2 * rate;
                    rects.y2 += buf[i].y2 * rate;
                    rects.score += buf[i].score * rate;
                }
                output.push_back(rects);
                break;
            }
            default: {
                printf("wrong type of nms.");
                exit(-1);
            }
            }
        }
    }
}
