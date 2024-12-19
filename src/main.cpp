
#include <iostream>
#include <chrono>
#include <iomanip>
#include<direct.h>
#include <sys/stat.h>

#include "Yolov8.h"
#include "DptV2.h"
#include "MyBokeh.hpp"
#include "myini.hpp"
#include "KalmanFilter.h"
#include "FaceDetect.h"
#include "BlazeFace.h"
#include "DptV2MNN.h"

using namespace std;
using namespace CS;

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#endif // !STB_IMAGE_IMPLEMENTATION

#define MODEL_WH  518 // 518  420

namespace CS
{
    class MyApplication {
    private:
        // shared_ptr<YOLOV8> yolov8;
        // shared_ptr<DepthAnythingInferenceNCNN> depthInfer;
        shared_ptr<DepthAnythingInferenceMNN> depthInfer_mnn;
        std::unique_ptr<MyBokeh> selectedApp;
        std::shared_ptr<KalmanFilter> kalmanFilter;
        // std::shared_ptr<FaceDetection> faceDetection;
        std::shared_ptr <UltraFace> ultraface;

        const std::string iniFilename = "config.ini";
        std::unique_ptr<IniFile> ini;
        cv::VideoWriter videoWriter;
        cv::VideoCapture cap;
        cv::Rect_<float> max_rect;
        int median_value;
        float focus_dis;

        int medianMat(const cv::Mat& Input) {
            // 确保输入是单通道8位无符号整型图像
            CV_Assert(Input.type() == CV_8UC1);
            int nVals = 256; // 8位图像的像素值范围是0到255
            // 计算直方图
            float range[] = { 0, nVals };
            const float* histRange = { range };
            bool uniform = true;
            bool accumulate = false;
            cv::Mat hist;
            cv::calcHist(&Input, 1, 0, cv::Mat(), hist, 1, &nVals, &histRange, uniform, accumulate);
            // 计算累积分布函数 (CDF)
            cv::Mat cdf;
            hist.copyTo(cdf);
            for (int i = 1; i < nVals; i++) {
                cdf.at<float>(i) += cdf.at<float>(i - 1);
            }
            cdf /= Input.total();
            // 计算中值
            int medianVal = 0;
            for (int i = 0; i < nVals; i++) {
                if (cdf.at<float>(i) >= 0.5) {
                    medianVal = i;
                    break;
                }
            }
            return medianVal; // 返回中值
        }

        bool checkAndCreateDirectory(const std::string& dirPath)
        {
            struct stat info;
            if (stat(dirPath.c_str(), &info) != 0) // 检查文件夹是否存在
            {
                if (_mkdir(dirPath.c_str()) == 0) {
                    std::cout << "[checkAndCreateDirectory] Directory created: " << dirPath << std::endl;
                    return true;
                }
                else {
                    std::cerr << "[checkAndCreateDirectory] Error creating directory: " << dirPath << std::endl;
                    return false;
                }
            }
            else if (info.st_mode & S_IFDIR) {
                return true;
            }
            else {
                std::cerr << "[checkAndCreateDirectory] Path exists but is not a directory: " << dirPath << std::endl;
                return false;
            }
        }


    public:
        MyApplication() {
            // ---------------- ini open video --------------------
            ini = make_unique<IniFile>(iniFilename);
            ini->load();
            std::string use_kalman = ini->getValue("use_kalman");
            bool m_use_kalman = (std::stoi(use_kalman) != 0);
            // -------------------- object ------------------------
            // yolov8 = make_shared<YOLOV8>();
            // depthInfer = make_shared<DepthAnythingInferenceNCNN>("v2_s", MODEL_WH, true, true);
            depthInfer_mnn = make_shared<DepthAnythingInferenceMNN>("v2_s" + std::to_string(MODEL_WH), MODEL_WH, false, m_use_kalman, true);
            kalmanFilter = std::make_shared<KalmanFilter>(MODEL_WH, 1e-5, 5e-4, 0.02);
            // faceDetection = std::make_shared<FaceDetection>(MODEL_PATH "/libfacedetection/face_detection_yunet_2023mar.onnx");
            ultraface = std::make_shared<UltraFace>(MODEL_PATH "/libfacedetection/RFB-320-quantINT8.mnn", 320, 240, std::thread::hardware_concurrency(), 0.65, 0.3, -1, false);
            selectedApp = std::make_unique<MyBokeh>();
            // ------------------ video path ----------------------
            const char* video_path = ASSERT_DIR "/video/12.mp4";
            std::string ini_video_path = ini->getValue("video_path");
            if (!ini_video_path.empty()) {
                video_path = ini_video_path.c_str();
            }
            std::cout << "ini video_path: " << ini_video_path << ", use video_path: " << video_path << std::endl;
            cap = cv::VideoCapture(video_path);
            // ------------------ write video ---------------------
            int video_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
            int video_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
            double video_fps = cap.get(cv::CAP_PROP_FPS);
            videoWriter.open(ASSERT_DIR "/video/output.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), video_fps, cv::Size(video_width, video_height), true);
            if (!videoWriter.isOpened()) {
                std::cerr << "Error: Could not open video writer." << std::endl;
                return;
            }
        }

        void run() {
            if (!cap.isOpened()) {
                std::cerr << "Error: Could not open video." << std::endl;
                return;
            }

            cv::Mat frame, result;
            bool use_color_map = false;

            while (true) {
                cap >> frame;
                if (frame.empty()) {
                    std::cout << "Frame is empty." << std::endl;
                    break;
                }

                // 目标检测
                auto start = std::chrono::high_resolution_clock::now();
                // yolov8->run(frame, max_rect, false);                                 // (NCNN) YOLO v7 人体检测(Abandoned)
                // max_rect = faceDetection->process_image(frame, false, false, false); // (ONNX) 人脸检测
                max_rect = ultraface->detectLargest(frame);                             // (MNN) ultraface人脸检测(归一化的坐标结果)
                
                // 深度估计
                //depthInfer->infer(frame, result, use_color_map);   // (NCNN) DepthAnything V2
                depthInfer_mnn->infer(frame, result, use_color_map); // (MNN) DepthAnything V2

                // 目标框的中位数
                if (max_rect.area() > 0) {
                    std::cout << "Max face Rect: x=" << max_rect.x << ", y=" << max_rect.y << ", width=" << max_rect.width << ", height=" << max_rect.height << std::endl;
                    // cv::Mat roi = result(max_rect);
                    std::cout << "result shape: " << result.cols << ", " << result.rows << std::endl;
                    cv::Rect new_max_rect = cv::Rect(max_rect.x * result.cols, max_rect.y * result.rows, max_rect.width * result.cols, max_rect.height * result.rows);
                    cv::Mat roi = result(new_max_rect); // 反归一化坐标
                    
                    //// TODO:显示一下人脸检测结果
                    //rectangle(result, new_max_rect, Scalar(255, 255, 255), 2);
                    //roi.copyTo(result(cv::Rect(result.cols - new_max_rect.width, 0, new_max_rect.width, new_max_rect.height)));
                    //// 二值化处理
                    ////cv::Scalar lowerb(178, 178, 178); // 下阈值（BGR）
                    ////cv::Scalar upperb(191, 191, 191); // 上阈值（BGR）
                    ////cv::inRange(result, lowerb, upperb, result); // 将 R 通道在 178-191 范围内的像素置为白色
                    //cv::imshow("result", result);
                    //cv::waitKey(10);

                    if (!roi.empty()) {
                        median_value = medianMat(roi);
                        Eigen::VectorXd measurement(2);
                        measurement << median_value, 0.0;
                        Eigen::VectorXd predicted = kalmanFilter->update_s(measurement);
                        focus_dis = predicted.transpose()[0];
                        std::cout << "Median value of the rectangle with kalmanFilter: " << median_value << "->" << focus_dis << std::endl;
                    }
                    else {
                        std::cerr << "ROI is empty!" << std::endl;
                    }
                }
                else {
                    std::cout << "No Person." << std::endl;
                }

                // GL 渲染
                selectedApp->setFocusDis(focus_dis/255.0f);
                cv::resize(result, result, cv::Size(640, 360)); // 使用小纹理尺寸，默认模型输出的(518, 291)大小显示会变形（暂时原因不明）
                selectedApp->setImage(frame, result);
                selectedApp->run();

                // BUG:保存视频
                cv::Mat renderedImage;
                std::string save_video = ini->getValue("save_video");
                int _save_video = std::stoi(save_video);
                if (_save_video > 0)
                {
                    selectedApp->readRenderResult(renderedImage);
                    videoWriter.write(renderedImage);
                }
                
                // 保存图片
                std::string save_frames = ini->getValue("save_frames");
                int _save_frames = std::stoi(save_frames);
                if (_save_frames > 0)
                {
                    static int i = 0;
                    std::ostringstream filename;
                    const char* saveImgDir = ASSERT_DIR "/video/frames";
                    filename << saveImgDir << "/" << std::setw(6) << std::setfill('0') << i++ << ".png";
                    checkAndCreateDirectory(std::string(saveImgDir));
                    selectedApp->readRenderResult(renderedImage);
                    cv::imwrite(filename.str(), renderedImage);
                }

                // 计时
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> duration = end - start;
                std::cout << "Total cost time: " << duration.count() << " ms" << std::endl;
                std::cout << "----------------------------------------------" << std::endl;
            }

            cap.release();
            cv::destroyAllWindows();
            videoWriter.release();
        }
    };
}


int main(int argc, char** argv)
{
    CS::MyApplication app;
    app.run();

    return 0;
}