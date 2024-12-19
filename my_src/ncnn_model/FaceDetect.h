#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

using namespace cv;
using namespace std;

namespace CS
{
    class FaceDetection {
    public:
        FaceDetection(const string& model_path = "libfacedetection/face_detection_yunet_2023mar.onnx",
            int backend = cv::dnn::DNN_BACKEND_DEFAULT, int target = cv::dnn::DNN_TARGET_CPU,
            float score_threshold = 0.6f, float nms_threshold = 0.3f, int top_k = 5000);

        cv::Ptr<cv::FaceDetectorYN> load_model();
        bool str2bool(const string& v);
        Mat visualize(const Mat& image, const Mat& faces, bool print_flag = false, double fps = 0.0);
        Rect find_largest_face(const Mat& faces);
        Mat detect_faces(const Mat& image);
        pair<float, float> normalize_coordinates(int coordx, int coordy, int image_width, int image_height);
        // 修改返回类型为 cv::Rect_<float>
        cv::Rect_<float> process_image(const Mat& image, bool save_result = false, bool visualize_result = false, bool normalize = false);

    private:
        string model_path;
        int backend;
        int target;
        float score_threshold;
        float nms_threshold;
        int top_k;
        cv::Ptr<cv::FaceDetectorYN> yunet;
    };
}

//int main() {
//    FaceDetection face_detector;
//    Mat image = imread("input.jpg"); // 读取输入图像
//    cv::Rect_<float> result = face_detector.process_image(image, true, true); // 处理图像并可视化结果
//
//    // 输出归一化坐标
//    cout << "Normalized coordinates: " << result.x << ", " << result.y << ", "
//         << result.width << ", " << result.height << endl;
//
//    return 0;
//}