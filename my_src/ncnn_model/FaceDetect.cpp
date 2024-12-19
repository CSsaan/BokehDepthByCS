#include "FaceDetect.h"


namespace CS
{
    FaceDetection::FaceDetection(const string& model_path, int backend, int target, float score_threshold, float nms_threshold, int top_k): 
        model_path(model_path), backend(backend), target(target), score_threshold(score_threshold), nms_threshold(nms_threshold), top_k(top_k) 
    {
        yunet = load_model();
    }

    cv::Ptr<cv::FaceDetectorYN> FaceDetection::load_model() 
    {
        return cv::FaceDetectorYN::create(model_path, "", Size(320, 320), score_threshold, nms_threshold, top_k, backend, target);
    }

    bool FaceDetection::str2bool(const string& v) 
    {
        if (v == "true" || v == "yes" || v == "on" || v == "y" || v == "t") return true;
        if (v == "false" || v == "no" || v == "off" || v == "n" || v == "f") return false;
        throw std::invalid_argument("Invalid boolean string");
    }

    Mat FaceDetection::visualize(const Mat& image, const Mat& faces, bool print_flag, double fps) 
    {
        Mat output = image.clone();

        if (fps > 0) {
            putText(output, "FPS: " + to_string(fps), Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
        }

        for (int i = 0; i < faces.rows; ++i) {
            int x = static_cast<int>(faces.at<float>(i, 0));
            int y = static_cast<int>(faces.at<float>(i, 1));
            int width = static_cast<int>(faces.at<float>(i, 2));
            int height = static_cast<int>(faces.at<float>(i, 3));

                if (print_flag) {
                cout << "Face " << i << ", top-left coordinates: (" << x << ", " << y
                    << "), box width: " << width << ", box height: " << height << endl;
                }

            rectangle(output, Rect(x, y, width, height), Scalar(0, 255, 0), 2);
            putText(output, to_string(height), Point(x, y + 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
        }

        return output;
    }

    Rect FaceDetection::find_largest_face(const Mat& faces) 
    {
        if (faces.empty()) return Rect();

        Rect max_face;
        int max_face_area = 0;

    for (int i = 0; i < faces.rows; ++i) {
        int x = static_cast<int>(faces.at<float>(i, 0));
        int y = static_cast<int>(faces.at<float>(i, 1));
        int width = static_cast<int>(faces.at<float>(i, 2));
        int height = static_cast<int>(faces.at<float>(i, 3));
        int face_area = width * height;

            if (face_area > max_face_area) {
                max_face_area = face_area;
                max_face = Rect(x, y, width, height);
            }
        }

        return max_face;
    }

    Mat FaceDetection::detect_faces(const Mat& image) 
    {
        yunet->setInputSize(Size(image.cols, image.rows));
        Mat faces; // 使用 cv::Mat 存储检测结果
        yunet->detect(image, faces);
        return faces;
    }

    pair<float, float> FaceDetection::normalize_coordinates(int coordx, int coordy, int image_width, int image_height) 
    {
        return { static_cast<float>(coordx) / image_width, static_cast<float>(coordy) / image_height };
    }

    // 修改返回类型为 cv::Rect_<float>
    cv::Rect_<float> FaceDetection::process_image(const Mat& image, bool save_result, bool visualize_result, bool normalize)
    {
        Mat faces = detect_faces(image); // 现在返回 cv::Mat
        if (save_result) {
            Mat vis_image = visualize(image, faces, true);
            imwrite("result.jpg", vis_image);
            cout << "result.jpg saved." << endl;
        }
        if (visualize_result) {
            Mat vis_image = visualize(image, faces, true);
            namedWindow("Test Faces by CS", WINDOW_AUTOSIZE);
            imshow("Test Faces by CS", vis_image);
            waitKey(0);
        }

        Rect largest_face = find_largest_face(faces);
        if (largest_face.area() > 0) {
            if (normalize)
            {
                float x_norm, y_norm, w_norm, h_norm;
                tie(x_norm, y_norm) = normalize_coordinates(largest_face.x, largest_face.y, image.cols, image.rows);
                tie(w_norm, h_norm) = normalize_coordinates(largest_face.width, largest_face.height, image.cols, image.rows);
                return cv::Rect_<float>(x_norm, y_norm, w_norm, h_norm); // 使用 cv::Rect_<float> 返回
            } 
            else {
                return cv::Rect_<float>(largest_face.x, largest_face.y, largest_face.width, largest_face.height);
            }
        }
        else {
            return cv::Rect_<float>(0, 0, 0, 0); // No face detected
        }
    }
}
