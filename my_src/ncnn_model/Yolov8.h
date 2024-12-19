#pragma once

#include <iostream>
#include<vector>
#include<stdio.h>
#include<algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <memory>
#include <vector>
#include <algorithm>
#include "layer.h"
#include "net.h"

#include <float.h>
#include <stdio.h>
using namespace std;


namespace CS
{
    class YOLOV8
    {
        struct Object
        {
            cv::Rect_<float> rect;
            int label;
            float prob;
        };

    private:
        float intersection_area(const Object& a, const Object& b);
        void qsort_descent_inplace(std::vector<Object>& objects, int left, int right);
        void qsort_descent_inplace(std::vector<Object>& objects);
        void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold, bool agnostic = false);
        float sigmoid(float x);
        float clampf(float d, float min, float max);
        void parse_yolov8_detections(float* inputs, float confidence_threshold, int num_channels, int num_anchors, int num_labels, int infer_img_width, int infer_img_height, std::vector<Object>& objects);
        int detect_yolov8(const cv::Mat& bgr, std::vector<Object>& objects);
        void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects, cv::Rect& max_rect, bool draw = false);


    public:
        int run(const cv::Mat& image, cv::Rect& max_rect, bool draw = false);
    };
}


//     ncnnoptimize.exe dptv2_s.param dptv2_s.bin dptv2_s-opt.param dptv2_s-opt.bin 0
//     ncnn2table.exe dptv2_s-opt.param dptv2_s-opt.bin imagelist.txt dptv2_s.table mean=[123,117,103] norm=[0.017,0.017,0.017] shape=[518,518,3] pixel=BGR thread=4 method=kl
//     ncnn2int8.exe dptv2_s-opt.param dptv2_s-opt.bin dptv2_s-int8.param dptv2_s-int8.bin dptv2_s.table
//     ncnnoptimize.exe dptv2_s.param dptv2_s.bin dptv2_s_fp16.param dptv2_s_fp16.bin 1
