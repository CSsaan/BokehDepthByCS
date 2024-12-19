#pragma once

#include <iostream>
#include <Eigen/Dense>
using namespace Eigen;

namespace CS
{
    class KalmanFilter {
    public:
        KalmanFilter(int rows_cols, double kal_q = 1e-5, double kal_r = 5e-4, double movement = 0.02);
        void processMaskArray(const float* mask_array, float* skin_mask_result);
        // KalmanFilter_s
        Eigen::VectorXd update_s(const Eigen::VectorXd& measurement);

    private:
        float movementLevel = 0.0f;
        int m_rows_cols;
        double m_kal_q;    // 过程噪声协方差
        double m_kal_r;    // 观测噪声协方差
        double m_movement;
        MatrixXf cur_mask;
        MatrixXf previous_mask;
        MatrixXf kalmanGain;
        MatrixXf pred_mask; // 结果
        MatrixXf p_mask;    // 初始协方差矩阵

        void kalman_filter_single_frame(const MatrixXf& currentMask, const MatrixXf& previousMask, MatrixXf& predictedMask, MatrixXf& covarianceMatrix, 
            float kal_q, float kal_r, float movementThreshold);

        // KalmanFilter_s parameters
        int state_dim;
        int measure_dim;
        Eigen::MatrixXd measurementMatrix;
        Eigen::MatrixXd transitionMatrix;
        Eigen::MatrixXd processNoiseCov;
        Eigen::MatrixXd measurementNoiseCov;
        Eigen::VectorXd statePost;
        Eigen::VectorXd statePre;
        Eigen::MatrixXd errorCovPost;
        Eigen::MatrixXd errorCovPre;
        void set_param_s(double processNoise = 0.03, double measurementNoise = 0.5);
        void test(void);
    };
}
