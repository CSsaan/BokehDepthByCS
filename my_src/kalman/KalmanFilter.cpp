#include "KalmanFilter.h"

#include <iostream>
#include <vector>

namespace CS
{
KalmanFilter::KalmanFilter(int rows_cols, double kal_q, double kal_r, double movement) :
    m_rows_cols(rows_cols), m_kal_q(kal_q), m_kal_r(kal_r), m_movement(movement)
{
    pred_mask = MatrixXf::Zero(m_rows_cols, m_rows_cols);
    p_mask = MatrixXf::Ones(m_rows_cols, m_rows_cols);
    //skin_mask_result = std::make_unique<float[]>(m_rows_cols * m_rows_cols);

    set_param_s(0.00003, 0.5);
}

void KalmanFilter::kalman_filter_single_frame(const MatrixXf& currentMask, const MatrixXf& previousMask, MatrixXf& predictedMask, MatrixXf& covarianceMatrix, float kal_q, float kal_r, float movementThreshold)
{
    // 预测步骤
    covarianceMatrix += MatrixXf::Constant(covarianceMatrix.rows(), covarianceMatrix.cols(), kal_q); // 预测误差协方差
    // 计算当前帧和前一帧的差异
    if (!previousMask.isZero()) {
        movementLevel = (currentMask - previousMask).cwiseAbs().mean(); // 计算差异程度
    }
    // 更新步骤
    kalmanGain = covarianceMatrix.array() / (covarianceMatrix.array() + kal_r); // 卡尔曼增益
    // 根据运动的级别动态调整 K
    if (movementLevel > movementThreshold) { // 如果帧间差异较大，认为是快速运动
        kalmanGain = kalmanGain.cwiseMin(0.7).cwiseMax(0.5);  // 快速运动时 K 更大，偏向观测值 1.0, 0.8
    }
    else { // 缓慢移动时 K 更小，偏向先验估计
        kalmanGain = kalmanGain.cwiseMin(0.8).cwiseMax(0.4);
    }
    // 更新状态
    predictedMask += kalmanGain.cwiseProduct(currentMask - predictedMask);
    covarianceMatrix = (MatrixXf::Ones(covarianceMatrix.rows(), covarianceMatrix.cols()) - kalmanGain).cwiseProduct(covarianceMatrix); // 更新协方差矩阵
}

void KalmanFilter::processMaskArray(const float* mask_array, float* skin_mask_result) {
    // Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> current_mask(mask_array, m_rows_cols, m_rows_cols);
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> current_mask(mask_array, m_rows_cols, m_rows_cols);
    Eigen::MatrixXf cur_mask = current_mask.cast<float>();
    kalman_filter_single_frame(cur_mask, previous_mask, pred_mask, p_mask, m_kal_q, m_kal_r, m_movement);
    previous_mask = cur_mask;

    int rows = pred_mask.rows();
    int cols = pred_mask.cols();
    int index = 0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            skin_mask_result[index] = pred_mask(i, j) * 255;
            index++;
        }
    }
}

void KalmanFilter::set_param_s(double processNoise, double measurementNoise) {
    // 状态变量和观测值的维度
    state_dim = 4; // 状态变量
    measure_dim = 2; // 观测值

    // 初始化矩阵
    measurementMatrix = Eigen::MatrixXd(2, 4);
    measurementMatrix << 1, 0, 0, 0,
        0, 1, 0, 0;

    transitionMatrix = Eigen::MatrixXd(4, 4);
    transitionMatrix << 1, 1, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 1,
        0, 0, 0, 1;

    processNoiseCov = Eigen::MatrixXd(4, 4);
    processNoiseCov << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1;
    processNoiseCov *= processNoise;

    measurementNoiseCov = Eigen::MatrixXd(2, 2);
    measurementNoiseCov << 1, 0,
        0, 1;
    measurementNoiseCov *= measurementNoise;

    // 初始化状态
    statePost = Eigen::VectorXd::Zero(state_dim);
    statePre = Eigen::VectorXd::Zero(state_dim);
    errorCovPost = Eigen::MatrixXd::Identity(state_dim, state_dim);
    errorCovPre = Eigen::MatrixXd::Identity(state_dim, state_dim);
}

Eigen::VectorXd KalmanFilter::update_s(const Eigen::VectorXd& measurement) {
    // 预测
    statePre = transitionMatrix * statePost;
    errorCovPre = transitionMatrix * errorCovPost * transitionMatrix.transpose() + processNoiseCov;

    // 更新
    Eigen::VectorXd innovation = measurement - measurementMatrix * statePre;
    Eigen::MatrixXd innovationCov = measurementMatrix * errorCovPre * measurementMatrix.transpose() + measurementNoiseCov;
    Eigen::MatrixXd kalmanGain = errorCovPre * measurementMatrix.transpose() * innovationCov.inverse();

    statePost = statePre + kalmanGain * innovation;
    errorCovPost = (Eigen::MatrixXd::Identity(state_dim, state_dim) - kalmanGain * measurementMatrix) * errorCovPre;

    return statePre;
}

void KalmanFilter::test(void)
{
    // 示例测量值
    std::vector<float> median_depth{ 
     -0.06421595,  0.05913586, -0.04444872,  0.08913296, -0.00827998,  0.11300071,
      0.09278156,  0.02769083,  0.04002265,  0.20200996,  0.11957323,  0.18899162,
      0.11673635,  0.07050968,  0.17766315,  0.12676238,  0.10103689,  0.17015767,
      0.21179092,  0.18619394,  0.18002534,  0.27593364,  0.2110188 ,  0.20912052,
      0.24017066,  0.27471361,  0.29810982,  0.27086975,  0.20331895,  0.32723878,
      0.31817894,  0.38892057,  0.36656634,  0.2752598 ,  0.33319591,  0.3637202 ,
      0.33501952,  0.43764431,  0.33149114,  0.34984251,  0.44121941,  0.37052215,
      0.45951954,  0.36865288,  0.48962035,  0.37959548,  0.47056577,  0.50792338,
      0.43764751,  0.45975614,  0.44897393,  0.54417235,  0.57278334,  0.49509131,
      0.44030756,  0.51889536,  0.52310881,  0.53265709,  0.58313832,  0.61128931,
      0.53357381,  0.60074764,  0.64113201,  0.67119414,  0.62278647,  0.59363843,
      0.66951007,  0.63203606,  0.70842112,  0.74340142,  0.64368514,  0.78311073,
      0.733898  ,  0.79102729,  0.75323578,  0.73059048,  0.69125997,  0.83166154,
      0.83084264,  0.77293373,  0.79388269,  0.87659323,  0.80595207,  0.85341179,
      0.79755016,  0.86640996,  0.88684092,  0.82657712,  0.86686368,  0.91321947,
      0.89300712,  0.82244895,  0.9683057 ,  0.92003551,  0.90019799,  0.86651532,
      0.92937855,  0.98737387,  0.98660369,  1.09066218 };
    std::cout << "Predicted State: " << std::endl;
    for (auto & item : median_depth)
    {
        Eigen::VectorXd measurement(2);
        measurement << item, 0.0;
        Eigen::VectorXd predicted = update_s(measurement);
        std::cout << predicted.transpose()[0] << ", ";
    }
}
}