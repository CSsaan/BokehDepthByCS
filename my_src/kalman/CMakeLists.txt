# my_src/ncnn_model/CMakeLists.txt
cmake_minimum_required(VERSION 3.8)

# 设置项目名称
project(kalman_filter LANGUAGES C CXX)

# 设置源文件
# file(GLOB SRC_FILES "./*.cpp")
set(SRC_FILES
    ./KalmanFilter.cpp
)

# set 所有include文件夹
set(INCLUDE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}"
	"${CMAKE_SOURCE_DIR}/include/eigen-3.4.0"
)

# 生成动态库 & 生成静态库
set(SHARED_LIB_NAME ${PROJECT_NAME}_shared)
set(STATIC_LIB_NAME ${PROJECT_NAME}_static)

add_library(${SHARED_LIB_NAME} SHARED ${SRC_FILES}) # 生成动态库
target_include_directories(${SHARED_LIB_NAME} PUBLIC ${INCLUDE_DIR})
add_library(${STATIC_LIB_NAME} STATIC ${SRC_FILES}) # 生成静态库
target_include_directories(${STATIC_LIB_NAME} PUBLIC ${INCLUDE_DIR})
