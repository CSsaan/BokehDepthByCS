# my_src/ncnn_model/CMakeLists.txt
cmake_minimum_required(VERSION 3.8)

# 设置项目名称
project(ncnn_model LANGUAGES C CXX)


# #####################################################################
# 自定义的子模块统一在根目录下的CMakeLists.txt来编译生成库，其余子模块无需重复编译
# add_subdirectory(${CMAKE_SOURCE_DIR}/my_src/kalman  ${CMAKE_BINARY_DIR}/kalman_build EXCLUDE_FROM_ALL)
# #####################################################################

# 设置源文件
# file(GLOB SRC_FILES "./*.cpp")
set(SRC_FILES
    ./DptV2.cpp
    ./Yolov8.cpp
    ./FaceDetect.cpp
    ./BlazeFace.cpp
    ./DptV2MNN.cpp
)

# set 所有include文件夹
set(INCLUDE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}"
	"${OPENCV_PATH}/include"
	"${NCNN_PATH}/include"
	"${NCNN_PATH}/include/ncnn"
    "${CMAKE_SOURCE_DIR}/include/eigen-3.4.0"
    "${CMAKE_SOURCE_DIR}/my_src/kalman"
    "${MNN_PATH}/include"
	"${MNN_PATH}/include/cv"
)

# 生成动态库 & 生成静态库
set(SHARED_LIB_NAME ${PROJECT_NAME}_shared)
set(STATIC_LIB_NAME ${PROJECT_NAME}_static)

add_library(${SHARED_LIB_NAME} SHARED ${SRC_FILES}) # 生成动态库
target_include_directories(${SHARED_LIB_NAME} PUBLIC ${INCLUDE_DIR})
add_library(${STATIC_LIB_NAME} STATIC ${SRC_FILES}) # 生成静态库
target_include_directories(${STATIC_LIB_NAME} PUBLIC ${INCLUDE_DIR})

# 链接第三方库
target_link_libraries(${SHARED_LIB_NAME}
    PRIVATE kalman_filter_shared
)
target_link_libraries(${STATIC_LIB_NAME}
    PRIVATE kalman_filter_static
)

# 宏定义
target_compile_definitions(${SHARED_LIB_NAME} PRIVATE 
    ${MY_DEFINITIONS}
)
target_compile_definitions(${STATIC_LIB_NAME} PRIVATE 
    ${MY_DEFINITIONS}
)