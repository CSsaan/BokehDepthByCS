cmake_minimum_required(VERSION 3.8)

set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "build type") # Debug  Release  RelWithDebInfo  MinSizeRel
set(CMAKE_SKIP_RPATH TRUE)

project(
	bokeh-depthv2-ncnn-mnn-opengl-opencv-ByCS # PROJECT_NAME
	VERSION 1.0.0.0
	LANGUAGES C CXX
)

if (CYGWIN)
	message(FATAL_ERROR "Sorry, DO NOT support Cygwin. Please use Visual Studio Cmake.")
endif ()

if (MINGW)
	message(FATAL_ERROR "Sorry, DO NOT support MinGW. Please use Visual Studio Cmake.")
endif ()

add_custom_target(
	LINK_HEADERS ALL
	COMMENT "link headers..."
)

# 设置模型文件的路径
set(MODEL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/model")
message(STATUS "Model path: ${MODEL_PATH}")

# C++ 标准设置
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (WIN32)
	# set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   /MP /wd4200")
	# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /wd4200")
	set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   /MT /W3")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MT /W3")
else ()
	set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -Wall -fPIC -pipe")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fPIC -pipe -fno-exceptions -Wno-invalid-offsetof")
	if (APPLE)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
	endif()
endif ()

# #####################################################################
# OpenCV 配置
SET(OPENCV_PATH  "${CMAKE_CURRENT_SOURCE_DIR}/lib/opencv")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    SET(OPENCV_LIB  # Debug版本
		"${OPENCV_PATH}/x64/vc16/lib/opencv_world4100d.lib"
	)
else()
    SET(OPENCV_LIB  # Release版本
		"${OPENCV_PATH}/x64/vc16/lib/opencv_world4100.lib"
	)
endif()

# 复制 DLL 文件到输出文件夹 ${CMAKE_CURRENT_BINARY_DIR}
file(COPY 
	${CMAKE_CURRENT_SOURCE_DIR}/config.ini
	${OPENCV_PATH}/x64/vc16/bin/opencv_videoio_ffmpeg4100_64.dll
    ${OPENCV_PATH}/x64/vc16/bin/opencv_videoio_msmf4100_64.dll
	${OPENCV_PATH}/x64/vc16/bin/opencv_videoio_msmf4100_64d.dll
	${OPENCV_PATH}/x64/vc16/bin/opencv_world4100.dll
	${OPENCV_PATH}/x64/vc16/bin/opencv_world4100d.dll
	DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
)
# VulkanSDK 配置
SET(VULKANSDK_PATH  "${CMAKE_CURRENT_SOURCE_DIR}/lib/VulkanSDK")
SET(VULKANSDK_LIB   
	${VULKANSDK_PATH}/Lib/vulkan-1.lib
)
# Protobuf
SET(PROTOBUF_PATH  ${CMAKE_CURRENT_SOURCE_DIR}/lib/protobuf)
SET(PROTOBUF_LIB    
	${PROTOBUF_PATH}/lib/libprotobuf.lib
)
# NCNN (Release版本)
SET(NCNN_PATH  ${CMAKE_CURRENT_SOURCE_DIR}/lib/ncnn)
SET(NCNN_LIB        
	${NCNN_PATH}/lib/ncnn.lib
	${NCNN_PATH}/lib/glslang.lib
	${NCNN_PATH}/lib/SPIRV.lib
	${NCNN_PATH}/lib/OGLCompiler.lib
	${NCNN_PATH}/lib/OSDependent.lib
	${NCNN_PATH}/lib/MachineIndependent.lib
	${NCNN_PATH}/lib/GenericCodeGen.lib
)
# MNN 配置
SET(MNN_PATH  ${CMAKE_CURRENT_SOURCE_DIR}/lib/mnn)
SET(MNN_LIB        
	${MNN_PATH}/lib/x64/Release/Dynamic/MT/MNN.lib
)
# 复制 DLL 文件到输出文件夹 ${CMAKE_CURRENT_BINARY_DIR}
file(COPY 
	${MNN_PATH}/lib/x64/Release/Dynamic/MT/MNN.dll
	DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
)

# 预处理宏定义
set(MY_DEFINITIONS 
	MODEL_PATH="${MODEL_PATH}"
	ASSERT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/assets"
	WINDOW_WIDTH=1080
	WINDOW_HEIGHT=1920
)
# #####################################################################
# 在子模块中cmake中配置 Include 路径，无需在根目录cmake中重复配置
add_subdirectory(my_src/ncnn_model)
add_subdirectory(my_src/render)
add_subdirectory(${CMAKE_SOURCE_DIR}/my_src/kalman ${CMAKE_BINARY_DIR}/kalman_build EXCLUDE_FROM_ALL)
# #####################################################################
# 源代码添加到可执行文件
# file(GLOB SRC_FILES "./*.cpp"  "./src/*.cpp")
set(SRC_FILES
    ./src/main.cpp
)
add_executable(${PROJECT_NAME} ${SRC_FILES} version.rc) # WIN32 使得windows平台不显示命令窗口
target_compile_options(${PROJECT_NAME} PRIVATE -Wall)

# Include 路径
include_directories(
	"${CMAKE_CURRENT_SOURCE_DIR}/include" 
	"${CMAKE_CURRENT_SOURCE_DIR}/src"
	"${OPENCV_PATH}/include"
	"${VULKANSDK_PATH}/Include"
	"${PROTOBUF_PATH}/include"
	"${PROTOBUF_PATH}/include/google"
	"${NCNN_PATH}/include"
	"${NCNN_PATH}/include/ncnn"
	"${MNN_PATH}/include"
	"${MNN_PATH}/include/cv"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/imgui"
)

# 库路径
SET(ALL_LIB_PATH 
	${OPENCV_LIB}
	${VULKANSDK_LIB}
	${PROTOBUF_LIB}
	${NCNN_LIB}
	${MNN_LIB}
)
target_link_libraries(${PROJECT_NAME} PRIVATE ${ALL_LIB_PATH}  ncnn_model_static  render_static  kalman_filter_static) # openCV采用的为静态版本

# 预处理宏定义: 将模型路径(MODEL_PATH)\资源文件路径(ASSERT_DIR)定义为预处理宏
target_compile_definitions(${PROJECT_NAME} PRIVATE 
    ${MY_DEFINITIONS}
)

if (CMAKE_VERSION VERSION_GREATER 3.12)
	set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)
endif()
