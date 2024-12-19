# my_src/ncnn_model/CMakeLists.txt
cmake_minimum_required(VERSION 3.8)

# 设置项目名称
project(render LANGUAGES C CXX)

# 设置源文件
# file(GLOB SRC_FILES "./*.cpp"   "${CMAKE_SOURCE_DIR}/include/imgui/*.cpp")
set(SRC_FILES
    ./MyBokeh.cpp
    ${CMAKE_SOURCE_DIR}/include/imgui/imgui.cpp
    ${CMAKE_SOURCE_DIR}/include/imgui/imgui_demo.cpp
    ${CMAKE_SOURCE_DIR}/include/imgui/imgui_draw.cpp
    ${CMAKE_SOURCE_DIR}/include/imgui/imgui_impl_glfw.cpp
    ${CMAKE_SOURCE_DIR}/include/imgui/imgui_impl_opengl3.cpp
    ${CMAKE_SOURCE_DIR}/include/imgui/imgui_tables.cpp
    ${CMAKE_SOURCE_DIR}/include/imgui/imgui_widgets.cpp
)

# include 头文件
include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${OPENCV_PATH}/include"
    "${CMAKE_SOURCE_DIR}/include/imgui"
)

# OpenGL 配置
SET(OPENGL_PATH  ${CMAKE_SOURCE_DIR}/lib/opengl)
add_subdirectory(${OPENGL_PATH}/glfw ${CMAKE_BINARY_DIR}/glfw_build EXCLUDE_FROM_ALL)
add_subdirectory(${OPENGL_PATH}/glew ${CMAKE_BINARY_DIR}/glew_build EXCLUDE_FROM_ALL)
add_subdirectory(${OPENGL_PATH}/glm ${CMAKE_BINARY_DIR}/glm_build EXCLUDE_FROM_ALL)

# 生成动态库 & 生成静态库
set(SHARED_LIB_NAME ${PROJECT_NAME}_shared)
set(STATIC_LIB_NAME ${PROJECT_NAME}_static)

add_library(${SHARED_LIB_NAME} SHARED ${SRC_FILES}) # 生成动态库
target_include_directories(${SHARED_LIB_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}" "${OPENGL_PATH}/glfw/include" "${OPENGL_PATH}/glew/include" "${OPENGL_PATH}/glm")
add_library(${STATIC_LIB_NAME} STATIC ${SRC_FILES}) # 生成静态库
target_include_directories(${STATIC_LIB_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}" "${OPENGL_PATH}/glfw/include" "${OPENGL_PATH}/glew/include" "${OPENGL_PATH}/glm")

# 链接第三方库
target_link_libraries(${SHARED_LIB_NAME}
    PRIVATE glfw
    PRIVATE libglew_static
    PRIVATE glm
)
target_link_libraries(${STATIC_LIB_NAME}
    PRIVATE glfw
    PRIVATE libglew_static
    PRIVATE glm
)

# 宏定义
target_compile_definitions(${SHARED_LIB_NAME} PRIVATE 
    ${MY_DEFINITIONS}
)
target_compile_definitions(${STATIC_LIB_NAME} PRIVATE 
    ${MY_DEFINITIONS}
)