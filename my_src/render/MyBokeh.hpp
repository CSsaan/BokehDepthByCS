/**
 * MyBokeh.hpp skeleton
 * Contributors:
 *      * CS
 * Licence:
 *      * MIT
 */

#pragma once

#include "Application.hpp"
#include "Shader.hpp"
#include "Texture.hpp"

#define GLM_ENABLE_EXPERIMENTAL
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/matrix_operation.hpp>

#include "glError.hpp"
#include "Framebuffer.hpp"

constexpr auto GL_WINDOW_SHOW_DOWNSCALE_RATE = 1;
constexpr auto BLUR_TIMES = 1;
constexpr auto FBO_TEX_WIDTH = 1920;
constexpr auto FBO_TEX_HEIGHT = 1080;

namespace CS
{
    class MyBokeh : public Application {
    public:
        MyBokeh();
        ~MyBokeh() = default;
        void setFocusDis(float dis);
        void readRenderResult(cv::Mat& frame);

    protected:
        std::string title = "Bokeh";
        void loop() override;
        void anotherImGui() override;
        void processInput(GLFWwindow* window);

    private:
        std::unique_ptr<Fbo> m_fbo = std::make_unique<Fbo>();
        std::unique_ptr<Texture> texture_boxBlur1_result = std::make_unique<Texture>(FBO_TEX_WIDTH, FBO_TEX_HEIGHT, GL_RGBA);
        std::unique_ptr<Texture> texture_boxBlur2_result = std::make_unique<Texture>(FBO_TEX_WIDTH, FBO_TEX_HEIGHT, GL_RGBA);
        std::unique_ptr<Shader> shaderProgram_boxBlur = std::make_unique<Shader>(ASSERT_DIR "/shader/Bokeh/BoxBlur.vert", ASSERT_DIR "/shader/Bokeh/BoxBlur.frag");

        float focus_dis{ 0.5f };
        const std::vector<std::string> imgPathItems = { "/picture/Lakewater_trees.jpg", "/picture/Sunset.jpg", "/picture/view2.jpg" };
        std::unique_ptr<Shader> shaderProgram = std::make_unique<Shader>(ASSERT_DIR "/shader/Bokeh/Bokeh.vert", ASSERT_DIR "/shader/Bokeh/Bokeh.frag");
        //std::unique_ptr<Texture> texture = std::make_unique<Texture>(ASSERT_DIR + imgPathItems[0]);
        std::unique_ptr<Texture> texture_frame = std::make_unique<Texture>(1920, 1080, GL_BGR);
        std::unique_ptr<Texture> texture_depth = std::make_unique<Texture>(1920, 1080, GL_BLUE);
        glm::vec2 focus_disScale{ 0.0f, 0.257f }; // dis < 0.01 ? focus_dis : dis
        glm::vec3 gamma_hardness_CoC{ 4.2f, 0.9f, 0.002 };
        void render();
    };
}