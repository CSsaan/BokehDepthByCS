#include "MyBokeh.hpp"
/**
 * MyBokeh.cpp
 * Contributors:
 *      * CS (author)
 * Licence:
 *      * MIT
 */
#include "MyBokeh.hpp"

#include <iostream>
#include <vector>

namespace CS
{
    MyBokeh::MyBokeh() {
        // funcIndex = CS_ZEBRA;
        // setting the window size and aspect ratio
        setWindowAspectRatio(texture_frame->getWidth(), texture_frame->getHeight(), GL_WINDOW_SHOW_DOWNSCALE_RATE);
    }

    void MyBokeh::anotherImGui() {
        // 3. Show another MyPseudocolor window.
        if (show_another_window) {
            ImGui::Begin("Hello, OpenGL Cmake CS!");  // Create a window called "Hello, world!" and append into it.
            ImGui::BulletText("This is a C++14 CMake project for OpenGL applications by CS.");
            ImGui::BulletText("It includes the following libraries: GLFW, Glew, glm, and assimp.");
            ImGui::BulletText("The project is designed to be cross-platform and can be compiled on \nLinux, Windows, and Mac.");

            // ImGui::Begin("Another Window", &show_another_window);  // Pass a pointer to our bool variable (the window will have a closing button that will clear the bool when clicked)
            // ImGui::Text(("Hello from " + title + " window!").c_str());
            ImGui::Spacing();

            //// Select Function
            //selectFunction();

            //ImGui::SeparatorText("");
            //// Select Image
            //if (ImGui::CollapsingHeader("Image")) {
            //    std::string selectedImagePath = imgPathItems[0];
            //    ImGui::SetNextItemWidth(ImGui::GetFontSize() * 10);
            //    if (ImGui::BeginCombo("select picture", selectedImagePath.c_str())) {
            //        for (const auto& imgPath : imgPathItems) {
            //            if (ImGui::Selectable(imgPath.c_str())) {
            //                selectedImagePath = imgPath;
            //                //texture->update(ASSERT_DIR + selectedImagePath);
            //                //setWindowAspectRatio(texture->getWidth(), texture->getHeight());
            //            }
            //        }
            //        ImGui::EndCombo();
            //    }
            //    ImGui::SameLine();
            //    static bool useLocalImgPath = false;
            //    ImGui::Checkbox("Use Local Image", &useLocalImgPath);
            //    if (useLocalImgPath) {
            //        static char inputBuffer[1024] = "";
            //        ImGui::InputTextWithHint(" ", "Image path", inputBuffer, IM_ARRAYSIZE(inputBuffer), ImGuiInputTextFlags_EnterReturnsTrue);
            //        if (ImGui::IsItemDeactivatedAfterEdit()) {
            //            //texture->update(inputBuffer);
            //            //setWindowAspectRatio(texture->getWidth(), texture->getHeight());
            //        }
            //    }
            //}

            // Options
            ImGui::SeparatorText("Focus Dis & Scale");
            ImGui::SetNextItemWidth(ImGui::GetFontSize() * 8);
            ImGui::SliderFloat("Focus Dis", &focus_disScale[0], 0.0f, 1.0f);
            ImGui::SameLine();
            HelpMarker("dis == 0.0 ? face_auto_focus_dis : Dis value.");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(ImGui::GetFontSize() * 8);
            ImGui::SliderFloat("Focus Scale", &focus_disScale[1], 0.0f, 1.0f);

            ImGui::SeparatorText("Focus Gamma & Hardness & CoC");
            ImGui::SetNextItemWidth(ImGui::GetFontSize() * 8);
            ImGui::SliderFloat("Gamma", &gamma_hardness_CoC[0], 1.0f, 10.0f);
            ImGui::SameLine();
            ImGui::SetNextItemWidth(ImGui::GetFontSize() * 8);
            ImGui::SliderFloat("Hardness", &gamma_hardness_CoC[1], 0.0f, 1.0f);
            ImGui::SameLine();
            ImGui::SetNextItemWidth(ImGui::GetFontSize() * 8);
            ImGui::SliderFloat("CoC", &gamma_hardness_CoC[2], 0.0f, 0.025f);

            //// Close
            //ImGui::Spacing();
            //if (ImGui::Button("Close window"))
            //    show_another_window = false;

            ImGui::End();
        }
    }

    void MyBokeh::loop() {
        processInput(getWindow());
        if (glfwWindowShouldClose(getWindow())) {
            exit();
            return;
        }
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(0.6784f, 0.8f, 1.0f, 1.0f);

        render();
    }

    void MyBokeh::render() {
        // Box Blur
        glViewport(0, 0, FBO_TEX_WIDTH, FBO_TEX_HEIGHT); // set Viewport size to FBO_result_texture size
        // 启用深度测试
        glEnable(GL_DEPTH_TEST);
        m_fbo->bind();
        for (int i = 0; i < 2 * BLUR_TIMES; ++i) // 循环次数需要为偶数
        {
            shaderProgram_boxBlur->use();
            glBindVertexArray(VAO);
            shaderProgram_boxBlur->setUniformMat4("model", glm::mat4(1.0f));
            shaderProgram_boxBlur->setUniform2f("iResolution", glm::vec2(1.0f * frame.cols, 1.0f * frame.rows));
            shaderProgram_boxBlur->setUniform1i("even_odd", i % 2);
            glCheckError(__FILE__, __LINE__);
            if (i % 2 == 0) // 横向 blur
            {
                if (i == 0) // 首次需要将原图传入
                {
                    if (depth.cols != texture_depth->getWidth() || depth.rows != texture_depth->getHeight())
                    {
                        texture_depth->update(depth.cols, depth.rows, GL_BGR);
                    }
                    texture_depth->bind(0);
                    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, depth.cols, depth.rows, GL_RED, GL_UNSIGNED_BYTE, depth.data);
                }
                else
                {
                    texture_boxBlur2_result->bind(0);
                }
                m_fbo->bindResultTexture(GL_TEXTURE_2D, texture_boxBlur1_result->getTexId(), 1); // result texture id is 1
            }
            else // 纵向 blur
            {
                texture_boxBlur1_result->bind(0);
                m_fbo->bindResultTexture(GL_TEXTURE_2D, texture_boxBlur2_result->getTexId(), 1); // result texture id is 1
            }
            glDrawArrays(GL_TRIANGLES, 0, 6);
            glCheckError(__FILE__, __LINE__);
            glBindTexture(GL_TEXTURE_2D, GL_NONE);
            glBindVertexArray(GL_NONE);
            shaderProgram_boxBlur->unuse();
        }
        m_fbo->unbind();
        // -------------------------------------------------------------------------
        setWindowViewport(); // set Viewport size to windows size
        shaderProgram->use();
        glBindVertexArray(VAO);
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, glm::vec3(0.0f, 0.0f, 0.0f));
        model = glm::rotate(model, glm::radians(0.0f), glm::vec3(0.0, 1.0, 0.0));
        model = glm::scale(model, glm::vec3(1.0f));
        shaderProgram->setUniformMat4("model", model);
        // shaderProgram->setUniform1f("IGlobalTime", static_cast<float>(glfwGetTime()));
        shaderProgram->setUniform2f("iResolution", glm::vec2(1.0f * frame.cols, 1.0f * frame.rows));
        shaderProgram->setUniform1f("focus_dis", focus_dis);
        shaderProgram->setUniform2f("focus_disScale", focus_disScale);
        shaderProgram->setUniform3f("gamma_hardness_CoC", gamma_hardness_CoC);
        glCheckError(__FILE__, __LINE__);

        if (frame.cols != texture_frame->getWidth() || frame.rows != texture_frame->getHeight())
        {
            texture_frame->update(frame.cols, frame.rows, GL_BGR);
            setWindowAspectRatio(frame.cols, frame.rows, GL_WINDOW_SHOW_DOWNSCALE_RATE);
        }
        texture_frame->bind(0);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, frame.cols, frame.rows, GL_BGR, GL_UNSIGNED_BYTE, frame.data);
        texture_boxBlur2_result->bind(1); // FBO blur 结果
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glCheckError(__FILE__, __LINE__);
        glBindTexture(GL_TEXTURE_2D, GL_NONE);
        glBindVertexArray(GL_NONE);
        shaderProgram->unuse();
    }

    void MyBokeh::setFocusDis(float dis) {
        std::cout << "setFocusDis: " << dis << std::endl;
        focus_dis = dis;
    }

    void MyBokeh::readRenderResult(cv::Mat& frame)
    {
        int w = getWidth();
        int h = getHeight();
        frame.create(h, w, CV_8UC3); // 假设使用 RGB 格式
        glReadPixels(0, 0, w, h, GL_BGR, GL_UNSIGNED_BYTE, frame.data);
        cv::flip(frame, frame, 0); // 垂直翻转以匹配 OpenCV 的坐标系 // OpenCV Mat 的数据是从左下角开始的，而 OpenGL 的坐标系是从左上角开始的
    }

    void MyBokeh::processInput(GLFWwindow* window) {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);

        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            std::cout << " GLFW_KEY_W " << std::endl;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            std::cout << " GLFW_KEY_S " << std::endl;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            std::cout << " GLFW_KEY_A " << std::endl;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            std::cout << " GLFW_KEY_D " << std::endl;
    }
}