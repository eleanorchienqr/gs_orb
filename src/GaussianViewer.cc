/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/


#include "GaussianViewer.h"
#include "LoopClosing.h"
#include "ORBmatcher.h"
#include "Optimizer.h"
#include "Converter.h"
#include "GeometricTools.h"

#include "Rasterizer.h"

#include<mutex>
#include<chrono>

#include "Renderer.h"
#include <GL/gl.h>

namespace ORB_SLAM3
{

GaussianViewer::GaussianViewer(System* pSys, Atlas *pAtlas, const float bMonocular, bool bInertial, const string &_strSeqName):
    mpSystem(pSys), mbMonocular(bMonocular), mbInertial(bInertial)
{
    
}

void GaussianViewer::SetLoopCloser(LoopClosing* pLoopCloser)
{
    mpLoopCloser = pLoopCloser;
}

void GaussianViewer::SetLocalMapper(LocalMapping* pLocalMapper)
{
    mpLocalMapper = pLocalMapper;
}

void GaussianViewer::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

void GaussianViewer::Run()
{
    while(1)
    {
        std::cout << ">>>>>>>>Start Gaussian Rendering " << std::endl;
        if (!glfwInit()) {
            throw std::runtime_error{"GLFW could not be initialized."};
        }  

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);

        /* Create a windowed mode window and its OpenGL context */
        mGLFWindow = glfwCreateWindow(mWindowSizeWidth, mWindowSizeHeight, "Gaussian Render View", NULL, NULL);
        if (mGLFWindow == NULL) {
            throw std::runtime_error{"GLFW window could not be created."};
        }
        /* Make the window's context current */
        glfwMakeContextCurrent(mGLFWindow);

        glewExperimental = 1;
        if (glewInit()) {
            throw std::runtime_error{"GLEW could not be initialized."};
        }
        
        glfwSwapInterval(0); // Disable vsync

        // glfwSetWindowUserPointer(mGLFWindow, this);

        float xscale, yscale;
        glfwGetWindowContentScale(mGLFWindow, &xscale, &yscale);

        // IMGUI init
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO(); (void)io;
        //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
        io.ConfigInputTrickleEventQueue = false; // new ImGui event handling seems to make camera controls laggy if this is true.
        ImGui::StyleColorsDark();
        ImGui_ImplGlfw_InitForOpenGL(mGLFWindow, true);
        ImGui_ImplOpenGL3_Init("#version 460 core");

        ImGui::GetStyle().ScaleAllSizes(xscale);
        ImFontConfig font_cfg;
        font_cfg.SizePixels = 13.0f * xscale;
        io.Fonts->AddFontDefault(&font_cfg);

        mRenderWindow = true;

        while (!glfwWindowShouldClose(mGLFWindow)) {
            glClear(GL_COLOR_BUFFER_BIT);
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            // ImGui::ShowDemoWindow();
            ImGUIWindowTest();

            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

            glfwSwapBuffers(mGLFWindow);
            glfwPollEvents();
        }
    }
}

void GaussianViewer::ImGUIWindowTest()
{
    bool my_tool_active = true;
    ImGui::Begin("My First Tool", &my_tool_active, ImGuiWindowFlags_MenuBar);
    if (ImGui::BeginMenuBar())
    {
        if (ImGui::BeginMenu("File"))
        {
            if (ImGui::MenuItem("Open..", "Ctrl+O")) { /* Do stuff */ }
            if (ImGui::MenuItem("Save", "Ctrl+S"))   { /* Do stuff */ }
            if (ImGui::MenuItem("Close", "Ctrl+W"))  { my_tool_active = false; }
            ImGui::EndMenu();
        }
        ImGui::EndMenuBar();
    }

    // Edit a color (stored as ~4 floats)
    float my_color[] = { 0.2f, 0.1f, 1.0f, 0.3f};
    ImGui::ColorEdit4("Color", my_color);

    // Plot some values
    const float my_values[] = { 0.2f, 0.1f, 1.0f, 0.5f, 0.9f, 2.2f };
    ImGui::PlotLines("Frame Times", my_values, IM_ARRAYSIZE(my_values));

    // Display contents in a scrolling region
    ImGui::TextColored(ImVec4(1,1,0,1), "Important Stuff");
    ImGui::BeginChild("Scrolling");
    for (int n = 0; n < 50; n++)
        ImGui::Text("%04d: Some text", n);
    ImGui::EndChild();
    ImGui::End();
}

} //namespace ORB_SLAM
