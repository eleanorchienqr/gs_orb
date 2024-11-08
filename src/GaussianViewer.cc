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
#include "Renderer.h"
#include "Texture.h"
#include "VertexBuffer.h"
#include "IndexBuffer.h"
#include "VertexArray.h"
#include "VertexBufferLayout.h"
#include "Shader.h"


#include<mutex>
#include<chrono>

typedef void (*ImGuiMarkerCallback)(const char* file, int line, const char* section, void* user_data);
extern ImGuiMarkerCallback      GImGuiMarkerCallback;
extern void*                        GImGuiMarkerCallbackUserData;
ImGuiMarkerCallback             GImGuiMarkerCallback = NULL;
void*                               GImGuiMarkerCallbackUserData = NULL;
#define IMGUI_MARKER(section)  do { if (GImGuiMarkerCallback != NULL) GImGuiMarkerCallback(__FILE__, __LINE__, section, GImGuiMarkerCallbackUserData); } while (0)

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
    // std::cout << ">>>>>>>>Start Gaussian Rendering " << std::endl;
    InitializeGLFW();
    InitializeImGUI();

    // float positions[] = {
    //     -0.5f, -0.5f, 0.0f, 1.0f,
    //      0.5f, -0.5f, 1.0f, 0.0f, 
    //      0.5f,  0.5f, 1.0f, 1.0f,
    //     -0.5f,  0.5f, 0.0f, 0.0f
    // };

    float positions[] = {
        -0.5f, -0.5f, 
         0.5f, -0.5f, 
         0.5f,  0.5f, 
        -0.5f,  0.5f
    };

    unsigned int indices[] = {
        0, 1, 2, 
        2, 3, 0,
    };

    VertexArray va;
    VertexBuffer vb(positions, 4 * 2 * sizeof(float));

    VertexBufferLayout layout;
    // layout.Push<float>(2);  
    layout.Push<float>(2);  
    va.AddBuffer(vb, layout);

    IndexBuffer ib(indices, 6);

    // Texture texture("/home/ray/Desktop/Dataset/TUM/rgbd_dataset_freiburg3_long_office_household/rgb/1341847980.722988.png");
    // texture.Bind();

    Shader shader("/home/ray/Desktop/ORB_SLAM3/src/Renderer/assets");
    shader.Bind();
    // shader.SetUnifrom1i("u_Texture", 0);
    shader.SetUnifrom4f("u_Color", 0.8f, 0.3f, 0.8f, 1.0f);

    va.Unbind();
    vb.Unbind();
    ib.Unbind();
    shader.Unbind();
    Renderer renderer;

    while (!glfwWindowShouldClose(mGLFWindow)) 
    {
        // glClear(GL_COLOR_BUFFER_BIT);
        renderer.Clear();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // ShowMenuBar();
        // ShowWidgets();

        renderer.Draw(va, ib, shader);

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(mGLFWindow);
        glfwPollEvents();
    }

    // while(1)
    // {

        
    // }

    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();
}

void GaussianViewer::InitializeGLFW()
{
    if (!glfwInit()) {
        throw std::runtime_error{"GLFW could not be initialized."};
    }  

    // glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    // glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    // glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    // glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);

    mGLFWindow = glfwCreateWindow(mWindowSizeWidth, mWindowSizeHeight, "Gaussian Render View", NULL, NULL);
    if (mGLFWindow == NULL) {
        throw std::runtime_error{"GLFW window could not be created."};
    }
    glfwMakeContextCurrent(mGLFWindow);

    glewExperimental = 1;
    if (glewInit()) {
        throw std::runtime_error{"GLEW could not be initialized."};
    }
    
    glfwSwapInterval(1); 

    GLCall(glEnable(GL_BLEND));
    GLCall(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
}

    
void GaussianViewer::InitializeImGUI()
{
    float xscale, yscale;
    glfwGetWindowContentScale(mGLFWindow, &xscale, &yscale);

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
}

void GaussianViewer::DrawFrameTest()
{
    cv::Mat TestImg = cv::imread("/home/ray/Desktop/Dataset/TUM/rgbd_dataset_freiburg3_long_office_household/rgb/1341847980.722988.png");

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);     // Operate on model-view matrix

    glEnable(GL_TEXTURE_2D);
    GLuint image_tex = matToTexture(TestImg, GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR, GL_CLAMP);

    /* Draw a quad */
    glBegin(GL_QUADS);
    glTexCoord2i(0, 0); glVertex2i(0,   0);
    glTexCoord2i(0, 1); glVertex2i(0,   mWindowSizeHeight);
    glTexCoord2i(1, 1); glVertex2i(mWindowSizeWidth, mWindowSizeHeight);
    glTexCoord2i(1, 0); glVertex2i(mWindowSizeWidth, 0);
    glEnd();

    glDeleteTextures(1, &image_tex);
    glDisable(GL_TEXTURE_2D);
}

GLuint GaussianViewer::matToTexture(const cv::Mat &mat, GLenum minFilter, GLenum magFilter, GLenum wrapFilter)
{
    // Generate a number for our textureID's unique handle
    GLuint textureID;
    glGenTextures(1, &textureID);

    // Bind to our texture handle
    glBindTexture(GL_TEXTURE_2D, textureID);

    // // Catch silly-mistake texture interpolation method for magnification
    // if (magFilter == GL_LINEAR_MIPMAP_LINEAR  ||
    //         magFilter == GL_LINEAR_MIPMAP_NEAREST ||
    //         magFilter == GL_NEAREST_MIPMAP_LINEAR ||
    //         magFilter == GL_NEAREST_MIPMAP_NEAREST)
    // {
    //     cout << "You can't use MIPMAPs for magnification - setting filter to GL_LINEAR" << endl;
    //     magFilter = GL_LINEAR;
    // }

    // // Set texture interpolation methods for minification and magnification
    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minFilter);
    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, magFilter);

    // Set texture clamping method
    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapFilter);
    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrapFilter);

    // use fast 4-byte alignment (default anyway) if possible
    // glPixelStorei(GL_UNPACK_ALIGNMENT, (mat.step & 3) ? 1 : 4);
    // glPixelStorei(GL_UNPACK_ROW_LENGTH, mat.step/mat.elemSize());

    cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);
    cv::flip(mat, mat, -1);

    // Create the texture
    glTexImage2D(GL_TEXTURE_2D,     // Type of texture
                 0,                 // Pyramid level (for mip-mapping) - 0 is the top level
                 GL_RGB,            // Internal colour format to convert to
                 mat.cols,          // Image width  i.e. 640 for Kinect in standard mode
                 mat.rows,          // Image height i.e. 480 for Kinect in standard mode
                 0,                 // Border width in pixels (can either be 1 or 0)
                 GL_RGB,            // Input image format (i.e. GL_RGB, GL_RGBA, GL_BGR etc.)
                 GL_UNSIGNED_BYTE,  // Image data type
                 mat.ptr());        // The actual image data itself

    // If we're using mipmaps then generate them. Note: This requires OpenGL 3.0 or higher
    // if (minFilter == GL_LINEAR_MIPMAP_LINEAR  ||
    //         minFilter == GL_LINEAR_MIPMAP_NEAREST ||
    //         minFilter == GL_NEAREST_MIPMAP_LINEAR ||
    //         minFilter == GL_NEAREST_MIPMAP_NEAREST)
    // {
    //     glGenerateMipmap(GL_TEXTURE_2D);
    // }

    return textureID;
}

void GaussianViewer::ShowMenuBar()
{
    if (ImGui::BeginMainMenuBar())
    {
        if (ImGui::BeginMenu("Edit"))
        {
            if (ImGui::MenuItem("Undo", "CTRL+Z")) {}
            if (ImGui::MenuItem("Redo", "CTRL+Y", false, false)) {} 
            ImGui::Separator();
            if (ImGui::MenuItem("Cut", "CTRL+X")) {}
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }

}

void GaussianViewer::ShowWidgets()
{
    ImGui::Begin("Options and Infos", &mToolActive, ImGuiWindowFlags_MenuBar);

    static int clicked = 0;
    if (ImGui::Button("Pause"))
        clicked++;
    if (clicked & 1)
    {
        ImGui::SameLine();
        ImGui::Text("Pause");
    }

    IMGUI_MARKER("ViewPoint Options");
    if (ImGui::CollapsingHeader("ViewPoint Options"))
    {

        ImGui::SeparatorText("Camera Follow Option");
        if (ImGui::BeginTable("split", 3))
        {
            ImGui::TableNextColumn(); ImGui::Checkbox("Follow Camera", &mFollowCamera);
            ImGui::TableNextColumn(); ImGui::Checkbox("From Behind", &mFromBehind);
            ImGui::EndTable();
        }


        ImGui::SeparatorText("ViewPoint List");

        const char* items[] = { "AAAA", "BBBB", "CCCC", "DDDD", "EEEE", "FFFF"};
        static int item_selected_idx = 0; // Here we store our selected data as an index.
        static bool item_highlight = false;
        int item_highlighted_idx = -1; // Here we store our highlighted data as an index.

        if (ImGui::BeginListBox(""))
        {
            for (int n = 0; n < IM_ARRAYSIZE(items); n++)
            {
                const bool is_selected = (item_selected_idx == n);
                if (ImGui::Selectable(items[n], is_selected))
                    item_selected_idx = n;

                if (item_highlight && ImGui::IsItemHovered())
                    item_highlighted_idx = n;

                // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                if (is_selected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndListBox();
        }
    }

    IMGUI_MARKER("3D Objects");
    if (ImGui::CollapsingHeader("3D Objects"))
    {
        if (ImGui::BeginTable("split", 3))
        {
            ImGui::TableNextColumn(); ImGui::Checkbox("Cameras", &mShowCameraObjects);
            ImGui::TableNextColumn(); ImGui::Checkbox("Active Window", &mShowActiveWindow);
            ImGui::TableNextColumn(); ImGui::Checkbox("Axis", &mShowAxis);
            ImGui::EndTable();
        }
    }

    IMGUI_MARKER("Render Options");
    if (ImGui::CollapsingHeader("Render Options"))
    {
        if (ImGui::BeginTable("split", 3))
        {
            ImGui::TableNextColumn(); ImGui::Checkbox("Depth", &mRenderDepth);
            ImGui::TableNextColumn(); ImGui::Checkbox("Opacity", &mRenderOpacity);
            ImGui::TableNextColumn(); ImGui::Checkbox("Time Shader", &mRenderTimeShader);
            ImGui::TableNextColumn(); ImGui::Checkbox("Elipsoid Shader", &mRenderElpsoidShader);
            ImGui::EndTable();
        }

        ImGui::SeparatorText("Gaussian Scale (0-1)");

        {
            static float f1 = 0.123f, f2 = 0.0f;
            ImGui::SliderFloat("Scale", &f1, 0.0f, 1.0f, "ratio = %.3f");
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
