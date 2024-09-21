#ifndef RENDERGUI_H
#define RENDERGUI_H

#include <iostream>


#include <Thirdparty/imgui/imgui.h>
#include <Thirdparty/imgui/backends/imgui_impl_glfw.h>
#include <Thirdparty/imgui/backends/imgui_impl_opengl3.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "GaussianRenderer.h"


namespace ORB_SLAM3
{
class GaussianRenderer;

class RenderGUI
{
private:
    // bool mGUInitialization; 
    // GaussianRenderer mGaussianRenderer;
    // self.g_camera
    const int mWindowSizeWidth = 1200;
    const int mWindowSizeHeight = 800;

    GLFWwindow* mGLFWindow = nullptr;
    bool mGUIRedraw = true;
    bool mRenderWindow = false; // open in InitializeWindow; control render or not

public:
    RenderGUI();
    ~RenderGUI();

    void redraw_gui_next_frame() {
		mGUIRedraw = true;
	}

    void InitializeWindow();
    void DestroyWindow();
    void ImGUIWindow();
    bool BeginFrameAndHandleUserInput();
    bool Frame();
};

} //namespace ORB_SLAM3

#endif // RENDERGUI_H