#ifndef RENDERGUI_H
#define RENDERGUI_H

#include <iostream>


#include <Thirdparty/imgui/imgui.h>
#include <Thirdparty/imgui/backends/imgui_impl_glfw.h>
#include <Thirdparty/imgui/backends/imgui_impl_opengl3.h>
// #include <Thirdparty/imguizmo/ImGuizmo.h>
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

public:
    RenderGUI();
    ~RenderGUI();

    bool InitializeWidget();

};

} //namespace ORB_SLAM3

#endif // RENDERGUI_H