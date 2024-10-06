#ifndef RENDERGUI_H
#define RENDERGUI_H

#include <functional>
#include <iostream>


#include <Thirdparty/imgui/imgui.h>
#include <Thirdparty/imgui/backends/imgui_impl_glfw.h>
#include <Thirdparty/imgui/backends/imgui_impl_opengl3.h>
// #include <Thirdparty/imguizmo/ImGuizmo.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <string>
#include <utility>
#include <vector>

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

class Test{

public:
    Test(){}
    virtual ~Test() {}

    virtual void OnUpdate(float deltaTime) {}
    virtual void OnRender() {}
    virtual void OnImGuiRender() {}


};

class TestClearColor : public Test
{
public:
    TestClearColor();
    ~TestClearColor();

    void OnUpdate(float deltaTime) override;
    void OnRender() override;
    void OnImGuiRender() override;

private:
    float mClearColor[4];

};

class TestMenu : public Test
{
public:
    TestMenu(Test*& CurrentTestPointer);

    void OnImGuiRender() override;

    template<typename T>
    void RegisterTest(const std::string& name)
    {
        std::cout << "Registering test" << name << std::endl;
        mTests.push_back(std::make_pair(name, []() { return new T(); }));
    }

private:
    Test*& mCurrentTest;
    std::vector<std::pair<std::string, std::function<Test*()>>> mTests;

};

} //namespace ORB_SLAM3

#endif // RENDERGUI_H