#ifndef RENDER_H
#define RENDER_H

#include <GL/glew.h>
#include <cassert>
#include <iostream>


#include <Thirdparty/imgui/imgui.h>
#include <Thirdparty/imgui/backends/imgui_impl_glfw.h>
#include <Thirdparty/imgui/backends/imgui_impl_opengl3.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "VertexArray.h"
#include "IndexBuffer.h"
#include "Shader.h"

#define GLCall(x) GLCLearError();x;assert(GLLogCall(#x, __FILE__, __LINE__))

void GLCLearError();
bool GLLogCall(const char* function, const char* file, int line);

class Renderer
{

public:
    void Clear() const;
    void Draw(const VertexArray& va, const IndexBuffer& ib, const Shader& shader);
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

#endif

