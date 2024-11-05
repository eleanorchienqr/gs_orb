#include "Renderer.h"

#include <iostream>

void GLCLearError()
{
    while (glGetError() != GL_NO_ERROR);
}

bool GLLogCall(const char* function, const char* file, int line)
{
    while (GLenum error = glGetError())
    {
        std::cout << "[OpenGL Error] (" << error << "): " << function << " " << file << ":" << line << std::endl;
        return false; 
    }
    return true;
}

void Renderer::Clear() const
{
    GLCall(glClear(GL_COLOR_BUFFER_BIT));
}

void Renderer::Draw(const VertexArray& va, const IndexBuffer& ib, const Shader& shader)
{
    shader.Bind();
    va.Bind();
    ib.Bind();

    GLCall(glDrawElements(GL_TRIANGLES, ib.GetCount(), GL_UNSIGNED_INT, nullptr));
}


TestClearColor::TestClearColor()
	:mClearColor { 0.2f, 0.3f, 0.8f, 1.0f }
{

}

TestClearColor::~TestClearColor()
{

}

void TestClearColor::OnUpdate(float deltaTime) 
{

}

void TestClearColor::OnRender() 
{
	GLCall(glClearColor(mClearColor[0], mClearColor[1], mClearColor[2], mClearColor[3]));
	GLCall(glClear(GL_COLOR_BUFFER_BIT));
}

void TestClearColor::OnImGuiRender() 
{
	ImGui::ColorEdit4("Clear Color", mClearColor);
}

TestMenu::TestMenu(Test*& CurrentTestPointer)
	:mCurrentTest (CurrentTestPointer)
{

}

void TestMenu::OnImGuiRender() 
{
	for (auto& test : mTests)
	{
		if (ImGui::Button(test.first.c_str()))
		{
			mCurrentTest = test.second();
		}
	}
}

