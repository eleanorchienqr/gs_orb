#include "RenderGUI.h"
#include "Renderer.h"
#include <GL/gl.h>

namespace ORB_SLAM3
{

RenderGUI::RenderGUI()
{
    

}

RenderGUI::~RenderGUI()
{
    
}

void RenderGUI::InitializeWindow()
{
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
}

void RenderGUI::DestroyWindow()
{
    if (!mRenderWindow) {
		throw std::runtime_error{"Window must be initialized to be destroyed."};
	}

    ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	glfwDestroyWindow(mGLFWindow);
	glfwTerminate();

	mGLFWindow = nullptr;
	mRenderWindow = false;
}

void RenderGUI::ImGUIWindow()
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

bool RenderGUI::BeginFrameAndHandleUserInput()
{
    if (glfwWindowShouldClose(mGLFWindow)) {
		
		DestroyWindow();
		return false;
	}

    while (!glfwWindowShouldClose(mGLFWindow)) {
		glClear(GL_COLOR_BUFFER_BIT);
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
		// ImGui::ShowDemoWindow();
        ImGUIWindow();

		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		glfwSwapBuffers(mGLFWindow);
		glfwPollEvents();
	}

    return true;
}

bool RenderGUI::Frame()
{
    if (mRenderWindow) {
		if (!BeginFrameAndHandleUserInput()) {
			return false;
		}
	}
    // if (m_render_window) {
	// 	if (m_gui_redraw) {
	// 		draw_gui();
	// 		m_gui_redraw = false;

	// 		m_last_gui_draw_time_point = std::chrono::steady_clock::now();
	// 	}

	// 	ImGui::EndFrame();
    // }
    return true;
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

}


