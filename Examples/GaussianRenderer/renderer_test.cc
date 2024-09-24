#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>

#include "Renderer.h"
#include "VertexBuffer.h"
#include "IndexBuffer.h"
#include "VertexArray.h"
#include "VertexBufferLayout.h"
#include "Shader.h"
#include "RenderGUI.h"

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

#include <Thirdparty/imgui/imgui.h>
#include <Thirdparty/imgui/backends/imgui_impl_glfw.h>
#include <Thirdparty/imgui/backends/imgui_impl_opengl3.h>

int main(void)
{
    GLFWwindow* window;

    /* Initialize the library */
    if (!glfwInit()) return -1;

    // glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    // glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    // glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(1600, 900, "Hello World", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }

    /* Make the window's context current */
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    GLenum err = glewInit();
    if (GLEW_OK != err) {
        std::cout << "Error: " << glewGetErrorString(err) << std::endl;
    }
    std::cout << "Status: Using GLEW " << glewGetString(GLEW_VERSION) << std::endl;
    std::cout << "Status: Using GLFW " << glfwGetVersionString() << std::endl;
    
    unsigned char* glVersion;
    GLCall(glVersion = (unsigned char*)glGetString(GL_VERSION));
    std::cout << "Status: Using GL " << glVersion << std::endl;
    
    GLCall(glEnable(GL_BLEND));
    GLCall(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));

    float positions[] = {
        -1.5f, -0.5f, 0.0f, 0.18f, 0.6f, 0.96f, 1.0f,//0
        -0.5f, -0.5f, 0.0f,0.18f, 0.6f, 0.96f, 1.0f,//1
        -0.5f, 0.5f,0.0f,0.18f, 0.6f, 0.96f, 1.0f,//2
        -1.5f, 0.5f, 0.0f,0.18f, 0.6f, 0.96f, 1.0f, //3

        0.5f, -0.5f, 0.0f,1.0f, 0.93f, 0.24f, 1.0f,//0
        1.5f, -0.5f, 0.0f,1.0f, 0.93f, 0.24f, 1.0f, //1
        1.5f, 0.5f,  0.0f,1.0f, 0.93f, 0.24f, 1.0f, //2
        0.5f, 0.5f,  0.0f,1.0f, 0.93f, 0.24f, 1.0f //3
    };

    unsigned int indices[] = {
        0, 1, 2, 2, 3, 0,

        4, 5, 6, 6, 7, 4
    };

    VertexArray va;
    VertexBuffer vb(positions, 2 * 4 * 7 * sizeof(float));

    VertexBufferLayout layout;
    layout.Push<float>(3);  // every vertex owns 3 floats as the position attribute
    layout.Push<float>(4);  // every vertex owns 4 floats as the color attribute
    va.AddBuffer(vb, layout);

    IndexBuffer ib(indices, 6 * 2);

    glm::mat4 proj = glm::ortho(-2.0f, 2.0f, -1.2f, 1.2f, -1.0f, 1.0f);         // projection matrix
    glm::mat4 view = glm::translate(glm::mat4(1.0f), glm::vec3(-0.4f, 0, 0));   // view matrix; camera right, object left

    Shader shader("/home/ray/Desktop/ORB_SLAM3/src/Renderer/assets");
    shader.Bind();
    shader.SetUnifrom4f("u_Color", 0.8f, 0.3f, 0.8f, 1.0f);
    
    
    va.Unbind();
    vb.Unbind();
    ib.Unbind();
    shader.Unbind();
    Renderer renderer;

    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
	io.ConfigInputTrickleEventQueue = false; // new ImGui event handling seems to make camera controls laggy if this is true.
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 460 core");
    ImGui::StyleColorsDark();

    ORB_SLAM3::Test* currentTest = nullptr;
    ORB_SLAM3::TestMenu* testMenu = new ORB_SLAM3::TestMenu(currentTest);
    currentTest = testMenu;
    ORB_SLAM3::TestClearColor test;

    testMenu->RegisterTest<ORB_SLAM3::TestClearColor>("Clear Color");

    glm::vec3 translationA(0.4f, 0.4f, 0);
    glm::vec3 translationB(-0.4f, 0.4f, 0);

    float r = 0.0f;
    float increment = 0.05f;

    while(!glfwWindowShouldClose(window))
    {
        renderer.Clear();

        test.OnUpdate(0.0f);
        test.OnRender();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Menu test
        if (currentTest)
        {
            currentTest->OnUpdate(0.0f);
            currentTest->OnRender();

            ImGui::Begin("Test");
            if (currentTest != testMenu && ImGui::Button("<-"))
            {
                delete currentTest;
                currentTest = testMenu;
            }
            currentTest->OnImGuiRender();

            ImGui::End();
        }

        shader.Bind();

        {
            glm::mat4 model = glm::translate(glm::mat4(1.0f), translationA);// model matrix
            glm::mat4 mvp = proj * view * model;
            shader.SetUnifrom4f("u_Color", r, 0.3f, 0.8f, 1.0f);
            shader.SetUnifromMat4f("u_MVP", mvp);
            renderer.Draw(va, ib, shader);
        }

        {
            glm::mat4 model = glm::translate(glm::mat4(1.0f), translationB);// model matrix
            glm::mat4 mvp = proj * view * model;
            shader.SetUnifrom4f("u_Color", r, 0.3f, 0.8f, 1.0f);
            shader.SetUnifromMat4f("u_MVP", mvp);
            // renderer.Draw(va, ib, shader);
        }
        

        if (r > 1.0f)
            increment = -0.05f;
        else if (r < 0.0f)
            increment = 0.05f;

        r += increment; 

        {
            ImGui::SliderFloat3("TranslationA", &translationA.x, 0.0f, 2.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
            ImGui::SliderFloat3("TranslationB", &translationB.x, 0.0f, 2.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
            ImGui::End();
        }

        //
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    delete currentTest;
    if(currentTest != testMenu)
        delete testMenu;

    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();
    return 0;
}