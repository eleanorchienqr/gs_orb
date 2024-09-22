#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>

#include "Renderer.h"
#include "VertexBuffer.h"
#include "IndexBuffer.h"
#include "VertexArray.h"
#include "VertexBufferLayout.h"
#include "Shader.h"

// #include "glm/glm.hpp"
// #include "glm/gtc/matrix_transform.hpp"

// #include <Thirdparty/imgui/imgui.h>
// #include <Thirdparty/imgui/backends/imgui_impl_glfw.h>
// #include <Thirdparty/imgui/backends/imgui_impl_opengl3.h>

// int main(void)
// {
//     GLFWwindow* window;

//     /* Initialize the library */
//     if (!glfwInit()) return -1;

//     // glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
//     // glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
//     // glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

//     /* Create a windowed mode window and its OpenGL context */
//     window = glfwCreateWindow(640, 480, "Hello World", NULL, NULL);
//     if (!window) {
//         glfwTerminate();
//         return -1;
//     }

//     /* Make the window's context current */
//     glfwMakeContextCurrent(window);
//     glfwSwapInterval(1);

//     GLenum err = glewInit();
//     if (GLEW_OK != err) {
//         std::cout << "Error: " << glewGetErrorString(err) << std::endl;
//     }
//     std::cout << "Status: Using GLEW " << glewGetString(GLEW_VERSION) << std::endl;
    
//     unsigned char* glVersion;
//     GLCall(glVersion = (unsigned char*)glGetString(GL_VERSION));
//     std::cout << "Status: Using GL " << glVersion << std::endl;

//     GLCall(glEnable(GL_BLEND));
//     GLCall(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));

//     float positions[] = {
//         -0.5f, -0.5f, //0
//         0.0f, 0.5f,  //1
//         0.5f, -0.5f,   //2
//     };

//     unsigned int buffer;
//     glGenBuffers(1, &buffer);
//     glBindBuffer(GL_ARRAY_BUFFER, buffer);
//     glBufferData(GL_ARRAY_BUFFER, 6 * sizeof(float), positions, GL_STATIC_DRAW);

//     glEnableVertexAttribArray(0);
//     glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float)*2, 0);


//     /* Loop until the user closes the window */
//     while (!glfwWindowShouldClose(window))
//     {
//         /* Render here */
//         glClear(GL_COLOR_BUFFER_BIT);

//         glDrawArrays(GL_TRIANGLES, 0, 3);
//         /* Swap front and back buffers */
//         glfwSwapBuffers(window);

//         /* Poll for and process events */
//         glfwPollEvents();
//     }

//     glfwTerminate();
//     return 0;
// }

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

    GLCall(glEnable(GL_BLEND));
    GLCall(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));

    float positions[] = {
        -0.5f, -0.5f, //0
        0.5f, -0.5f,  //1
        0.5f, 0.5f,   //2
        -0.5f, 0.5f   //3
    };

    unsigned int indices[] = {
        0, 1, 2, 
        2, 3, 0
    };

    VertexArray va;
    VertexBuffer vb(positions, 4 * 2 * sizeof(float));

    VertexBufferLayout layout;
    layout.Push<float>(2);  // every vertex owns 2 floats as the position attribute
    va.AddBuffer(vb, layout);

    IndexBuffer ib(indices, 6);

    Shader shader("/home/ray/Desktop/ORB_SLAM3/src/Renderer/assets");
    shader.Bind();
    // shader.SetUnifrom4f("u_Color", 0.8f, 0.3f, 0.8f, 1.0f);
    
    va.Unbind();
    vb.Unbind();
    ib.Unbind();
    shader.Unbind();
    Renderer renderer;

    float r = 0.0f;
    float increment = 0.05f;

    while(!glfwWindowShouldClose(window))
    {
        renderer.Clear();

        shader.Bind();
        // shader.SetUnifrom4f("u_Color", 0.8f, 0.3f, 0.8f, 1.0f);
        // shader.SetUnifrom4f("u_Color", r, 0.3f, 0.8f, 1.0f);

        renderer.Draw(va, ib, shader);

        if (r > 1.0f)
            increment = -0.05f;
        else if (r < 0.0f)
            increment = 0.05f;

        r += increment; 

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glfwTerminate();
    return 0;
}