#include "RenderGUI.h"

namespace ORB_SLAM3
{

RenderGUI::RenderGUI()
{
    

}

RenderGUI::~RenderGUI()
{
    
}

bool RenderGUI::InitializeWidget()
{
    //ImGUI test
    IMGUI_CHECKVERSION();

    GLFWwindow* window;
    
    /* Initialize the library */
    if(!glfwInit())
        return -1;

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(1600, 960, "Gaussian Renderer Test", NULL, NULL);
    if(!window)
    {
        glfwTerminate();
        return -1;
    }

    /* Make the window's context current */
    glfwMakeContextCurrent(window);

    if(glewInit() != GLEW_OK)
        std::cout << "GLEW Error!" << std::endl;

    /* Loop until the user closes the window*/
    while(!glfwWindowShouldClose(window))
    {
        /* Render here */
        glClear(GL_COLOR_BUFFER_BIT);

        glBegin(GL_TRIANGLES);
        glVertex2f(-0.5f, 0.5f);
        glVertex2f(0.0f, 0.5f);
        glVertex2f(0.5f, -0.5f);
        glEnd();

        /* Swap front and back buffers */
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
 
    glfwTerminate();
    exit(EXIT_SUCCESS);
}
}