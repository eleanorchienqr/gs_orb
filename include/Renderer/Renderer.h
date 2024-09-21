#ifndef RENDER_H
#define RENDER_H

#include <GL/glew.h>
#include <cassert>

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

#endif

