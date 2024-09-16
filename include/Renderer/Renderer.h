#include <GL/glew.h>
#include <cassert>

#define GLCall(x) GLCLearError();x;assert(GLLogCall(#x, __FILE__, __LINE__))

void GLCLearError();
bool GLLogCall(const char* function, const char* file, int line);



