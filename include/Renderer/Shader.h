#ifndef SHADER_H
#define SHADER_H

#include <GL/gl.h>
#include <string>
#include <unordered_map>

#include "glm/glm.hpp"

struct ShaderProgramSource
{
    std::string VertexSource;
    std::string FragmentSource;
};

class Shader
{
private:
    std::string m_FilePath;
    unsigned int m_RendererID;
    //caching for uniforms
    mutable std::unordered_map<std::string, GLint> m_UniformLocationCache;

public:
    Shader(const std::string& filepath);
    ~Shader();

    void Bind() const;
    void Unbind() const;

    // Set uniforms
    void SetUnifrom1i(const std::string& name, int value);
    void SetUnifrom4f(const std::string& name, float v0, float v1, float v2, float v3);
    void SetUnifromMat4f(const std::string& name, const glm::mat4& matrix);

private:
    ShaderProgramSource ParseShader(const std::string& filepath);
    unsigned int CompileShader(unsigned int type, const std::string& source);
    unsigned int CreateShader(const std::string& vertexShader, const std::string& fragmentShader);
    GLint GetUniformLocation(const std::string& name) const;
};
#endif