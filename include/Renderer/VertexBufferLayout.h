#ifndef VERTEXBUFFERLAYOUT_H
#define VERTEXBUFFERLAYOUT_H

#pragma once

#include <vector>
#include <GL/glew.h>
#include <cassert>

struct VertexBufferElement
{
    unsigned int type;
    unsigned int count;
    unsigned char normalized;

    static unsigned int GetSizeofType( unsigned int type)
    {
        switch (type)
        {
            case GL_FLOAT:          return 4;
            case GL_UNSIGNED_INT:   return 4;
            case GL_UNSIGNED_BYTE:  return 1;
        }
        assert(false);
        return 0;
    }
};

class VertexBufferLayout
{
private:
    std::vector<VertexBufferElement> m_Elements;
    unsigned int m_Stride;
public:
    VertexBufferLayout():m_Stride(0){}

    inline unsigned int GetStride() const {return m_Stride;}
    inline const std::vector<VertexBufferElement> GetElements() const{return m_Elements;}

    template< typename T >
    void Push(unsigned int count, unsigned char normalized = GL_FALSE){ Push_(count, normalized, (T*)0); }

    void Push_(unsigned int count, unsigned char normalized, float*){
        m_Elements.push_back({GL_FLOAT, count, GL_FALSE});
        m_Stride += count * VertexBufferElement::GetSizeofType(GL_FLOAT);
    }

    void Push_(unsigned int count, unsigned char normalized, unsigned int*){
        m_Elements.push_back({GL_UNSIGNED_INT, count, GL_FALSE});
        m_Stride += count * VertexBufferElement::GetSizeofType(GL_UNSIGNED_INT);
    }

    void Push_(unsigned int count, unsigned char normalized, unsigned char*){
        m_Elements.push_back({GL_UNSIGNED_BYTE, count, GL_TRUE});
        m_Stride += count * VertexBufferElement::GetSizeofType(GL_UNSIGNED_BYTE);
    }
};

#endif



