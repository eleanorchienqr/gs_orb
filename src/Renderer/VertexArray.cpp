#include "VertexArray.h"
#include "Renderer.h"

#include <iostream>

VertexArray::VertexArray()
{
    GLCall(glGenVertexArrays(1, &m_Renderer_ID));
}

VertexArray::~VertexArray()
{
    GLCall(glDeleteVertexArrays(1, &m_Renderer_ID));
}

void VertexArray::AddBuffer(const VertexBuffer& vb, const VertexBufferLayout& layout)
{
    Bind();
    vb.Bind();
    const auto& elements = layout.GetElements();
    unsigned int offset = 0;
    for(unsigned int i = 0; i < elements.size(); i++)
    {
        const auto& element = elements[i];

        std::cout << "element.count: " << element.count << std::endl;     // size of one attribute
        std::cout << "elements.size(): " << elements.size() << std::endl; // numbers of attributes

        GLCall(glEnableVertexAttribArray(i));
        GLCall(glVertexAttribPointer(i, element.count, element.type, element.normalized, layout.GetStride(), (const void*)offset));
        offset += element.count * VertexBufferElement::GetSizeofType(element.type);
    }
}

void VertexArray::Bind() const
{
    GLCall(glBindVertexArray(m_Renderer_ID));
}


void VertexArray::Unbind() const
{
     GLCall(glBindVertexArray(0));
}