#include "Texture.h"

Texture::Texture(const std::string& path)
    :mRendererID(0), mFilePath(path), mLocalBuffer(nullptr), mWidth(0), mHeight(0), mChannels(0)
{
    cv::Mat TestImg = cv::imread(mFilePath); // BGR order
    
    // cv::cvtColor(TestImg, TestImg, cv::COLOR_BGR2RGB);
    cv::flip(TestImg, TestImg, 0);

    // mLocalBuffer = TestImg.ptr();
    mWidth = TestImg.cols;
    mHeight = TestImg.rows;
    mChannels = TestImg.channels();

    mLocalBuffer = TestImg.ptr(); 

    GLCall(glGenTextures(1, &mRendererID));
    GLCall(glBindTexture(GL_TEXTURE_2D, mRendererID));

    GLCall(glTextureParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
    GLCall(glTextureParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
    GLCall(glTextureParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP));
    GLCall(glTextureParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP));

    GLCall(glTexImage2D(GL_TEXTURE_2D,        // Type of texture
                        0,                    // Pyramid level (for mip-mapping) - 0 is the top level
                        GL_RGB8,              // Internal colour format to convert to
                        mWidth,               // Image width  i.e. 640 for Kinect in standard mode
                        mHeight,              // Image height i.e. 480 for Kinect in standard mode
                        0,                    // Border width in pixels (can either be 1 or 0)
                        GL_BGR,               // Input image format (i.e. GL_RGB, GL_RGBA, GL_BGR etc.)
                        GL_UNSIGNED_BYTE,     // Image data type
                        mLocalBuffer)         // The actual image data itself
    );

    GLCall(glGenerateMipmap(GL_TEXTURE_2D));
    // GLCall(glBindTexture(GL_TEXTURE_2D, 0));
}

Texture::Texture(cv::Mat Img)
{
    cv::flip(Img, Img, 0);

    mWidth = Img.cols;
    mHeight = Img.rows;
    mChannels = Img.channels();

    mLocalBuffer = Img.ptr(); 

    GLCall(glGenTextures(1, &mRendererID));
    GLCall(glBindTexture(GL_TEXTURE_2D, mRendererID));

    GLCall(glTextureParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
    GLCall(glTextureParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
    GLCall(glTextureParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP));
    GLCall(glTextureParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP));

    GLCall(glTexImage2D(GL_TEXTURE_2D,        // Type of texture
                        0,                    // Pyramid level (for mip-mapping) - 0 is the top level
                        GL_RGB8,              // Internal colour format to convert to
                        mWidth,               // Image width  i.e. 640 for Kinect in standard mode
                        mHeight,              // Image height i.e. 480 for Kinect in standard mode
                        0,                    // Border width in pixels (can either be 1 or 0)
                        GL_BGR,               // Input image format (i.e. GL_RGB, GL_RGBA, GL_BGR etc.)
                        GL_UNSIGNED_BYTE,     // Image data type
                        mLocalBuffer)         // The actual image data itself
    );

    GLCall(glGenerateMipmap(GL_TEXTURE_2D));
}

Texture::~Texture()
{
    GLCall(glDeleteTextures(1, &mRendererID));
}

void Texture::Bind(unsigned int slot) const
{
    GLCall(glActiveTexture(GL_TEXTURE0 + slot));
    GLCall(glBindTexture(GL_TEXTURE_2D, mRendererID));
}

void Texture::Unbind() const
{
    GLCall(glBindTexture(GL_TEXTURE_2D, 0));
}