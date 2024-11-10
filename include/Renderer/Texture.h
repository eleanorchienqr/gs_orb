#include "Renderer.h"
#include <opencv2/opencv.hpp>

class Texture
{
private:
    unsigned int mRendererID;
    std::string mFilePath;
    unsigned char* mLocalBuffer;
    int mWidth, mHeight, mChannels;

public:
    Texture(const std::string& path);
    Texture(cv::Mat Img);
    
    ~Texture();

    void Bind(unsigned int slot = 0) const;
    void Unbind() const;

    inline int GetWidth() const {return mWidth;}
    inline int GetHeight() const {return mHeight;}

};