
#ifndef GAUSSIANRENDERER_H
#define GAUSSIANRENDERER_H
#include <vector>

#include "MapGaussian.h"

namespace ORB_SLAM3
{

class MapGaussian;

class GaussianRenderer
{
private:
    std::vector<MapGaussian*> MapGaussians;

};

} //namespace ORB_SLAM3

#endif // GAUSSIANRENDERER_H