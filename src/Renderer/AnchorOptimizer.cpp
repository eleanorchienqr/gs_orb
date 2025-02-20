#include "Renderer/AnchorOptimizer.h"
#include "Renderer/Rasterizer.h"

#include <Thirdparty/simple-knn/spatial.h>

namespace GaussianSplatting{
    // AnchorOptimizer::AnchorOptimizer(int SizeofInitAnchors, const int AnchorFeatureDim, const int AnchorSizeofOffsets, const int CamNum, 
    //                 torch::Tensor AnchorWorldPos, torch::Tensor AnchorFeatures, torch::Tensor AnchorScales, 
    //                 torch::Tensor AnchorRotations, torch::Tensor AnchorOffsets,
    //                 ORB_SLAM3::FeatureBankMLP FBNet, ORB_SLAM3::OpacityMLP OpacityNet, ORB_SLAM3::CovarianceMLP CovNet, ORB_SLAM3::ColorMLP ColorNet,
    //                 const int ImHeight, const int ImWidth, const float TanFovx, const float TanFovy,
    //                 std::vector<torch::Tensor> ViewMatrices, std::vector<cv::Mat> TrainedImages):
    //                 mSizeofAnchors(SizeofInitAnchors), mFeatureDim(AnchorFeatureDim), mSizeofOffsets(AnchorSizeofOffsets), mSizeofCameras(CamNum),
    //                 mAchorPos(AnchorWorldPos), mAchorFeatures(AnchorFeatures), mAchorScales(AnchorScales), mAchorRotations(AnchorRotations), mOffsets(AnchorOffsets),
    //                 mFeatureMLP(FBNet),mOpacityMLP(OpacityNet), mCovarianceMLP(CovNet), mColorMLP(ColorNet),
    //                 mImHeight(ImHeight), mImWidth(ImWidth), mTanFovx(TanFovx), mTanFovy(TanFovy), 
    //                 mViewMatrices(ViewMatrices), mTrainedImages(TrainedImages)
    //                 {

    //                 }

}