#include <torch/torch.h>
#include <deque>

#include "Config.h"
#include "KeyFrame.h"

namespace GaussianSplatting{

struct Expon_lr_func {
    float lr_init;
    float lr_final;
    float lr_delay_steps;
    float lr_delay_mult;
    int64_t max_steps;
    Expon_lr_func(float lr_init = 0.f, float lr_final = 1.f, float lr_delay_mult = 1.f, int64_t max_steps = 1000000, float lr_delay_steps = 0.f)
        : lr_init(lr_init),
          lr_final(lr_final),
          lr_delay_mult(lr_delay_mult),
          max_steps(max_steps),
          lr_delay_steps(lr_delay_steps) {}

    float operator()(int64_t step) const {
        if (step < 0 || (lr_init == 0.0 && lr_final == 0.0)) {
            return 0.0;
        }
        float delay_rate;
        if (lr_delay_steps > 0. && step != 0) {
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * std::sin(0.5 * M_PI * std::clamp((float)step / lr_delay_steps, 0.f, 1.f));
        } else {
            delay_rate = 1.0;
        }
        float t = std::clamp(static_cast<float>(step) / static_cast<float>(max_steps), 0.f, 1.f);
        float log_lerp = std::exp(std::log(lr_init) * (1.f - t) + std::log(lr_final) * t);
        return delay_rate * log_lerp;
    }
};


class LossMonitor {

public:
    explicit LossMonitor(size_t size) : _buffer_size(size) {}
    float Update(float newLoss);
    bool IsConverging(float threshold);

private:
    std::deque<float> _loss_buffer;
    std::deque<float> _rate_of_change_buffer;
    size_t _buffer_size;
};

class GaussianOptimizer
{
public:
    // Constructer
    GaussianOptimizer(const ORB_SLAM3::OptimizationParameters &OptimParams);
    GaussianOptimizer(const ORB_SLAM3::MonoGSOptimizationParameters &OptimParams);

    // Optimization body
    void InitializeOptimization(const std::vector<ORB_SLAM3::KeyFrame *> &vpKFs, const std::vector<ORB_SLAM3::MapPoint *> &vpMP, const bool bInitializeScale);
    void TrainingSetup();
    void Optimize();

    // MonoGS Test
    void InitializeOptimization(ORB_SLAM3::KeyFrame* pKF);
    void InitializeGaussianFromRGBD(float cx, float cy, float fx, float fy);
    void TrainingSetupMonoGS();
    void OptimizeMonoGS();

    // Camera params
    std::pair<torch::Tensor, float> GetNerfppNorm();
    std::pair<torch::Tensor, float> GetCenterAndDiag();

    // Getters
    torch::Tensor GetViewMatrix(Sophus::SE3f &Tcw);
    torch::Tensor SetProjMatrix();
    torch::Tensor SetProjMatrixMonoGS();

    std::vector<int> GetRandomIndices(const int &max_index);
    torch::Tensor GetViewMatrixWithIndex(const int &CamIndex);
    torch::Tensor GetProjMatrixWithIndex(const int &CamIndex);
    torch::Tensor GetCamCenterWithIndex(const int &CamIndex);
    torch::Tensor GetGTImgTensor(const int &CamIndex);

    std::vector<long> GetGaussianRootIndex();

    torch::Tensor GetWorldPos(const torch::Tensor indices);
    torch::Tensor GetWorldRot(const torch::Tensor indices);
    torch::Tensor GetScale(const torch::Tensor indices);
    torch::Tensor GetOpacity(const torch::Tensor indices);
    torch::Tensor GetFeaturest(const torch::Tensor indices);
    torch::Tensor GetFeatureDC(const torch::Tensor indices);

    torch::Tensor GetWorldPos();
    torch::Tensor GetWorldRot();
    torch::Tensor GetScale();
    torch::Tensor GetOpacity();
    torch::Tensor GetFeaturest();
    torch::Tensor GetFeatureDC();

    int GetActiveSHDegree();

    // Converter
    torch::Tensor CVMatToTensor(cv::Mat mat);
    cv::Mat TensorToCVMat(torch::Tensor tensor);
    torch::Tensor RotQuaToMatrix(torch::Tensor r);

    // Loss functions
    torch::Tensor CreateWindow();
    torch::Tensor L1Loss(const torch::Tensor& network_output, const torch::Tensor& gt);
    torch::Tensor SSIM(const torch::Tensor& img1, const torch::Tensor& img2);
    torch::Tensor GaussianKernel1D(int window_size, float sigma);

    // Densification and prune
    void AddDensificationStats(torch::Tensor& viewspace_point_tensor, torch::Tensor& update_filter);
    void DensifyAndPrune(float max_grad, float min_opacity, float max_screen_size);
    void DensifyAndClone(torch::Tensor& grads, float grad_threshold);
    void DensifyAndSplit(torch::Tensor& grads, float grad_threshold, float min_opacity, float max_screen_size);
    void PrunePoints(torch::Tensor mask);

    void DensificationPostfix(torch::Tensor& newMeans3D, torch::Tensor& newFeaturesDC, torch::Tensor& newFeaturesRest,
                              torch::Tensor& newScales, torch::Tensor& newRotation, torch::Tensor& newOpacity);
    void CatTensorstoOptimizer(torch::Tensor& extension_tensor, torch::Tensor& old_tensor, int param_position);
    void PruneOptimizer(torch::Tensor& old_tensor, const torch::Tensor& mask, int param_position);
    void ResetOpacity();
    
    // Learning rate updater
    void UpdateLR(float iteration);

    // SH degree updater
    void UpdateSHDegree();

    // Utils
    inline torch::Tensor InverseSigmoid(torch::Tensor x) {return torch::log(x / (1 - x));}

    // Tree management
    void UpdateIndiceForestAfterClone(const torch::Tensor indices);
    void UpdateIndiceForestAfterSplit(const torch::Tensor indices);
    void UpdateIndiceForestAfterPrune(const torch::Tensor indices);

protected:
    ORB_SLAM3::OptimizationParameters mOptimParams;
    ORB_SLAM3::MonoGSOptimizationParameters mMonoGSOptimParams;

    // Gaussian associated members
    int mSizeofGaussians;
    torch::Tensor mMeans3D;
    torch::Tensor mOpacity;
    torch::Tensor mScales;
    torch::Tensor mRotation;
    torch::Tensor mFeaturesDC;
    torch::Tensor mFeaturesRest;

    // torch::Tensor mMeans2D;                          // reset every iter
    // torch::Tensor mFeatures;                         // reset every iter
    // torch::Tensor mCov3DPrecomp = torch::Tensor();   // reset every iter
    // torch::Tensor mColorsPrecomp = torch::Tensor();  // reset every iter

    // Cameras/KeyFrames associated members
    int mSizeofCameras;
    std::vector<cv::Mat> mTrainedImages;
    std::vector<torch::Tensor> mTrainedImagesTensor;
    std::vector<torch::Tensor> mViewMatrices;
    std::vector<torch::Tensor> mProjMatrices;
    std::vector<torch::Tensor> mCameraCenters;

    // Single Camera/KeyFrame associated members
    cv::Mat mTrainedImage;
    torch::Tensor mTrainedImageTensor;
    torch::Tensor mInitalDepthTensor;
    torch::Tensor mViewMatrix;
    torch::Tensor mFullProjMatrix;
    torch::Tensor mCameraCenter;

    int mImHeight;
    int mImWidth;
    float mTanFovx;
    float mTanFovy;
    float mFx;
    float mFy;
    float mCx;
    float mCy;
    
    torch::Tensor mProjMatrix;
    
    torch::Tensor mBackground = torch::tensor({1.f, 1.f, 1.f}).to(torch::kCUDA);
    bool mWhiteBackground = true;
    float mScaleModifier = 1.f;
    int mSHDegree = 3;
    int mMaxSHDegree = 3;
    int mActiveSHDegree = 0;
    bool mPrefiltered = false;
    float mNear = 0.01f;
    float mFar = 100.0f;

    // Cameras associated members
    float mNerfNormRadius;
    torch::Tensor mNerfNormTranslation;

    // Training associated members
    float mSpatialLRScale = 6.0;
    float mPercentDense;
    torch::Tensor mPosGradientAccum;
    torch::Tensor mDenom;
    Expon_lr_func mPosSchedulerArgs;
    std::unique_ptr<torch::optim::Adam> mOptimizer;
    // torch::Device mDevice;

    // Loss related members
    LossMonitor* mLossMonitor;
    torch::Tensor mSSIMWindow;
    int mWindowSize = 11;
    int mChannel = 3;
    const float mC1 = 0.01 * 0.01;
    const float mC2 = 0.03 * 0.03;

    // For tree management
    std::vector<long> mvpGaussianRootIndex; // Same
    // std::vector<GaussianIndiceTree*> mvpGaussianIndiceForest;
    // std::vector<GaussianIndiceNode*> mvpGaussianIndiceNodes; // correspond to mMeans3D index
    
};

}