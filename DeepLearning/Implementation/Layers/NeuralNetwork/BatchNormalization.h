#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"

class BatchNormalization : public Layer {
   public:

    virtual ~BatchNormalization() {}

    void setTrainingMode(bool training) {
        assert(running == false);
        this->training = training;
    }

    BatchNormalization(bool training)
        : training(training) {}

    virtual void compile() {
        cudnnStatus_t cudnnStatus;

        assert(featureInput.isPresent());
        assert(featureOutput.isPresent());

        featureInputDescriptor = cudnnTensorDescriptor_t();
        cudnnStatus = cudnnCreateTensorDescriptor(&featureInputDescriptor.get());
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
        cudnnStatus = cudnnSetTensor4dDescriptor(
            featureInputDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, fix batchSize, numFeatures, inputHeight, inputWidth);
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);

        featureOutputDescriptor = cudnnTensorDescriptor_t();
        cudnnStatus = cudnnCreateTensorDescriptor(&featureOutputDescriptor.get());
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
        cudnnStatus = cudnnSetTensor4dDescriptor(
            featureOutputDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, fix batchSize, numFeatures, outputHeight, outputWidth);
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);

        cudnnStatus = cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(stream.getCudnnHandle(),
                                                                               cudnnBatchNormMode_t                    mode,
                                                                               cudnnBatchNormOps_t                     bnOps,
                                                                               const cudnnTensorDescriptor_t           xDesc,
                                                                               const cudnnTensorDescriptor_t           zDesc,
                                                                               const cudnnTensorDescriptor_t           yDesc, 
                                                                               const cudnnTensorDescriptor_t           bnScaleBiasMeanVarDesc,
                                                                               const cudnnActivationDescriptor_t       activationDesc, 
                                                                               size_t                                  *sizeInBytes);
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);

        cudnnStatus = cudnnGetBatchNormalizationTrainingExReserveSpaceSize(stream.getCudnnHandle(),
                                                                           cudnnBatchNormMode_t                mode,
                                                                           cudnnBatchNormOps_t                 bnOps,
                                                                           const cudnnActivationDescriptor_t   activationDesc, 
                                                                           const cudnnTensorDescriptor_t       xDesc, 
                                                                           size_t                              *sizeInBytes);
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);

        cudnnStatus = cudnnGetBatchNormalizationBackwardExWorkspaceSize(stream.getCudnnHandle(),
                                                                        cudnnBatchNormMode_t                mode,
                                                                        cudnnBatchNormOps_t                 bnOps,
                                                                        const cudnnTensorDescriptor_t       xDesc,
                                                                        const cudnnTensorDescriptor_t       yDesc, 
                                                                        const cudnnTensorDescriptor_t       dyDesc,
                                                                        const cudnnTensorDescriptor_t       dzDesc, 
                                                                        const cudnnTensorDescriptor_t       dxDesc,
                                                                        const cudnnTensorDescriptor_t       dBnScaleBiasDesc,
                                                                        const cudnnActivationDescriptor_t   activationDesc, 
                                                                        size_t                              *sizeInBytes);
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
    }

    void cleanup() {
        cudnnStatus_t cudnnStatus;

        if (poolingDescriptor.isPresent()) {
            cudnnStatus = cudnnDestroyPoolingDescriptor(poolingDescriptor.get());
            assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
            poolingDescriptor.clear();
        }

        if (featureInputDescriptor.isPresent()) {
            cudnnStatus = cudnnDestroyTensorDescriptor(featureInputDescriptor.get());
            assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
            featureInputDescriptor.clear();
        }

        if (featureOutputDescriptor.isPresent()) {
            cudnnStatus = cudnnDestroyTensorDescriptor(featureOutputDescriptor.get());
            assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
            featureOutputDescriptor.clear();
        }
    }

    virtual void infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream) {
        assert(inputTensor.isPresent());
        assert(outputTensor.isPresent());

        cudnnStatus_t cudnnStatus;

        if(training) {
            cudnnStatus = cudnnBatchNormalizationForwardTrainingEx(stream.getCudnnHandle(),
                                                                    cudnnBatchNormMode_t                mode,
                                                                    cudnnBatchNormOps_t                 bnOps,
                                                                    const void                          *alpha,
                                                                    const void                          *beta,
                                                                    const cudnnTensorDescriptor_t       xDesc,
                                                                    const void                          *xData,
                                                                    const cudnnTensorDescriptor_t       zDesc, 
                                                                    const void                          *zData, 
                                                                    const cudnnTensorDescriptor_t       yDesc,
                                                                    void                                *yData,
                                                                    const cudnnTensorDescriptor_t       bnScaleBiasMeanVarDesc,
                                                                    const void                          *bnScaleData,
                                                                    const void                          *bnBiasData, 
                                                                    double                              exponentialAverageFactor,
                                                                    void                                *resultRunningMeanData,
                                                                    void                                *resultRunningVarianceData,
                                                                    double                              epsilon,
                                                                    void                                *saveMean,
                                                                    void                                *saveInvVariance,
                                                                    const cudnnActivationDescriptor_t   activationDesc, 
                                                                    void                                *workspace,
                                                                    size_t                              workSpaceSizeInBytes
                                                                    void                                *reserveSpace
                                                                    size_t                              reserveSpaceSizeInBytes);
            assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
        } else {
            cudnnStatus = cudnnBatchNormalizationForwardInference(stream.getCudnnHandle(),
                                                                  cudnnBatchNormMode_t             mode,
                                                                  const void                      *alpha,
                                                                  const void                      *beta,
                                                                  const cudnnTensorDescriptor_t    xDesc,
                                                                  const void                      *x,
                                                                  const cudnnTensorDescriptor_t    yDesc,
                                                                  void                            *y,
                                                                  const cudnnTensorDescriptor_t    bnScaleBiasMeanVarDesc,
                                                                  const void                      *bnScale,
                                                                  const void                      *bnBias,
                                                                  const void                      *estimatedMean,
                                                                  const void                      *estimatedVariance,
                                                                  double                           epsilon);
            assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
        }
    }

    virtual void backProp(Optional<Tensor> dataIn, Optional<Tensor> errorIn, Optional<Tensor> errorOut, Stream stream) {
        if (errorOut.isEmpty())
            return;
        assert(errorIn.isPresent());

        cudnnStatus_t cudnnStatus;
        cudnnStatus = cudnnBatchNormalizationBackwardEx(stream.getCudnnHandle(),
                                                        cudnnBatchNormMode_t                mode,
                                                        cudnnBatchNormOps_t                 bnOps,
                                                        const void                          *alphaDataDiff,
                                                        const void                          *betaDataDiff,
                                                        const void                          *alphaParamDiff,
                                                        const void                          *betaParamDiff,
                                                        const cudnnTensorDescriptor_t       xDesc,
                                                        const void                          *xData,
                                                        const cudnnTensorDescriptor_t       yDesc,
                                                        const void                          *yData,
                                                        const cudnnTensorDescriptor_t       dyDesc,
                                                        const void                          *dyData,
                                                        const cudnnTensorDescriptor_t       dzDesc,
                                                        void                                *dzData,
                                                        const cudnnTensorDescriptor_t       dxDesc,
                                                        void                                *dxData,
                                                        const cudnnTensorDescriptor_t       dBnScaleBiasDesc,
                                                        const void                          *bnScaleData,
                                                        const void                          *bnBiasData,
                                                        void                                *dBnScaleData,
                                                        void                                *dBnBiasData,
                                                        double                              epsilon,
                                                        const void                          *savedMean,
                                                        const void                          *savedInvVariance,
                                                        const cudnnActivationDescriptor_t   activationDesc,
                                                        void                                *workspace,
                                                        size_t                              workSpaceSizeInBytes
                                                        void                                *reserveSpace
                                                        size_t                              reserveSpaceSizeInBytes);
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
    }

   private:
    static const float ALPHA_NO_SCALE;
    static const float BETA_CLEAR;

    bool training;

    Optional<cudnnTensorDescriptor_t> featureOutputDescriptor;
    Optional<cudnnTensorDescriptor_t> featureInputDescriptor;
};
