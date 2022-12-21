#pragma once

#include "ConvolutionKernelRequirement.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Optional.h"
#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/ComputeTopology/MachineEvaluator.h"

#include <cuda.h>
#include <cuda_fp16.h>
#include <cudnn.h>

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace ThorImplementation {

/**
 * GpuConvolution is a singleton object that can find the optimal convolution kernel for a convolution operation with a given set of
 * parameters.
 *
 * Once the optimal kernel is found, it will be use on all subsequent matching convolution calls.
 *
 * It is not required to first evaluate the optimal kernel, when the optimal kernel is not known then a pretty-good-fit kernel can be used.
 *
 * The recommendation is that if you can know before hand the parameters of the operation, and you will be using this operation many times,
 * then first find the optimal kernel.
 *
 * The singleton object is accessed as GpuConvolution::instance().convolutionForward(...)
 *
 * Note: By having the member functions be non-static, they cannot be called unless the constructor has already been called.
 *       Member variables should be non-static so that they are only initialized upon creation of the singleton instance.
 */
class GpuConvolution {
   public:
    static GpuConvolution &instance() {
        static GpuConvolution singletonInstance;  // Guaranteed to be destroyed. Instantiated on first use.
        return singletonInstance;
    }

    virtual ~GpuConvolution() {}

    // Finds the optimal forward kernel for the convolution operation given the parameters
    void chooseOptimalKernelForward(ConvolutionKernelRequirement convolutionKernelRequirement, Stream stream);
    // Finds the optimal backwardData and backwardFilter kernel for the convolution operation given the parameters
    void chooseOptimalKernelBackward(ConvolutionKernelRequirement convolutionKernelRequirement, Stream stream);

    uint64_t getForwardWorkspaceSizeInBytes(ConvolutionKernelRequirement convolutionKernelRequirement);
    uint64_t getBackwardDataWorkspaceSizeInBytes(ConvolutionKernelRequirement convolutionKernelRequirement);
    uint64_t getBackwardFilterWorkspaceSizeInBytes(ConvolutionKernelRequirement convolutionKernelRequirement);
    uint64_t getBackwardBiasWorkspaceSizeInBytes(ConvolutionKernelRequirement convolutionKernelRequirement);

    void convolutionForward(ConvolutionKernelRequirement convolutionKernelRequirement,
                            Tensor dataInput,
                            Tensor weights,
                            Optional<Tensor> biases,
                            Tensor dataOutput,
                            Optional<Tensor> workspace,
                            Stream stream);
    void convolutionBackwardData(ConvolutionKernelRequirement convolutionKernelRequirement,
                                 Tensor errorInput,
                                 Tensor weights,
                                 Tensor errorOutput,
                                 Optional<Tensor> workspace,
                                 Stream stream);
    void convolutionBackwardFilter(ConvolutionKernelRequirement convolutionKernelRequirement,
                                   Tensor dataInput,
                                   Tensor errorInput,
                                   Tensor weightsGradient,
                                   Optional<Tensor> workspace,
                                   Stream stream,
                                   bool accumulateGradient);
    void convolutionBackwardBias(ConvolutionKernelRequirement convolutionKernelRequirement,
                                 Tensor errorInput,
                                 Tensor biasesGradient,
                                 Optional<Tensor> workspace,
                                 Stream stream,
                                 bool accumulateGradient);

    void printBackwardFilterKernelInfo(ConvolutionKernelRequirement convolutionKernelRequirement);

   private:
    std::mutex forwardMutex;
    std::mutex backwardDataMutex;
    std::mutex backwardFilterMutex;

    std::unordered_map<ConvolutionKernelRequirement, cudnnConvolutionFwdAlgoPerf_t> optimalForwardKernels;
    std::unordered_map<ConvolutionKernelRequirement, cudnnConvolutionBwdDataAlgoPerf_t> optimalBackwardDataKernels;
    std::unordered_map<ConvolutionKernelRequirement, cudnnConvolutionBwdFilterAlgoPerf_t> optimalBackwardFilterKernels;

    static constexpr int MAX_ALGOS = 5000;

    static constexpr bool useCudnnForwardBias = true;
    static constexpr bool useCudnnBackwardBias = true;

    GpuConvolution();

    void chooseOptimalKernelBackwardData(ConvolutionKernelRequirement convolutionKernelRequirement, Stream stream);
    void chooseOptimalKernelBackwardFilter(ConvolutionKernelRequirement convolutionKernelRequirement, Stream stream);

    static const float ALPHA_NO_SCALE;
    static const float BETA_CLEAR;
    static const float BETA_ACCUMULATE;

    void addConvolutionBias(Tensor dataOutput, Tensor biases, Stream stream);
    void computeConvolutionBiasesGradient(Tensor errorInput, Tensor biasesGradient, Tensor workspace, Stream stream);
};

}  // namespace ThorImplementation
