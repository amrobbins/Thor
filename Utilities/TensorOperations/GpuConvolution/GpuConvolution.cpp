#include "GpuConvolution.h"

const float GpuConvolution::ALPHA_NO_SCALE = 1.0f;
const float GpuConvolution::BETA_CLEAR = 0.0f;
const float GpuConvolution::BETA_ACCUMULATE = 1.0f;

GpuConvolution::GpuConvolution() {}

// Finds the optimal forward kernel for the convolution operation given the parameters
void GpuConvolution::chooseOptimalKernelForward(ConvolutionKernelRequirement convolutionKernelRequirement, Stream stream) {
    GpuConvolution::instance().forwardMutex.lock();

    // Will only evaluate kernel once per gpu type
    // First check in memory.
    // If not in memory, check in file-based map
    // If not in file based map, measure latency to determine optimal kernel
    if (GpuConvolution::instance().optimalForwardKernels.count(convolutionKernelRequirement) == 1) {
        GpuConvolution::instance().forwardMutex.unlock();
        return;
    }

    cudnnConvolutionFwdAlgoPerf_t dummy;
    optimalForwardKernels[convolutionKernelRequirement] = dummy;

    GpuConvolution::instance().forwardMutex.unlock();

    ScopedGpu scopedGpu(stream.getGpuNum());

    // Evaluate best kernel
    cudnnStatus_t cudnnStatus;
    int returnedAlgoCount;
    cudnnConvolutionFwdAlgoPerf_t perfResults[MAX_ALGOS];
    cudnnStatus = cudnnFindConvolutionForwardAlgorithm(stream.getCudnnHandle(),
                                                       convolutionKernelRequirement.getDataInputTensorDescriptor(),
                                                       convolutionKernelRequirement.getWeightsFilterDescriptor(),
                                                       convolutionKernelRequirement.getConvolutionDescriptor(),
                                                       convolutionKernelRequirement.getDataOutputTensorDescriptor(),
                                                       MAX_ALGOS,
                                                       &returnedAlgoCount,
                                                       perfResults);
    assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
    assert(returnedAlgoCount > 0);

    GpuConvolution::instance().forwardMutex.lock();

    assert(perfResults[0].status == CUDNN_STATUS_SUCCESS);
    optimalForwardKernels[convolutionKernelRequirement] = perfResults[0];

    GpuConvolution::instance().forwardMutex.unlock();
}

// Finds the optimal backwardData and backwardFilter kernel for the convolution operation given the parameters
void GpuConvolution::chooseOptimalKernelBackward(ConvolutionKernelRequirement convolutionKernelRequirement, Stream stream) {
    GpuConvolution::instance().chooseOptimalKernelBackwardData(convolutionKernelRequirement, stream);
    GpuConvolution::instance().chooseOptimalKernelBackwardFilter(convolutionKernelRequirement, stream);
}

void GpuConvolution::chooseOptimalKernelBackwardData(ConvolutionKernelRequirement convolutionKernelRequirement, Stream stream) {
    GpuConvolution::instance().backwardDataMutex.lock();

    // Will only evaluate kernel once per gpu type
    // First check in memory.
    // If not in memory, check in file-based map
    // If not in file based map, measure latency to determine optimal kernel
    if (GpuConvolution::instance().optimalBackwardDataKernels.count(convolutionKernelRequirement) == 1) {
        GpuConvolution::instance().backwardDataMutex.unlock();
        return;
    }

    cudnnConvolutionBwdDataAlgoPerf_t dummy;
    optimalBackwardDataKernels[convolutionKernelRequirement] = dummy;

    GpuConvolution::instance().backwardDataMutex.unlock();

    ScopedGpu scopedGpu(stream.getGpuNum());

    // Evaluate best kernel
    cudnnStatus_t cudnnStatus;
    int returnedAlgoCount;
    cudnnConvolutionBwdDataAlgoPerf_t perfResults[MAX_ALGOS];
    cudnnStatus = cudnnFindConvolutionBackwardDataAlgorithm(stream.getCudnnHandle(),
                                                            convolutionKernelRequirement.getWeightsFilterDescriptor(),
                                                            convolutionKernelRequirement.getErrorInputTensorDescriptor(),
                                                            convolutionKernelRequirement.getConvolutionDescriptor(),
                                                            convolutionKernelRequirement.getErrorOutputTensorDescriptor(),
                                                            MAX_ALGOS,
                                                            &returnedAlgoCount,
                                                            perfResults);
    assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
    assert(returnedAlgoCount > 0);

    GpuConvolution::instance().backwardDataMutex.lock();

    optimalBackwardDataKernels[convolutionKernelRequirement] = perfResults[0];

    GpuConvolution::instance().backwardDataMutex.unlock();
}

void GpuConvolution::chooseOptimalKernelBackwardFilter(ConvolutionKernelRequirement convolutionKernelRequirement, Stream stream) {
    GpuConvolution::instance().backwardFilterMutex.lock();

    // Will only evaluate kernel once per gpu type
    // First check in memory.
    // If not in memory, check in file-based map
    // If not in file based map, measure latency to determine optimal kernel
    if (GpuConvolution::instance().optimalBackwardFilterKernels.count(convolutionKernelRequirement) == 1) {
        GpuConvolution::instance().backwardFilterMutex.unlock();
        return;
    }

    cudnnConvolutionBwdFilterAlgoPerf_t dummy;
    optimalBackwardFilterKernels[convolutionKernelRequirement] = dummy;

    GpuConvolution::instance().backwardFilterMutex.unlock();

    ScopedGpu scopedGpu(stream.getGpuNum());

    // Evaluate best kernel
    cudnnStatus_t cudnnStatus;
    int returnedAlgoCount;
    cudnnConvolutionBwdFilterAlgoPerf_t perfResults[MAX_ALGOS];
    cudnnStatus = cudnnFindConvolutionBackwardFilterAlgorithm(stream.getCudnnHandle(),
                                                              convolutionKernelRequirement.getDataInputTensorDescriptor(),
                                                              convolutionKernelRequirement.getErrorInputTensorDescriptor(),
                                                              convolutionKernelRequirement.getConvolutionDescriptor(),
                                                              convolutionKernelRequirement.getWeightsGradientFilterDescriptor(),
                                                              MAX_ALGOS,
                                                              &returnedAlgoCount,
                                                              perfResults);
    assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
    assert(returnedAlgoCount > 0);

    GpuConvolution::instance().backwardFilterMutex.lock();

    optimalBackwardFilterKernels[convolutionKernelRequirement] = perfResults[0];

    GpuConvolution::instance().backwardFilterMutex.unlock();
}

uint64_t GpuConvolution::getForwardWorkspaceSizeInBytes(ConvolutionKernelRequirement convolutionKernelRequirement) {
    // chooseOptimalKernelForward(...) must first be called for this kernel requirement
    GpuConvolution::instance().forwardMutex.lock();
    assert(optimalForwardKernels.count(convolutionKernelRequirement) == 1);
    uint64_t workspaceSize = optimalForwardKernels[convolutionKernelRequirement].memory;
    GpuConvolution::instance().forwardMutex.unlock();
    return workspaceSize;
}

uint64_t GpuConvolution::getBackwardDataWorkspaceSizeInBytes(ConvolutionKernelRequirement convolutionKernelRequirement) {
    // chooseOptimalKernelBackwardData(...) must first be called for this kernel requirement
    GpuConvolution::instance().backwardDataMutex.lock();
    assert(optimalBackwardDataKernels.count(convolutionKernelRequirement) == 1);
    uint64_t workspaceSize = optimalBackwardDataKernels[convolutionKernelRequirement].memory;
    GpuConvolution::instance().backwardDataMutex.unlock();
    return workspaceSize;
}

uint64_t GpuConvolution::getBackwardFilterWorkspaceSizeInBytes(ConvolutionKernelRequirement convolutionKernelRequirement) {
    // chooseOptimalKernelBackwardFilter(...) must first be called for this kernel requirement
    GpuConvolution::instance().backwardFilterMutex.lock();
    assert(optimalBackwardFilterKernels.count(convolutionKernelRequirement) == 1);
    uint64_t workspaceSize = optimalBackwardFilterKernels[convolutionKernelRequirement].memory;
    GpuConvolution::instance().backwardFilterMutex.unlock();
    return workspaceSize;
}

uint64_t GpuConvolution::getBackwardBiasWorkspaceSizeInBytes(ConvolutionKernelRequirement convolutionKernelRequirement) {
    return convolutionKernelRequirement.getBatchSize() * convolutionKernelRequirement.getNumOutputChannels() * sizeof(float);
}

void GpuConvolution::convolutionForward(ConvolutionKernelRequirement convolutionKernelRequirement,
                                        Tensor dataInput,
                                        Tensor weights,
                                        Optional<Tensor> biases,
                                        Tensor dataOutput,
                                        Optional<Tensor> workspace,
                                        Stream stream) {
    assert(dataInput.getPlacement() == weights.getPlacement());
    assert(dataInput.getPlacement() == dataOutput.getPlacement());

    GpuConvolution::instance().forwardMutex.lock();
    assert(optimalForwardKernels.count(convolutionKernelRequirement) == 1);
    cudnnConvolutionFwdAlgoPerf_t optimalKernel = optimalForwardKernels[convolutionKernelRequirement];
    GpuConvolution::instance().forwardMutex.unlock();

    if (optimalKernel.memory > 0) {
        assert(workspace.isPresent());
        assert(optimalKernel.memory == workspace.get().getDescriptor().getArraySizeInBytes());
    }

    ScopedGpu scopedGpu(stream.getGpuNum());

    cudnnStatus_t cudnnStatus;
    cudnnStatus = cudnnConvolutionForward(stream.getCudnnHandle(),
                                          &ALPHA_NO_SCALE,
                                          convolutionKernelRequirement.getDataInputTensorDescriptor(),
                                          dataInput.getMemPtr(),
                                          convolutionKernelRequirement.getWeightsFilterDescriptor(),
                                          weights.getMemPtr(),
                                          convolutionKernelRequirement.getConvolutionDescriptor(),
                                          optimalKernel.algo,
                                          workspace.isPresent() ? workspace.get().getMemPtr() : nullptr,
                                          optimalKernel.memory,
                                          &BETA_CLEAR,
                                          convolutionKernelRequirement.getDataOutputTensorDescriptor(),
                                          dataOutput.getMemPtr());
    if (cudnnStatus == 3) {
        // This looks like an issue with cudnn. Better to choose a suboptimal working algorithm than to crash...
        int maxAlgoCount;
        cudnnStatus = cudnnGetConvolutionForwardAlgorithmMaxCount(stream.getCudnnHandle(), &maxAlgoCount);
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);

        cudnnConvolutionFwdAlgoPerf_t *perfResults = new cudnnConvolutionFwdAlgoPerf_t[maxAlgoCount];

        int returnedAlgoCount;
        cudnnStatus = cudnnGetConvolutionForwardAlgorithm_v7(stream.getCudnnHandle(),
                                                             convolutionKernelRequirement.getDataInputTensorDescriptor(),
                                                             convolutionKernelRequirement.getWeightsFilterDescriptor(),
                                                             convolutionKernelRequirement.getConvolutionDescriptor(),
                                                             convolutionKernelRequirement.getDataOutputTensorDescriptor(),
                                                             maxAlgoCount,
                                                             &returnedAlgoCount,
                                                             perfResults);
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);

        // Choose the best algo with no workspace
        int i;
        for (i = 0; i < maxAlgoCount; ++i) {
            if (perfResults[i].status != CUDNN_STATUS_SUCCESS)
                continue;
            if (perfResults[i].memory != 0)
                continue;
            break;
        }
        if (i < maxAlgoCount) {
            GpuConvolution::instance().forwardMutex.lock();
            optimalKernel = perfResults[i];
            optimalForwardKernels[convolutionKernelRequirement] = optimalKernel;
            GpuConvolution::instance().forwardMutex.unlock();
        } else {
            GpuConvolution::instance().forwardMutex.lock();
            optimalKernel.algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
            optimalKernel.memory = 0;
            optimalForwardKernels[convolutionKernelRequirement] = optimalKernel;
            GpuConvolution::instance().forwardMutex.unlock();
        }

        // printf("!!!!!! Switched to algo %d\n", algo);
        cudnnStatus = cudnnConvolutionForward(stream.getCudnnHandle(),
                                              &ALPHA_NO_SCALE,
                                              convolutionKernelRequirement.getDataInputTensorDescriptor(),
                                              dataInput.getMemPtr(),
                                              convolutionKernelRequirement.getWeightsFilterDescriptor(),
                                              weights.getMemPtr(),
                                              convolutionKernelRequirement.getConvolutionDescriptor(),
                                              optimalKernel.algo,
                                              nullptr,
                                              optimalKernel.memory,
                                              &BETA_CLEAR,
                                              convolutionKernelRequirement.getDataOutputTensorDescriptor(),
                                              dataOutput.getMemPtr());
    }

    if (cudnnStatus == 3) {
        // printf("!!!!!! Still failed, fall back used\n");
        // If it still doesn't work then go with algo CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM which is stable and does not use a workspace.

        GpuConvolution::instance().forwardMutex.lock();
        optimalKernel.algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        optimalKernel.memory = 0;
        optimalForwardKernels[convolutionKernelRequirement] = optimalKernel;
        GpuConvolution::instance().forwardMutex.unlock();

        cudnnStatus = cudnnConvolutionForward(stream.getCudnnHandle(),
                                              &ALPHA_NO_SCALE,
                                              convolutionKernelRequirement.getDataInputTensorDescriptor(),
                                              dataInput.getMemPtr(),
                                              convolutionKernelRequirement.getWeightsFilterDescriptor(),
                                              weights.getMemPtr(),
                                              convolutionKernelRequirement.getConvolutionDescriptor(),
                                              optimalKernel.algo,
                                              nullptr,
                                              optimalKernel.memory,
                                              &BETA_CLEAR,
                                              convolutionKernelRequirement.getDataOutputTensorDescriptor(),
                                              dataOutput.getMemPtr());
    }

    if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
        printf("cudnnStatus %d\n", cudnnStatus);
        // printf("algo %d, memory %ld, inPtr %p, wPtr %p wsPtr %p, outPtr %p\n", optimalKernel.algo,
        // optimalKernel.memory,
        //    dataInput.getMemPtr(), weights.getMemPtr(), workspace.isPresent() ? workspace.get().getMemPtr() : nullptr,
        //    dataOutput.getMemPtr());
        fflush(stdout);
    }
    assert(cudnnStatus == CUDNN_STATUS_SUCCESS);

    if (biases.isPresent()) {
        assert(biases.get().getPlacement() == weights.getPlacement());
        vector<unsigned long> biasDimensions = biases.get().getDescriptor().getDimensions();
        assert(biasDimensions.size() == 1);
        assert(biasDimensions[0] == dataOutput.getDescriptor().getDimensions()[1]);

        addConvolutionBias(dataOutput, biases, stream);
    }
}

void GpuConvolution::convolutionBackwardData(ConvolutionKernelRequirement convolutionKernelRequirement,
                                             Tensor errorInput,
                                             Tensor weights,
                                             Tensor errorOutput,
                                             Optional<Tensor> workspace,
                                             Stream stream) {
    assert(errorInput.getPlacement() == weights.getPlacement());
    assert(errorInput.getPlacement() == errorOutput.getPlacement());

    GpuConvolution::instance().backwardDataMutex.lock();
    assert(optimalBackwardDataKernels.count(convolutionKernelRequirement) == 1);
    cudnnConvolutionBwdDataAlgoPerf_t optimalKernel = optimalBackwardDataKernels[convolutionKernelRequirement];
    GpuConvolution::instance().backwardDataMutex.unlock();

    if (optimalKernel.memory > 0) {
        if (!workspace.isPresent()) {
            printf("algo %d workspaceBytes %ld\n", optimalKernel.algo, optimalKernel.memory);
        }
        assert(workspace.isPresent());
        assert(optimalKernel.memory == workspace.get().getDescriptor().getArraySizeInBytes());
    }

    ScopedGpu scopedGpu(stream.getGpuNum());

    cudnnStatus_t cudnnStatus;
    cudnnStatus = cudnnConvolutionBackwardData(stream.getCudnnHandle(),
                                               &ALPHA_NO_SCALE,
                                               convolutionKernelRequirement.getWeightsFilterDescriptor(),
                                               weights.getMemPtr(),
                                               convolutionKernelRequirement.getErrorInputTensorDescriptor(),
                                               errorInput.getMemPtr(),
                                               convolutionKernelRequirement.getConvolutionDescriptor(),
                                               optimalKernel.algo,
                                               workspace.isPresent() ? workspace.get().getMemPtr() : nullptr,
                                               optimalKernel.memory,
                                               &BETA_CLEAR,
                                               convolutionKernelRequirement.getErrorOutputTensorDescriptor(),
                                               errorOutput.getMemPtr());
    assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
}

void GpuConvolution::convolutionBackwardFilter(ConvolutionKernelRequirement convolutionKernelRequirement,
                                               Tensor dataInput,
                                               Tensor errorInput,
                                               Tensor weightsGradient,
                                               Optional<Tensor> workspace,
                                               Stream stream,
                                               bool accumulateGradient) {
    assert(dataInput.getPlacement() == errorInput.getPlacement());
    assert(dataInput.getPlacement() == weightsGradient.getPlacement());

    GpuConvolution::instance().backwardFilterMutex.lock();
    assert(optimalBackwardFilterKernels.count(convolutionKernelRequirement) == 1);
    cudnnConvolutionBwdFilterAlgoPerf_t optimalKernel = optimalBackwardFilterKernels[convolutionKernelRequirement];
    GpuConvolution::instance().backwardFilterMutex.unlock();

    if (optimalKernel.memory > 0) {
        assert(workspace.isPresent());
        assert(optimalKernel.memory == workspace.get().getDescriptor().getArraySizeInBytes());
    }

    ScopedGpu scopedGpu(stream.getGpuNum());

    cudnnStatus_t cudnnStatus;
    cudnnStatus = cudnnConvolutionBackwardFilter(stream.getCudnnHandle(),
                                                 &ALPHA_NO_SCALE,
                                                 convolutionKernelRequirement.getDataInputTensorDescriptor(),
                                                 dataInput.getMemPtr(),
                                                 convolutionKernelRequirement.getErrorInputTensorDescriptor(),
                                                 errorInput.getMemPtr(),
                                                 convolutionKernelRequirement.getConvolutionDescriptor(),
                                                 optimalKernel.algo,
                                                 workspace.isPresent() ? workspace.get().getMemPtr() : nullptr,
                                                 optimalKernel.memory,
                                                 accumulateGradient ? &BETA_ACCUMULATE : &BETA_CLEAR,
                                                 convolutionKernelRequirement.getWeightsGradientFilterDescriptor(),
                                                 weightsGradient.getMemPtr());
    assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
}

void GpuConvolution::convolutionBackwardBias(Tensor errorInput, Tensor biasesGradient, Tensor workspace, Stream stream) {
    computeConvolutionBiasesGradient(errorInput, biasesGradient, workspace, stream);
}
