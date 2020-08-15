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

    // Returned algos don't always run, choose the first one that runs.
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, stream.getGpuNum());

    vector<unsigned long> inputDimensions;
    inputDimensions.push_back(convolutionKernelRequirement.getBatchSize());
    inputDimensions.push_back(convolutionKernelRequirement.getNumInputChannels());
    inputDimensions.push_back(convolutionKernelRequirement.getNumInputRows());
    inputDimensions.push_back(convolutionKernelRequirement.getNumInputColumns());
    TensorDescriptor inputDescriptor(TensorDescriptor::DataType::FP16, inputDimensions);
    Tensor dataInput(gpuPlacement, inputDescriptor);

    vector<unsigned long> outputDimensions;
    outputDimensions.push_back(convolutionKernelRequirement.getBatchSize());
    outputDimensions.push_back(convolutionKernelRequirement.getNumOutputChannels());
    outputDimensions.push_back(convolutionKernelRequirement.getNumOutputRows());
    outputDimensions.push_back(convolutionKernelRequirement.getNumOutputColumns());
    TensorDescriptor outputDescriptor(TensorDescriptor::DataType::FP16, outputDimensions);
    Tensor dataOutput(gpuPlacement, outputDescriptor);

    vector<unsigned long> weightsDimensions;
    weightsDimensions.push_back(convolutionKernelRequirement.getNumOutputChannels());
    weightsDimensions.push_back(convolutionKernelRequirement.getNumInputChannels());
    weightsDimensions.push_back(convolutionKernelRequirement.getFilterHeight());
    weightsDimensions.push_back(convolutionKernelRequirement.getFilterWidth());
    TensorDescriptor weightsDescriptor(TensorDescriptor::DataType::FP16, weightsDimensions);
    Tensor weights(gpuPlacement, weightsDescriptor);

    for (int i = 0; i < returnedAlgoCount; ++i) {
        if (perfResults[i].status != CUDNN_STATUS_SUCCESS)
            continue;
        uint64_t workspaceSizeInBytes = perfResults[i].memory;
        Optional<Tensor> workspace;
        if (workspaceSizeInBytes > 0)
            workspace = Tensor(gpuPlacement, TensorDescriptor(TensorDescriptor::DataType::UINT8, workspaceSizeInBytes));

        cudnnStatus_t cudnnStatus;
        cudnnStatus = cudnnConvolutionForward(stream.getCudnnHandle(),
                                              &ALPHA_NO_SCALE,
                                              convolutionKernelRequirement.getDataInputTensorDescriptor(),
                                              dataInput.getMemPtr(),
                                              convolutionKernelRequirement.getWeightsFilterDescriptor(),
                                              weights.getMemPtr(),
                                              convolutionKernelRequirement.getConvolutionDescriptor(),
                                              perfResults[i].algo,
                                              workspace.isPresent() ? workspace.get().getMemPtr() : nullptr,
                                              perfResults[i].memory,
                                              &BETA_ACCUMULATE,
                                              convolutionKernelRequirement.getDataOutputTensorDescriptor(),
                                              dataOutput.getMemPtr());
        if (cudnnStatus == CUDNN_STATUS_SUCCESS) {
            GpuConvolution::instance().forwardMutex.lock();
            optimalForwardKernels[convolutionKernelRequirement] = perfResults[i];
            GpuConvolution::instance().forwardMutex.unlock();
            return;
        }
    }

    assert(false);
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

    // Returned algos don't always run, choose the first one that runs.
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, stream.getGpuNum());

    vector<unsigned long> inputDimensions;
    inputDimensions.push_back(convolutionKernelRequirement.getBatchSize());
    inputDimensions.push_back(convolutionKernelRequirement.getNumInputChannels());
    inputDimensions.push_back(convolutionKernelRequirement.getNumInputRows());
    inputDimensions.push_back(convolutionKernelRequirement.getNumInputColumns());
    TensorDescriptor inputDescriptor(TensorDescriptor::DataType::FP16, inputDimensions);
    Tensor errorOutput(gpuPlacement, inputDescriptor);

    vector<unsigned long> outputDimensions;
    outputDimensions.push_back(convolutionKernelRequirement.getBatchSize());
    outputDimensions.push_back(convolutionKernelRequirement.getNumOutputChannels());
    outputDimensions.push_back(convolutionKernelRequirement.getNumOutputRows());
    outputDimensions.push_back(convolutionKernelRequirement.getNumOutputColumns());
    TensorDescriptor outputDescriptor(TensorDescriptor::DataType::FP16, outputDimensions);
    Tensor errorInput(gpuPlacement, outputDescriptor);

    vector<unsigned long> weightsDimensions;
    weightsDimensions.push_back(convolutionKernelRequirement.getNumOutputChannels());
    weightsDimensions.push_back(convolutionKernelRequirement.getNumInputChannels());
    weightsDimensions.push_back(convolutionKernelRequirement.getFilterHeight());
    weightsDimensions.push_back(convolutionKernelRequirement.getFilterWidth());
    TensorDescriptor weightsDescriptor(TensorDescriptor::DataType::FP16, weightsDimensions);
    Tensor weights(gpuPlacement, weightsDescriptor);

    for (int i = 0; i < returnedAlgoCount; ++i) {
        if (perfResults[i].status != CUDNN_STATUS_SUCCESS)
            continue;
        uint64_t workspaceSizeInBytes = perfResults[i].memory;
        Optional<Tensor> workspace;
        if (workspaceSizeInBytes > 0)
            workspace = Tensor(gpuPlacement, TensorDescriptor(TensorDescriptor::DataType::UINT8, workspaceSizeInBytes));

        cudnnStatus_t cudnnStatus;
        cudnnStatus = cudnnConvolutionBackwardData(stream.getCudnnHandle(),
                                                   &ALPHA_NO_SCALE,
                                                   convolutionKernelRequirement.getWeightsFilterDescriptor(),
                                                   weights.getMemPtr(),
                                                   convolutionKernelRequirement.getErrorInputTensorDescriptor(),
                                                   errorInput.getMemPtr(),
                                                   convolutionKernelRequirement.getConvolutionDescriptor(),
                                                   perfResults[i].algo,
                                                   workspace.isPresent() ? workspace.get().getMemPtr() : nullptr,
                                                   perfResults[i].memory,
                                                   &BETA_ACCUMULATE,
                                                   convolutionKernelRequirement.getErrorOutputTensorDescriptor(),
                                                   errorOutput.getMemPtr());
        if (cudnnStatus == CUDNN_STATUS_SUCCESS) {
            GpuConvolution::instance().backwardDataMutex.lock();
            optimalBackwardDataKernels[convolutionKernelRequirement] = perfResults[i];
            GpuConvolution::instance().backwardDataMutex.unlock();
            return;
        }
    }

    assert(false);
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

    // Returned algos don't always run, choose the first one that runs.
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, stream.getGpuNum());

    vector<unsigned long> inputDimensions;
    inputDimensions.push_back(convolutionKernelRequirement.getBatchSize());
    inputDimensions.push_back(convolutionKernelRequirement.getNumInputChannels());
    inputDimensions.push_back(convolutionKernelRequirement.getNumInputRows());
    inputDimensions.push_back(convolutionKernelRequirement.getNumInputColumns());
    TensorDescriptor inputDescriptor(TensorDescriptor::DataType::FP16, inputDimensions);
    Tensor dataInput(gpuPlacement, inputDescriptor);

    vector<unsigned long> outputDimensions;
    outputDimensions.push_back(convolutionKernelRequirement.getBatchSize());
    outputDimensions.push_back(convolutionKernelRequirement.getNumOutputChannels());
    outputDimensions.push_back(convolutionKernelRequirement.getNumOutputRows());
    outputDimensions.push_back(convolutionKernelRequirement.getNumOutputColumns());
    TensorDescriptor outputDescriptor(TensorDescriptor::DataType::FP16, outputDimensions);
    Tensor errorInput(gpuPlacement, outputDescriptor);

    vector<unsigned long> weightsDimensions;
    weightsDimensions.push_back(convolutionKernelRequirement.getNumOutputChannels());
    weightsDimensions.push_back(convolutionKernelRequirement.getNumInputChannels());
    weightsDimensions.push_back(convolutionKernelRequirement.getFilterHeight());
    weightsDimensions.push_back(convolutionKernelRequirement.getFilterWidth());
    TensorDescriptor weightsDescriptor(TensorDescriptor::DataType::FP16, weightsDimensions);
    Tensor weightsGradient(gpuPlacement, weightsDescriptor);

    for (int i = 0; i < returnedAlgoCount; ++i) {
        if (perfResults[i].status != CUDNN_STATUS_SUCCESS)
            continue;
        uint64_t workspaceSizeInBytes = perfResults[i].memory;
        Optional<Tensor> workspace;
        if (workspaceSizeInBytes > 0)
            workspace = Tensor(gpuPlacement, TensorDescriptor(TensorDescriptor::DataType::UINT8, workspaceSizeInBytes));

        cudnnStatus_t cudnnStatus;
        cudnnStatus = cudnnConvolutionBackwardFilter(stream.getCudnnHandle(),
                                                     &ALPHA_NO_SCALE,
                                                     convolutionKernelRequirement.getDataInputTensorDescriptor(),
                                                     dataInput.getMemPtr(),
                                                     convolutionKernelRequirement.getErrorInputTensorDescriptor(),
                                                     errorInput.getMemPtr(),
                                                     convolutionKernelRequirement.getConvolutionDescriptor(),
                                                     perfResults[i].algo,
                                                     workspace.isPresent() ? workspace.get().getMemPtr() : nullptr,
                                                     perfResults[i].memory,
                                                     &BETA_ACCUMULATE,
                                                     convolutionKernelRequirement.getWeightsGradientFilterDescriptor(),
                                                     weightsGradient.getMemPtr());
        if (cudnnStatus == CUDNN_STATUS_SUCCESS) {
            GpuConvolution::instance().backwardFilterMutex.lock();
            optimalBackwardFilterKernels[convolutionKernelRequirement] = perfResults[i];
            GpuConvolution::instance().backwardFilterMutex.unlock();
            return;
        }
    }

    assert(false);
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
    if (useCudnnBackwardBias)
        return 0;
    else
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

    if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
        printf("cudnnStatus %d\n", cudnnStatus);
        fflush(stdout);
    }
    assert(cudnnStatus == CUDNN_STATUS_SUCCESS);

    if (biases.isPresent()) {
        assert(biases.get().getPlacement() == weights.getPlacement());
        vector<unsigned long> biasDimensions = biases.get().getDescriptor().getDimensions();
        assert(biasDimensions.size() == 1);
        assert(biasDimensions[0] == dataOutput.getDescriptor().getDimensions()[1]);

        if (useCudnnForwardBias) {
            cudnnStatus = cudnnAddTensor(stream.getCudnnHandle(),
                                         &ALPHA_NO_SCALE,
                                         convolutionKernelRequirement.getBiasesTensorDescriptor(),
                                         biases.get().getMemPtr(),
                                         &BETA_ACCUMULATE,
                                         convolutionKernelRequirement.getDataOutputTensorDescriptor(),
                                         dataOutput.getMemPtr());
            assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
        } else {
            addConvolutionBias(dataOutput, biases, stream);
        }
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
    if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
        printf("cudnnStatus %d\n", cudnnStatus);
        fflush(stdout);
    }
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
    if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
        printf("cudnnStatus %d\n", cudnnStatus);
        fflush(stdout);
    }
    assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
}

void GpuConvolution::convolutionBackwardBias(ConvolutionKernelRequirement convolutionKernelRequirement,
                                             Tensor errorInput,
                                             Tensor biasesGradient,
                                             Optional<Tensor> workspace,
                                             Stream stream,
                                             bool accumulateGradient) {
    if (useCudnnBackwardBias) {
        cudnnStatus_t cudnnStatus;
        cudnnStatus = cudnnConvolutionBackwardBias(stream.getCudnnHandle(),
                                                   &ALPHA_NO_SCALE,
                                                   convolutionKernelRequirement.getErrorInputTensorDescriptor(),
                                                   errorInput.getMemPtr(),
                                                   accumulateGradient ? &BETA_ACCUMULATE : &BETA_CLEAR,
                                                   convolutionKernelRequirement.getBiasesTensorDescriptor(),
                                                   biasesGradient.getMemPtr());
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
    } else {
        computeConvolutionBiasesGradient(errorInput, biasesGradient, workspace, stream);
    }
}
