#include "Utilities/TensorOperations/GpuConvolution/GpuConvolution.h"

using namespace ThorImplementation;
using namespace std;

const float GpuConvolution::ALPHA_NO_SCALE = 1.0f;
const float GpuConvolution::BETA_CLEAR = 0.0f;
const float GpuConvolution::BETA_ACCUMULATE = 1.0f;

// Finds the optimal forward kernel for the convolution operation given the parameters
void GpuConvolution::chooseOptimalKernelForward(ConvolutionKernelRequirement convolutionKernelRequirement, Stream stream) {
    std::lock_guard<std::mutex> lock(instance().measureMutex);

    // Will only evaluate kernel once per gpu type
    // First check in memory.
    // If not in memory, check in file-based map
    // If not in file based map, measure latency to determine optimal kernel
    if (GpuConvolution::instance().optimalForwardKernels.contains(convolutionKernelRequirement)) {
        return;
    }

    ScopedGpu scopedGpu(stream.getGpuNum());

    // Ensure there are no unreported runtime errors
    cudnnStatus_t cudnnStatus;
    stream.synchronize();
    cudnnQueryRuntimeError(stream.getCudnnHandle(), &cudnnStatus, CUDNN_ERRQUERY_BLOCKING, nullptr);

    // Evaluate best kernel
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

    // uint64_t maxWorkspaceSizeInBytes = dataInput.getDescriptor().getArraySizeInBytes() +
    // dataOutput.getDescriptor().getArraySizeInBytes();

    for (int i = 0; i < returnedAlgoCount; ++i) {
        if (perfResults[i].status != CUDNN_STATUS_SUCCESS)
            continue;
        uint64_t workspaceSizeInBytes = perfResults[i].memory;
        Optional<Tensor> workspace;
        // if (workspaceSizeInBytes > maxWorkspaceSizeInBytes)
        //    continue;
        if (workspaceSizeInBytes > 0)
            workspace = Tensor(gpuPlacement, TensorDescriptor(TensorDescriptor::DataType::UINT8, {workspaceSizeInBytes}));

        // Clear any possible runtime errors
        assert(cudnnQueryRuntimeError(stream.getCudnnHandle(), &cudnnStatus, CUDNN_ERRQUERY_BLOCKING, nullptr) == CUDNN_STATUS_SUCCESS);

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
            // Check for a runtime error
            assert(cudnnQueryRuntimeError(stream.getCudnnHandle(), &cudnnStatus, CUDNN_ERRQUERY_BLOCKING, nullptr) == CUDNN_STATUS_SUCCESS);
            if (cudnnStatus == CUDNN_STATUS_SUCCESS) {
                GpuConvolution::instance().optimalForwardKernels.put(convolutionKernelRequirement, perfResults[i]);
                return;
            }
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
    std::lock_guard<std::mutex> lock(instance().measureMutex);

    // Will only evaluate kernel once per gpu type
    // First check in memory.
    // If not in memory, check in file-based map
    // If not in file based map, measure latency to determine optimal kernel
    if (GpuConvolution::instance().optimalBackwardDataKernels.contains(convolutionKernelRequirement)) {
        return;
    }

    ScopedGpu scopedGpu(stream.getGpuNum());

    // Ensure there are no unreported runtime errors
    cudnnStatus_t cudnnStatus;
    stream.synchronize();
    cudnnQueryRuntimeError(stream.getCudnnHandle(), &cudnnStatus, CUDNN_ERRQUERY_BLOCKING, nullptr);

    // Evaluate best kernel
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

    // uint64_t maxWorkspaceSizeInBytes = errorOutput.getDescriptor().getArraySizeInBytes() +
    // errorInput.getDescriptor().getArraySizeInBytes();

    for (int i = 0; i < returnedAlgoCount; ++i) {
        if (perfResults[i].status != CUDNN_STATUS_SUCCESS)
            continue;
        uint64_t workspaceSizeInBytes = perfResults[i].memory;
        // if (workspaceSizeInBytes > maxWorkspaceSizeInBytes)
        //    continue;

        Optional<Tensor> workspace;
        if (workspaceSizeInBytes > 0)
            workspace = Tensor(gpuPlacement, TensorDescriptor(TensorDescriptor::DataType::UINT8, {workspaceSizeInBytes}));

        // Clear any possible runtime errors
        assert(cudnnQueryRuntimeError(stream.getCudnnHandle(), &cudnnStatus, CUDNN_ERRQUERY_BLOCKING, nullptr) == CUDNN_STATUS_SUCCESS);

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
            // Check for a runtime error
            assert(cudnnQueryRuntimeError(stream.getCudnnHandle(), &cudnnStatus, CUDNN_ERRQUERY_BLOCKING, nullptr) == CUDNN_STATUS_SUCCESS);
            if (cudnnStatus == CUDNN_STATUS_SUCCESS) {
                GpuConvolution::instance().optimalBackwardDataKernels.put(convolutionKernelRequirement, perfResults[i]);
                return;
            }
        }
    }

    assert(false);
}

void GpuConvolution::chooseOptimalKernelBackwardFilter(ConvolutionKernelRequirement convolutionKernelRequirement, Stream stream) {
    std::lock_guard<std::mutex> lock(instance().measureMutex);

    // Will only evaluate kernel once per gpu type
    // First check in memory.
    // If not in memory, check in file-based map
    // If not in file based map, measure latency to determine optimal kernel
    if (GpuConvolution::instance().optimalBackwardFilterKernels.contains(convolutionKernelRequirement)) {
        return;
    }

    ScopedGpu scopedGpu(stream.getGpuNum());

    // Ensure there are no unreported runtime errors
    cudnnStatus_t cudnnStatus;
    stream.synchronize();
    cudnnQueryRuntimeError(stream.getCudnnHandle(), &cudnnStatus, CUDNN_ERRQUERY_BLOCKING, nullptr);

    // Evaluate best kernel
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

    // uint64_t maxWorkspaceSizeInBytes = dataInput.getDescriptor().getArraySizeInBytes() +
    // errorInput.getDescriptor().getArraySizeInBytes();

    for (int i = 0; i < returnedAlgoCount; ++i) {
        if (perfResults[i].status != CUDNN_STATUS_SUCCESS)
            continue;

        // FIXME: I'm seeing that the nondeterministic algorithms often give very wrong results, not sure why, problem with atomics?
        // FIXME: For now I am not using them, I should check later to see if they start to work better.
        // if (perfResults[i].determinism == 0)
        //    continue;

        uint64_t workspaceSizeInBytes = perfResults[i].memory;
        // if (workspaceSizeInBytes > maxWorkspaceSizeInBytes)
        //    continue;
        Optional<Tensor> workspace;
        if (workspaceSizeInBytes > 0)
            workspace = Tensor(gpuPlacement, TensorDescriptor(TensorDescriptor::DataType::UINT8, {workspaceSizeInBytes}));

        // Clear any possible runtime errors
        assert(cudnnQueryRuntimeError(stream.getCudnnHandle(), &cudnnStatus, CUDNN_ERRQUERY_BLOCKING, nullptr) == CUDNN_STATUS_SUCCESS);

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
            // Check for a runtime error
            assert(cudnnQueryRuntimeError(stream.getCudnnHandle(), &cudnnStatus, CUDNN_ERRQUERY_BLOCKING, nullptr) == CUDNN_STATUS_SUCCESS);
            if (cudnnStatus == CUDNN_STATUS_SUCCESS) {
                GpuConvolution::instance().optimalBackwardFilterKernels.put(convolutionKernelRequirement, perfResults[i]);
                return;
            }
        }
    }

    assert(false);
}

uint64_t GpuConvolution::getForwardWorkspaceSizeInBytes(ConvolutionKernelRequirement convolutionKernelRequirement) {
    auto optimalKernel = GpuConvolution::instance().optimalForwardKernels.get(convolutionKernelRequirement);
    assert(optimalKernel.has_value());
    return optimalKernel->memory;
}

uint64_t GpuConvolution::getBackwardDataWorkspaceSizeInBytes(ConvolutionKernelRequirement convolutionKernelRequirement) {
    auto optimalKernel = GpuConvolution::instance().optimalBackwardDataKernels.get(convolutionKernelRequirement);
    assert(optimalKernel.has_value());
    return optimalKernel->memory;
}

uint64_t GpuConvolution::getBackwardFilterWorkspaceSizeInBytes(ConvolutionKernelRequirement convolutionKernelRequirement) {
    auto optimalKernel = GpuConvolution::instance().optimalBackwardFilterKernels.get(convolutionKernelRequirement);
    assert(optimalKernel.has_value());
    return optimalKernel->memory;
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

    auto maybeOptimalKernel = GpuConvolution::instance().optimalForwardKernels.get(convolutionKernelRequirement);
    assert(maybeOptimalKernel.has_value());
    cudnnConvolutionFwdAlgoPerf_t optimalKernel = *maybeOptimalKernel;

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

    auto maybeOptimalKernel = GpuConvolution::instance().optimalBackwardDataKernels.get(convolutionKernelRequirement);
    assert(maybeOptimalKernel.has_value());
    cudnnConvolutionBwdDataAlgoPerf_t optimalKernel = *maybeOptimalKernel;

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

    auto maybeOptimalKernel = GpuConvolution::instance().optimalBackwardFilterKernels.get(convolutionKernelRequirement);
    assert(maybeOptimalKernel.has_value());
    cudnnConvolutionBwdFilterAlgoPerf_t optimalKernel = *maybeOptimalKernel;

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

void GpuConvolution::printBackwardFilterKernelInfo(ConvolutionKernelRequirement convolutionKernelRequirement) {
    auto maybeAlgo = GpuConvolution::instance().optimalBackwardFilterKernels.get(convolutionKernelRequirement);
    assert(maybeAlgo.has_value());
    const auto& algo = *maybeAlgo;

    printf("algo %d status %d time %f workspaceSize %ld determinism %d mathType %d\n",
           algo.algo,
           algo.status,
           algo.time,
           algo.memory,
           algo.determinism,
           algo.mathType);
}
