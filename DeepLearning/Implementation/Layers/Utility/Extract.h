#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "Utilities/TensorOperations/Misc/Extract.h"
#include "Utilities/TensorOperations/Misc/Pad.h"

namespace ThorImplementation {

/**
 * Extracts a chunk of a tensor defined by a span in each dimension.
 *
 */
class Extract : public Layer {
   public:
    virtual ~Extract() {}

    // The dimensionSpans map key is the dimension number (starting from dimension 0 - the most significant dimension)
    // i.e. if there is a c++ array data[x][y][z] then dimensionSpans[0] represents dimensionSpans[x].
    // The span for a dimension is represented as pair(indexOfFirstElementInSpan, indexOfLastElementInSpan)
    Extract(std::vector<std::pair<unsigned int, unsigned int>> dimensionSpans) {
        assert(!dimensionSpans.empty());
        this->dimensionSpans = dimensionSpans;
    }

    virtual Optional<Tensor> createFeatureOutputTensor() {
        assert(featureInput.isPresent());
        std::vector<unsigned long> outputTensorDimensions = getExtractedTensorDimensions(dimensionSpans);
        assert(featureInput.get().getDescriptor().getDimensions().size() == outputTensorDimensions.size());
        return Tensor(featureInput.get().getPlacement(),
                      TensorDescriptor(featureInput.get().getDescriptor().getDataType(), outputTensorDimensions));
    }

    virtual void compileImpl() {
        Layer::compileImpl();
        assert(featureInput.isPresent());
        assert(featureOutput.isPresent());

        TensorPlacement placement = featureInput.get().getPlacement();
        std::vector<unsigned long> strideArrayDimensions;

        std::vector<unsigned long> inputTensorDimensions = featureInput.get().getDescriptor().getDimensions();
        strideArrayDimensions.push_back(inputTensorDimensions.size());
        TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
        Tensor stridePerPaddedDimension(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::UINT64, strideArrayDimensions));
        stridePerPaddedDimension_d = Tensor(placement, TensorDescriptor(TensorDescriptor::DataType::UINT64, strideArrayDimensions));
        unsigned long *strideCpu = (unsigned long *)stridePerPaddedDimension.getMemPtr();
        strideCpu[strideArrayDimensions[0] - 1] = 1;
        for (int i = (int)strideArrayDimensions[0] - 2; i >= 0; --i)
            strideCpu[i] = inputTensorDimensions[i + 1] * strideCpu[i + 1];
        stridePerPaddedDimension_d.copyFromAsync(stridePerPaddedDimension, stream);

        std::vector<unsigned long> outputTensorDimensions = featureOutput.get().getDescriptor().getDimensions();
        Tensor stridePerUnpaddedDimension(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::UINT64, strideArrayDimensions));
        stridePerUnpaddedDimension_d = Tensor(placement, TensorDescriptor(TensorDescriptor::DataType::UINT64, strideArrayDimensions));
        strideCpu = (unsigned long *)stridePerUnpaddedDimension.getMemPtr();
        strideCpu[strideArrayDimensions[0] - 1] = 1;
        for (int i = (int)strideArrayDimensions[0] - 2; i >= 0; --i)
            strideCpu[i] = outputTensorDimensions[i + 1] * strideCpu[i + 1];
        stridePerUnpaddedDimension_d.copyFromAsync(stridePerUnpaddedDimension, stream);

        for (unsigned int d = 0; d < dimensionSpans.size(); ++d)
            paddingAmount[d] =
                std::pair<unsigned int, unsigned int>(dimensionSpans[d].first, (inputTensorDimensions[d] - 1) - dimensionSpans[d].second);

        Tensor padBefore(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::UINT32, strideArrayDimensions));
        padBefore_d = Tensor(placement, TensorDescriptor(TensorDescriptor::DataType::UINT32, strideArrayDimensions));
        Tensor padAfter(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::UINT32, strideArrayDimensions));
        padAfter_d = Tensor(placement, TensorDescriptor(TensorDescriptor::DataType::UINT32, strideArrayDimensions));
        unsigned int *padBeforeCpu = (unsigned int *)padBefore.getMemPtr();
        unsigned int *padAfterCpu = (unsigned int *)padAfter.getMemPtr();
        for (unsigned int i = 0; i < featureInput.get().getDescriptor().getDimensions().size(); ++i) {
            auto it = paddingAmount.find(i);
            unsigned long inputDimensionSize = featureInput.get().getDescriptor().getDimensions()[i];
            if (it == paddingAmount.end()) {
                padBeforeCpu[i] = 0;
                padAfterCpu[i] = inputDimensionSize - 1;
            } else {
                std::pair<unsigned int, unsigned int> dimensionPadding = it->second;
                padBeforeCpu[i] = dimensionPadding.first;
                padAfterCpu[i] = padBeforeCpu[i] + inputDimensionSize - 1;
                unsigned long outputDimensionSize = featureOutput.get().getDescriptor().getDimensions()[i];
                assert(inputDimensionSize - (outputDimensionSize + dimensionPadding.first) == dimensionPadding.second);
            }
        }
        padBefore_d.copyFromAsync(padBefore, stream);
        padAfter_d.copyFromAsync(padAfter, stream);
    }

    virtual void infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream) {
        assert(inputTensor.isPresent());
        assert(outputTensor.isPresent());
        assert(inputTensor.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        ScopedGpu scopedGpu(inputTensor.get().getPlacement().getDeviceNum());

        launchExtract((half *)outputTensor.get().getMemPtr(),
                      (half *)inputTensor.get().getMemPtr(),
                      outputTensor.get().getDescriptor().getTotalNumElements(),
                      outputTensor.get().getDescriptor().getDimensions().size(),
                      (unsigned long *)stridePerPaddedDimension_d.getMemPtr(),
                      (unsigned long *)stridePerUnpaddedDimension_d.getMemPtr(),
                      (unsigned int *)padBefore_d.getMemPtr(),
                      (unsigned int *)padAfter_d.getMemPtr(),
                      stream);
    }

    virtual void backProp(Optional<Tensor> dataIn, Optional<Tensor> errorIn, Optional<Tensor> errorOut, Stream stream) {
        if (errorOut.isEmpty())
            return;

        assert(errorIn.isPresent());
        assert(errorIn.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        ScopedGpu scopedGpu(errorIn.get().getPlacement().getDeviceNum());

        launchPad((half *)errorOut.get().getMemPtr(),
                  (half *)errorIn.get().getMemPtr(),
                  errorOut.get().getDescriptor().getTotalNumElements(),
                  errorOut.get().getDescriptor().getDimensions().size(),
                  (unsigned long *)stridePerPaddedDimension_d.getMemPtr(),
                  (unsigned long *)stridePerUnpaddedDimension_d.getMemPtr(),
                  (unsigned int *)padBefore_d.getMemPtr(),
                  (unsigned int *)padAfter_d.getMemPtr(),
                  stream);
    }

    std::vector<unsigned long> getExtractedTensorDimensions(std::vector<std::pair<unsigned int, unsigned int>> dimensionSpans) {
        std::vector<unsigned long> dimensions;
        for (unsigned int d = 0; d < dimensionSpans.size(); ++d)
            dimensions.push_back((dimensionSpans[d].second - dimensionSpans[d].first) + 1);
        return dimensions;
    }

   private:
    std::vector<std::pair<unsigned int, unsigned int>> dimensionSpans;
    std::map<unsigned int, std::pair<unsigned int, unsigned int>> paddingAmount;

    Tensor stridePerPaddedDimension_d;
    Tensor stridePerUnpaddedDimension_d;
    Tensor padBefore_d;
    Tensor padAfter_d;
};

}  // namespace ThorImplementation
