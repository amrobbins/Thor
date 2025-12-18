#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "Utilities/TensorOperations/Misc/Extract.h"
#include "Utilities/TensorOperations/Misc/Pad.h"

namespace ThorImplementation {

/**
 * Zero pad's the tensor
 *
 */
class Pad : public Layer {
   public:
    virtual ~Pad() {}

    // The paddingAmount map key is the dimension number (starting from dimension 0 - the most significant dimension)
    // i.e. if there is a c++ array data[x][y][z] then paddingAmount[0] represents paddingAmount[x].
    // The corresponding value is a pair where the first integer represents the number of elements of padding at the
    // beginning of that dimension and the second integer represents the number of elements of padding at the end of
    // that dimension.
    Pad(std::map<unsigned int, std::pair<unsigned int, unsigned int>> paddingAmount) {
        assert(!paddingAmount.empty());
        this->paddingAmount = paddingAmount;
    }

    virtual Optional<Tensor> createFeatureOutputTensor() {
        assert(featureInput.isPresent());
        std::vector<unsigned long> paddedDimensions = getDimensionsAfterPadding(featureInput);
        return Tensor(featureInput.get().getPlacement(),
                      TensorDescriptor(featureInput.get().getDescriptor().getDataType(), paddedDimensions));
    }

    virtual void compile() {
        assert(featureInput.isPresent());
        assert(featureOutput.isPresent());

        TensorPlacement placement = featureInput.get().getPlacement();
        std::vector<unsigned long> strideArrayDimensions;

        std::vector<unsigned long> inputTensorDimensions = featureInput.get().getDescriptor().getDimensions();
        strideArrayDimensions.push_back(inputTensorDimensions.size());
        TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
        Tensor stridePerUnpaddedDimension(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::UINT64, strideArrayDimensions));
        stridePerUnpaddedDimension_d = Tensor(placement, TensorDescriptor(TensorDescriptor::DataType::UINT64, strideArrayDimensions));
        unsigned long *strideCpu = (unsigned long *)stridePerUnpaddedDimension.getMemPtr();
        strideCpu[strideArrayDimensions[0] - 1] = 1;
        for (int i = (int)strideArrayDimensions[0] - 2; i >= 0; --i)
            strideCpu[i] = inputTensorDimensions[i + 1] * strideCpu[i + 1];
        stridePerUnpaddedDimension_d.copyFromAsync(stridePerUnpaddedDimension, stream);

        std::vector<unsigned long> outputTensorDimensions = featureOutput.get().getDescriptor().getDimensions();
        Tensor stridePerPaddedDimension(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::UINT64, strideArrayDimensions));
        stridePerPaddedDimension_d = Tensor(placement, TensorDescriptor(TensorDescriptor::DataType::UINT64, strideArrayDimensions));
        strideCpu = (unsigned long *)stridePerPaddedDimension.getMemPtr();
        strideCpu[strideArrayDimensions[0] - 1] = 1;
        for (int i = (int)strideArrayDimensions[0] - 2; i >= 0; --i)
            strideCpu[i] = outputTensorDimensions[i + 1] * strideCpu[i + 1];
        stridePerPaddedDimension_d.copyFromAsync(stridePerPaddedDimension, stream);

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
                assert(outputDimensionSize - (inputDimensionSize + dimensionPadding.first) == dimensionPadding.second);
            }
        }
        padBefore_d.copyFromAsync(padBefore, stream);
        padAfter_d.copyFromAsync(padAfter, stream);
    }

    virtual void infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream) {
        assert(inputTensor.isPresent());
        assert(inputTensor.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        assert(outputTensor.isPresent());
        ScopedGpu scopedGpu(inputTensor.get().getPlacement().getDeviceNum());

        launchPad((half *)outputTensor.get().getMemPtr(),
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

        launchExtract((half *)errorOut.get().getMemPtr(),
                      (half *)errorIn.get().getMemPtr(),
                      errorOut.get().getDescriptor().getTotalNumElements(),
                      errorOut.get().getDescriptor().getDimensions().size(),
                      (unsigned long *)stridePerPaddedDimension_d.getMemPtr(),
                      (unsigned long *)stridePerUnpaddedDimension_d.getMemPtr(),
                      (unsigned int *)padBefore_d.getMemPtr(),
                      (unsigned int *)padAfter_d.getMemPtr(),
                      stream);
    }

    std::vector<unsigned long> getDimensionsAfterPadding(Tensor unpaddedTensor) {
        std::vector<unsigned long> dimensions = unpaddedTensor.getDescriptor().getDimensions();
        for (auto it = paddingAmount.begin(); it != paddingAmount.end(); ++it) {
            unsigned int dimension = it->first;
            assert(dimension < dimensions.size());
            std::pair<unsigned int, unsigned int> dimensionPadding = it->second;
            dimensions[dimension] += (dimensionPadding.first + dimensionPadding.second);
        }
        return dimensions;
    }

   private:
    std::map<unsigned int, std::pair<unsigned int, unsigned int>> paddingAmount;

    Tensor stridePerPaddedDimension_d;
    Tensor stridePerUnpaddedDimension_d;
    Tensor padBefore_d;
    Tensor padAfter_d;
};

}  // namespace ThorImplementation
