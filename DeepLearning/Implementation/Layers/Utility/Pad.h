#pragma once

#include <optional>
#include "DeepLearning/Implementation/ThorError.h"

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
    ~Pad() override {}

    // The paddingAmount map key is the dimension number (starting from dimension 0 - the most significant dimension)
    // i.e. if there is a c++ array data[x][y][z] then paddingAmount[0] represents paddingAmount[x].
    // The corresponding value is a pair where the first integer represents the number of elements of padding at the
    // beginning of that dimension and the second integer represents the number of elements of padding at the end of
    // that dimension.
    Pad(std::map<unsigned int, std::pair<unsigned int, unsigned int>> paddingAmount) {
        THOR_THROW_IF_FALSE(!paddingAmount.empty());
        this->paddingAmount = paddingAmount;
    }

    std::optional<Tensor> createFeatureOutputTensor() override {
        THOR_THROW_IF_FALSE(featureInput.has_value());
        std::vector<unsigned long> paddedDimensions = getDimensionsAfterPadding(featureInput.value());
        return Tensor(featureInput.value().getPlacement(),
                      TensorDescriptor(featureInput.value().getDescriptor().getDataType(), paddedDimensions));
    }

    void compileImpl() override {
        Layer::compileImpl();
        THOR_THROW_IF_FALSE(featureInput.has_value());
        THOR_THROW_IF_FALSE(featureOutput.has_value());

        TensorPlacement placement = featureInput.value().getPlacement();
        std::vector<unsigned long> strideArrayDimensions;

        std::vector<unsigned long> inputTensorDimensions = featureInput.value().getDescriptor().getDimensions();
        strideArrayDimensions.push_back(inputTensorDimensions.size());
        TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
        Tensor stridePerUnpaddedDimension(cpuPlacement, TensorDescriptor(DataType::UINT64, strideArrayDimensions));
        stridePerUnpaddedDimension_d = Tensor(placement, TensorDescriptor(DataType::UINT64, strideArrayDimensions));
        unsigned long *strideCpu = (unsigned long *)stridePerUnpaddedDimension.getMemPtr();
        strideCpu[strideArrayDimensions[0] - 1] = 1;
        for (int i = (int)strideArrayDimensions[0] - 2; i >= 0; --i)
            strideCpu[i] = inputTensorDimensions[i + 1] * strideCpu[i + 1];
        stridePerUnpaddedDimension_d.copyFromAsync(stridePerUnpaddedDimension, stream);

        std::vector<unsigned long> outputTensorDimensions = featureOutput.value().getDescriptor().getDimensions();
        Tensor stridePerPaddedDimension(cpuPlacement, TensorDescriptor(DataType::UINT64, strideArrayDimensions));
        stridePerPaddedDimension_d = Tensor(placement, TensorDescriptor(DataType::UINT64, strideArrayDimensions));
        strideCpu = (unsigned long *)stridePerPaddedDimension.getMemPtr();
        strideCpu[strideArrayDimensions[0] - 1] = 1;
        for (int i = (int)strideArrayDimensions[0] - 2; i >= 0; --i)
            strideCpu[i] = outputTensorDimensions[i + 1] * strideCpu[i + 1];
        stridePerPaddedDimension_d.copyFromAsync(stridePerPaddedDimension, stream);

        Tensor padBefore(cpuPlacement, TensorDescriptor(DataType::UINT32, strideArrayDimensions));
        padBefore_d = Tensor(placement, TensorDescriptor(DataType::UINT32, strideArrayDimensions));
        Tensor padAfter(cpuPlacement, TensorDescriptor(DataType::UINT32, strideArrayDimensions));
        padAfter_d = Tensor(placement, TensorDescriptor(DataType::UINT32, strideArrayDimensions));
        unsigned int *padBeforeCpu = (unsigned int *)padBefore.getMemPtr();
        unsigned int *padAfterCpu = (unsigned int *)padAfter.getMemPtr();
        for (unsigned int i = 0; i < featureInput.value().getDescriptor().getDimensions().size(); ++i) {
            auto it = paddingAmount.find(i);
            unsigned long inputDimensionSize = featureInput.value().getDescriptor().getDimensions()[i];
            if (it == paddingAmount.end()) {
                padBeforeCpu[i] = 0;
                padAfterCpu[i] = inputDimensionSize - 1;
            } else {
                std::pair<unsigned int, unsigned int> dimensionPadding = it->second;
                padBeforeCpu[i] = dimensionPadding.first;
                padAfterCpu[i] = padBeforeCpu[i] + inputDimensionSize - 1;
                unsigned long outputDimensionSize = featureOutput.value().getDescriptor().getDimensions()[i];
                THOR_THROW_IF_FALSE(outputDimensionSize - (inputDimensionSize + dimensionPadding.first) == dimensionPadding.second);
            }
        }
        padBefore_d.copyFromAsync(padBefore, stream);
        padAfter_d.copyFromAsync(padAfter, stream);
    }

    void infer(std::optional<Tensor> inputTensor, std::optional<Tensor> outputTensor, Stream stream) override {
        THOR_THROW_IF_FALSE(inputTensor.has_value());
        THOR_THROW_IF_FALSE(inputTensor.value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        THOR_THROW_IF_FALSE(outputTensor.has_value());
        ScopedGpu scopedGpu(inputTensor.value().getPlacement().getDeviceNum());
        THOR_THROW_IF_FALSE(inputTensor.value().getDescriptor().getDataType() == outputTensor.value().getDescriptor().getDataType());

        launchPad(outputTensor.value().getMemPtr(),
                  inputTensor.value().getMemPtr(),
                  outputTensor.value().getDescriptor().getTotalNumElements(),
                  outputTensor.value().getDescriptor().getDimensions().size(),
                  (unsigned long *)stridePerPaddedDimension_d.getMemPtr(),
                  (unsigned long *)stridePerUnpaddedDimension_d.getMemPtr(),
                  (unsigned int *)padBefore_d.getMemPtr(),
                  (unsigned int *)padAfter_d.getMemPtr(),
                  inputTensor.value().getDescriptor().getDataType(),
                  stream);
    }

    void backProp(std::optional<Tensor> dataIn, std::optional<Tensor> errorIn, std::optional<Tensor> errorOut, Stream stream) override {
        if (!errorOut.has_value())
            return;

        THOR_THROW_IF_FALSE(errorIn.has_value());
        THOR_THROW_IF_FALSE(errorIn.value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        ScopedGpu scopedGpu(errorIn.value().getPlacement().getDeviceNum());
        THOR_THROW_IF_FALSE(errorIn.value().getDescriptor().getDataType() == errorOut.value().getDescriptor().getDataType());

        launchExtract(errorOut.value().getMemPtr(),
                      errorIn.value().getMemPtr(),
                      errorOut.value().getDescriptor().getTotalNumElements(),
                      errorOut.value().getDescriptor().getDimensions().size(),
                      (unsigned long *)stridePerPaddedDimension_d.getMemPtr(),
                      (unsigned long *)stridePerUnpaddedDimension_d.getMemPtr(),
                      (unsigned int *)padBefore_d.getMemPtr(),
                      (unsigned int *)padAfter_d.getMemPtr(),
                      errorOut.value().getDescriptor().getDataType(),
                      stream);
    }

    std::vector<unsigned long> getDimensionsAfterPadding(Tensor unpaddedTensor) {
        std::vector<unsigned long> dimensions = unpaddedTensor.getDescriptor().getDimensions();
        for (auto it = paddingAmount.begin(); it != paddingAmount.end(); ++it) {
            unsigned int dimension = it->first;
            THOR_THROW_IF_FALSE(dimension < dimensions.size());
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
