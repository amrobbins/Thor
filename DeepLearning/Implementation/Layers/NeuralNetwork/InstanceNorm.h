#pragma once

#include "DeepLearning/Implementation/Layers/TrainableLayer.h"
#include "DeepLearning/Implementation/Parameter/PhysicalParameter.h"
#include "DeepLearning/Implementation/ThorError.h"

#include <optional>
#include <vector>

namespace ThorImplementation {

class InstanceNorm : public TrainableLayer {
   public:
    ~InstanceNorm() override;

    InstanceNorm(const TensorPlacement& placement,
                 bool inferenceOnly,
                 uint64_t channelCount,
                 std::optional<double> epsilon = std::nullopt,
                 std::optional<TensorDescriptor::DataType> parameterDataType = std::nullopt,
                 std::vector<std::shared_ptr<PhysicalParameter>> physicalParameters = {},
                 int64_t stampedId = -1);

    std::string getLayerType() override { return "InstanceNorm"; }

    uint64_t getChannelCount() const { return channelCount; }
    double getEpsilon() const { return epsilon; }
    void setEpsilon(double value);

    std::optional<Tensor> createFeatureOutputTensor() override;
    std::optional<Tensor> createErrorOutputTensor(bool backPropagateError, uint32_t connectionNumber) override;

    uint64_t flopCountForward() override;
    uint64_t flopCountBackward() override;

    void cleanup() override;

   protected:
    void compileImpl() override;

   private:
    void computeFeatureOut(uint32_t connectionNumber) override;
    std::optional<Event> computeErrorOutAccumulateWeightsGradienFused(uint32_t connectionNumber,
                                                                      bool clearWeightsGradientFirstIfFused) override;
    void accumulateWeightsGradient(uint32_t connectionNumber, bool clearGradientFirst) override;

    static uint64_t checkedChannelCount(uint64_t channelCount);
    void validateConfiguredInput(const Tensor& input) const;
    uint64_t computeSpatialElementCount(const Tensor& input) const;

    uint64_t channelCount = 0;
    uint64_t batchSize = 0;
    uint64_t spatialElementCount = 0;
    double epsilon = 1.0e-5;
    TensorDescriptor::DataType parameterDataType = TensorDescriptor::DataType::FP32;

    Tensor weights;
    Tensor biases;

    std::vector<Tensor> saveMean;
    std::vector<Tensor> saveInvVariance;
    std::vector<Tensor> scratchDScale;
    std::vector<Tensor> scratchDBias;
    std::optional<Tensor> scratchErrorOutput;
};

}  // namespace ThorImplementation
