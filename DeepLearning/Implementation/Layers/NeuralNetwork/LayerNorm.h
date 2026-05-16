#pragma once

#include "DeepLearning/Implementation/Layers/TrainableLayer.h"
#include "DeepLearning/Implementation/Parameter/PhysicalParameter.h"
#include "DeepLearning/Implementation/ThorError.h"

#include <optional>
#include <vector>

namespace ThorImplementation {

class LayerNorm : public TrainableLayer {
   public:
    ~LayerNorm() override;

    LayerNorm(const TensorPlacement& placement,
              bool inferenceOnly,
              std::vector<uint64_t> normalizedShape,
              std::optional<double> epsilon = std::nullopt,
              std::optional<TensorDescriptor::DataType> parameterDataType = std::nullopt,
              std::vector<std::shared_ptr<PhysicalParameter>> physicalParameters = {},
              int64_t stampedId = -1);

    std::string getLayerType() override { return "LayerNorm"; }

    const std::vector<uint64_t>& getNormalizedShape() const { return normalizedShape; }
    uint64_t getNormalizedFeatureCount() const { return normalizedFeatureCount; }
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

    static uint64_t checkedNormalizedFeatureCount(const std::vector<uint64_t>& normalizedShape);
    void validateConfiguredInput(const Tensor& input) const;
    uint64_t computeOuterSize(const Tensor& input) const;

    std::vector<uint64_t> normalizedShape;
    uint64_t normalizedFeatureCount = 0;
    uint64_t outerSize = 0;
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
