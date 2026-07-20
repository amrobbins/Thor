#pragma once

#include <optional>
#include "DeepLearning/Implementation/Layers/Layer.h"
#include "Utilities/TensorOperations/Cub/CubReduction.h"

#include <nlohmann/json.hpp>

#include <memory>

namespace ThorImplementation {

/**
 * b: batch dimension
 * c: the row-major flattening of every non-batch loss dimension
 *
 * Input loss may have dimensions [b][d0]...[dn]. Reduction treats it as a zero-copy [b][c] view.
 *
 * Batch: sum each item's c losses, then average those sums across b -> [1]
 * Classwise: average each flattened loss position across b -> [c]
 * Elementwise: sum each item's c losses -> [b]
 */
class LossShaper : public Layer {
   public:
    enum class OutputLossType { BATCH = 1107, CLASSWISE, ELEMENTWISE };

    LossShaper(OutputLossType outputLossType);
    ~LossShaper() override;

    std::optional<Tensor> createFeatureOutputTensor() override;
    void compileImpl() override;
    void infer(std::optional<Tensor> inputTensor, std::optional<Tensor> outputTensor, Stream stream) override;
    virtual void backward(std::optional<Tensor> errorInput);
    void backProp(std::optional<Tensor> dataIn, std::optional<Tensor> errorIn, std::optional<Tensor> errorOut, Stream stream) override;

    std::string getType() override;

    static std::vector<uint64_t> getOutputDimensions(std::vector<uint64_t> inputDimensions, OutputLossType outputLossType);

   private:
    bool uninitialized;

    OutputLossType outputLossType;
    std::shared_ptr<StampedCubReduction> reduction;
};

NLOHMANN_JSON_SERIALIZE_ENUM(LossShaper::OutputLossType,
                             {
                                 {LossShaper::OutputLossType::BATCH, "batch"},
                                 {LossShaper::OutputLossType::ELEMENTWISE, "elementwise"},
                                 {LossShaper::OutputLossType::CLASSWISE, "classwise"},
                             })

}  // namespace ThorImplementation
