#pragma once

#include <optional>
#include "DeepLearning/Implementation/Layers/Layer.h"
#include "Utilities/TensorOperations/Cub/CubReduction.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ThorImplementation {

/**
 * Input loss has implementation dimensions [b][d0]...[dn].
 *
 * BATCH: sum every non-batch loss value, then average those per-example sums across b -> [1]
 * PER_OUTPUT: average across b while preserving every non-batch loss axis -> [d0]...[dn]
 * PER_EXAMPLE: sum every non-batch loss value independently for each example -> [b]
 */
class LossShaper : public Layer {
   public:
    enum class OutputLossType { BATCH, PER_EXAMPLE, PER_OUTPUT };

    LossShaper(OutputLossType outputLossType);
    ~LossShaper() override;

    std::optional<Tensor> createFeatureOutputTensor() override;
    void compileImpl() override;
    void infer(std::optional<Tensor> inputTensor, std::optional<Tensor> outputTensor, Stream stream) override;
    virtual void backward(std::optional<Tensor> errorInput);
    void backProp(std::optional<Tensor> dataIn, std::optional<Tensor> errorIn, std::optional<Tensor> errorOut, Stream stream) override;

    std::string getType() override;

    static std::vector<uint32_t> getReductionAxes(const std::vector<uint64_t>& inputDimensions,
                                                   OutputLossType outputLossType);
    static float getReductionOutputScale(const std::vector<uint64_t>& inputDimensions,
                                         OutputLossType outputLossType);
    static std::vector<uint64_t> getOutputDimensions(std::vector<uint64_t> inputDimensions, OutputLossType outputLossType);

   private:
    bool uninitialized;

    OutputLossType outputLossType;
    std::shared_ptr<StampedCubReduction> reduction;
};

inline void to_json(nlohmann::json &j, const LossShaper::OutputLossType &outputLossType) {
    switch (outputLossType) {
        case LossShaper::OutputLossType::BATCH:
            j = "batch";
            return;
        case LossShaper::OutputLossType::PER_OUTPUT:
            j = "per_output";
            return;
        case LossShaper::OutputLossType::PER_EXAMPLE:
            j = "per_example";
            return;
    }
    throw std::invalid_argument("Unsupported OutputLossType enum value.");
}

inline void from_json(const nlohmann::json &j, LossShaper::OutputLossType &outputLossType) {
    const std::string serializedOutputLossType = j.get<std::string>();
    if (serializedOutputLossType == "batch") {
        outputLossType = LossShaper::OutputLossType::BATCH;
    } else if (serializedOutputLossType == "per_output") {
        outputLossType = LossShaper::OutputLossType::PER_OUTPUT;
    } else if (serializedOutputLossType == "per_example") {
        outputLossType = LossShaper::OutputLossType::PER_EXAMPLE;
    } else {
        throw std::invalid_argument("Unsupported output loss type '" + serializedOutputLossType +
                                    "'. Expected batch, per_example, or per_output.");
    }
}

}  // namespace ThorImplementation
