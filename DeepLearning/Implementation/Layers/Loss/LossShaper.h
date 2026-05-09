#pragma once

#include <optional>
#include "DeepLearning/Implementation/Layers/Layer.h"
#include "Utilities/TensorOperations/Misc/BatchReduce.h"

#include <nlohmann/json.hpp>

#include <memory>

namespace ThorImplementation {

/**
 * b: batch dimension
 * c: class dimension (categorical) or output number dimension (numerical)
 *
 * Input loss is always raw loss with dimensions [b][c]
 *
 * Batch [b][c] -> [1]
 * Classwise [b][c] -> [c]
 * Elementwise [b][c] -> [b]
 * Raw [b][c] -> [b][c]
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
    std::shared_ptr<BatchReduce> batchReduce;
};

NLOHMANN_JSON_SERIALIZE_ENUM(LossShaper::OutputLossType,
                             {
                                 {LossShaper::OutputLossType::BATCH, "batch"},
                                 {LossShaper::OutputLossType::ELEMENTWISE, "elementwise"},
                                 {LossShaper::OutputLossType::CLASSWISE, "classwise"},
                             })

}  // namespace ThorImplementation
