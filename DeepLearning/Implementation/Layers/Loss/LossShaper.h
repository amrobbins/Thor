#pragma once

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
 */
class LossShaper : public Layer {
   public:
    enum class OutputLossType { BATCH = 1107, CLASSWISE, ELEMENTWISE };

    LossShaper(OutputLossType outputLossType);
    virtual ~LossShaper();

    virtual Optional<Tensor> createFeatureOutputTensor();
    virtual void compile();
    virtual void infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream);
    virtual void backward(Optional<Tensor> errorInput);
    virtual void backProp(Optional<Tensor> dataIn, Optional<Tensor> errorIn, Optional<Tensor> errorOut, Stream stream);

    virtual std::string getType();

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
