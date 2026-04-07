#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/CudnnHelper.h"
#include "Utilities/TensorMathFusion/DynamicExpression.h"
#include "Utilities/TensorMathFusion/StampedEquation.h"

// New Shape: an optimizer optimizes exactly 1 tensor.
//            The layer owns the weights gradient memory and materializes it as needed. Layer manages lifetime of gradient memory.

namespace ThorImplementation {
class TrainableWeightsBiasesLayer;

class Optimizer {
   public:
    using DataType = TensorDescriptor::DataType;

    // Optimizer(uint64_t id, const Tensor &parameters, DynamicExpression optimizerExpression)
    //     : parameters(parameters),
    //       gradientUpdateStream(Stream::getNextGradientUpdateStream(parameters.getPlacement().getDeviceNum())),
    //       id(id) {
    //     assert(parameters.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    // }

    // Does physical optimizer take weights and grads in constructor or compile? -> during compile.

    Optimizer(uint64_t id) : id(id) {}

    virtual void compile(const Tensor &weights, Stream &gradientUpdateStream) = 0;
    virtual void compile() { assert(false); /*FIXME DELETE*/ }
    bool isCompiled() const { return compiled; }

    // Note: It is the responsibility of the layer to ensure all dependencies are available at the start of gradient update stream,
    //       and that the data stream will be blocked until the end of the gradient update stream
    virtual void updateWeights(uint32_t batchSize) = 0;

    virtual std::unordered_map<std::string, float> updateHyperParameters(uint64_t epoch, uint64_t batch, uint64_t batchesPerEpoch) = 0;
    virtual std::unordered_map<std::string, float> getAllHyperParameters() = 0;

    virtual Stream getGradientUpdateStream() const { return gradientUpdateStream; }

    uint64_t getId() const { return id; }
    bool operator==(const Optimizer &other) const { return id == other.id; }

    static std::string listOfStrings(const std::vector<std::string> &items) {
        std::string listAsString;

        for (size_t i = 0; i < items.size(); ++i) {
            listAsString += items[i];
            if (i + 1 < items.size()) {
                listAsString += ", ";
            }
        }

        if (listAsString.empty())
            listAsString = "<none>";

        return listAsString;
    }

    std::vector<std::string> getOptimizerParameterNames() {
        if (!compiled)
            throw std::runtime_error("getOptimizerParameterNames() called on an uncompiled optimizer. It must be compiled first.");
        assert(updateEquationStamped != nullptr);
        return updateEquationStamped->outputNames();
    }

    Tensor getOptimizerParameterTensor(const std::string &parameterName) {
        if (!compiled)
            throw std::runtime_error("getOptimizerParameterTensor() called on an uncompiled optimizer. It must be compiled first.");
        assert(updateEquationStamped != nullptr);
        std::unordered_map<std::string, Tensor> optimizerParameters = updateEquationStamped->getFinalOutputs();
        if (!optimizerParameters.contains(parameterName))
            throw std::runtime_error("Request to get optimizer parameter " + parameterName +
                                     " but the optimizer has no parameter by that name."
                                     " Parameters: " +
                                     listOfStrings(getOptimizerParameterNames()));
        return optimizerParameters[parameterName];
    }

    virtual ~Optimizer() = default;

    Optional<Tensor> getWeightsGradient() { return weightsGradient; }

   protected:
    Tensor weights;
    Stream gradientUpdateStream;
    bool compiled = false;

    Optional<Tensor> weightsGradient;

    std::unique_ptr<StampedExecutionPlan> updateEquationStamped;
    virtual DynamicExpression buildExpression() = 0;

   private:
    const uint64_t id;
};

}  // namespace ThorImplementation
