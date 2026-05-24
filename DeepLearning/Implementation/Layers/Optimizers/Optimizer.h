#pragma once

#include <optional>
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Implementation/Parameter/Parameterizable.h"

#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Optimizers/SparseRowGradient.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/StampedEquation.h"

// New Shape: an optimizer optimizes exactly 1 tensor.
// Dense optimizers own dense gradient storage. Sparse-row optimizers own reduced sparse gradient storage.

namespace ThorImplementation {

class Optimizer : public Parameterizable {
   public:
    using DataType = TensorDescriptor::DataType;

    Optimizer(uint64_t id) : id(id) {}

    virtual void compile(const Tensor &weights, Stream &gradientUpdateStream) = 0;
    virtual SparseRowGradient compileSparseRows(const Tensor &weights, uint64_t maxSparseRows, Stream &gradientUpdateStream) {
        (void)weights;
        (void)maxSparseRows;
        (void)gradientUpdateStream;
        throw std::runtime_error("Optimizer does not support sparse row gradients.");
    }
    virtual void compile() { THOR_UNREACHABLE(); /*FIXME DELETE*/ }
    bool isCompiled() const { return compiled; }

    [[nodiscard]] virtual bool supportsSparseRowGradients() const { return false; }

    // Note: It is the responsibility of the layer to ensure all dependencies are available at the start of gradient update stream,
    //       and that the data stream will be blocked until the end of the gradient update stream
    virtual void updateWeights(uint32_t batchSize) = 0;
    virtual void updateSparseRows(uint32_t batchSize) {
        (void)batchSize;
        throw std::runtime_error("Optimizer does not support sparse row gradients.");
    }

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
        if (updateEquationStamped == nullptr)
            return listParameters();
        return updateEquationStamped->outputNames();
    }

    Tensor getOptimizerParameterTensor(const std::string &parameterName) {
        if (!compiled)
            throw std::runtime_error("getOptimizerParameterTensor() called on an uncompiled optimizer. It must be compiled first.");
        if (updateEquationStamped == nullptr) {
            if (!hasParameter(parameterName))
                throw std::runtime_error("Request to get optimizer parameter " + parameterName +
                                         " but the optimizer has no parameter by that name."
                                         " Parameters: " +
                                         listOfStrings(getOptimizerParameterNames()));
            return getParameterStorage(parameterName);
        }
        std::unordered_map<std::string, Tensor> optimizerParameters = updateEquationStamped->getFinalOutputs();
        if (!optimizerParameters.contains(parameterName))
            throw std::runtime_error("Request to get optimizer parameter " + parameterName +
                                     " but the optimizer has no parameter by that name."
                                     " Parameters: " +
                                     listOfStrings(getOptimizerParameterNames()));
        return optimizerParameters[parameterName];
    }

    ~Optimizer() override = default;

    std::optional<Tensor> getWeightsGradient() { return weightsGradient; }
    std::optional<SparseRowGradient> getSparseRowGradient() { return sparseRowGradient; }

    virtual std::shared_ptr<Optimizer> clone() const = 0;

   protected:
    Tensor weights;
    Stream gradientUpdateStream;
    bool compiled = false;

    std::optional<Tensor> weightsGradient;
    std::optional<SparseRowGradient> sparseRowGradient;

    std::unique_ptr<StampedExecutionPlan> updateEquationStamped;

   private:
    const uint64_t id;
};

}  // namespace ThorImplementation
