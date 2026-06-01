#include "DeepLearning/Implementation/Layers/Optimizers/CustomOptimizer.h"

#include "DeepLearning/Implementation/Initializers/ZerosInitializer.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Parameter/PhysicalParameter.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Expression/FusedEquation.h"

#include <algorithm>
#include <set>
#include <stdexcept>

using namespace std;

namespace ThorImplementation {

CustomOptimizerUpdateContext::CustomOptimizerUpdateContext(const Tensor& weights,
                                                           Expression gradient,
                                                           std::string namePrefix,
                                                           Mode mode)
    : weights_(weights), gradient_(std::move(gradient)), namePrefix_(std::move(namePrefix)), mode_(mode) {}

Expression CustomOptimizerUpdateContext::weights(DataType outputDType, DataType computeDType) const {
    return Expression::input(weightsInputName(), outputDType, computeDType);
}

Expression CustomOptimizerUpdateContext::state(const std::string& name, DataType outputDType, DataType computeDType) const {
    if (name.empty()) {
        throw std::invalid_argument("CustomOptimizer state input name cannot be empty.");
    }
    return Expression::input(stateInputName(name), outputDType, computeDType);
}

Expression CustomOptimizerUpdateContext::runtimeScalar(const std::string& name, DataType outputDType, DataType computeDType) const {
    if (name.empty()) {
        throw std::invalid_argument("CustomOptimizer runtime scalar name cannot be empty.");
    }
    return Expression::runtimeScalar(runtimeScalarName(name), outputDType, computeDType);
}

std::string CustomOptimizerUpdateContext::weightsInputName() const { return namePrefix_ + "weights_in"; }

std::string CustomOptimizerUpdateContext::stateInputName(const std::string& name) const { return namePrefix_ + name + "_in"; }

std::string CustomOptimizerUpdateContext::runtimeScalarName(const std::string& name) const { return namePrefix_ + name; }

namespace {

void validateName(const std::string& name, const std::string& what) {
    if (name.empty()) {
        throw std::invalid_argument("CustomOptimizer " + what + " name cannot be empty.");
    }
    if (name == "weights" || name == "weights_in" || name == "gradient") {
        throw std::invalid_argument("CustomOptimizer " + what + " name '" + name + "' is reserved.");
    }
    if (name.length() >= 2 && name[0] == '_' && name[1] == '_') {
        throw std::invalid_argument("CustomOptimizer " + what + " name cannot start with __: " + name);
    }
}

std::vector<uint64_t> stateShapeForWeights(const CustomOptimizerStateSpec& spec, const Tensor& weights) {
    return spec.shape.value_or(weights.getDimensions());
}

}  // namespace

CustomOptimizer::CustomOptimizer(uint64_t id,
                                 std::vector<CustomOptimizerStateSpec> stateSpecs,
                                 UpdateExpressionBuilder updateExpressionBuilder,
                                 RuntimeScalarBuilder runtimeScalarBuilder,
                                 bool supportsSparseRowGradients)
    : Optimizer(id),
      stateSpecs_(std::move(stateSpecs)),
      updateExpressionBuilder_(std::move(updateExpressionBuilder)),
      runtimeScalarBuilder_(std::move(runtimeScalarBuilder)),
      supportsSparseRowGradients_(supportsSparseRowGradients) {
    if (!updateExpressionBuilder_) {
        throw std::invalid_argument("CustomOptimizer requires an update expression builder.");
    }

    std::set<std::string> seenStateNames;
    for (const CustomOptimizerStateSpec& spec : stateSpecs_) {
        validateName(spec.name, "state");
        if (!seenStateNames.insert(spec.name).second) {
            throw std::invalid_argument("Duplicate CustomOptimizer state name: " + spec.name);
        }
        if (spec.shape.has_value()) {
            if (spec.shape->empty()) {
                throw std::invalid_argument("CustomOptimizer state '" + spec.name + "' shape cannot be empty.");
            }
            for (uint64_t dim : spec.shape.value()) {
                if (dim == 0) {
                    throw std::invalid_argument("CustomOptimizer state '" + spec.name + "' shape cannot contain zero dimensions.");
                }
            }
        }
    }
}

void CustomOptimizer::validateReadyToBuildExpression(const Tensor& weights) const {
    THOR_THROW_IF_FALSE(weights.isInitialized());
    THOR_THROW_IF_FALSE(gradientUpdateStream.isInitialized());
    if (weights.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
        throw std::invalid_argument("CustomOptimizer weights tensor must be on GPU.");
    }
}

void CustomOptimizer::ensureStateParameters(const Tensor& weights) {
    validateReadyToBuildExpression(weights);

    for (const CustomOptimizerStateSpec& spec : stateSpecs_) {
        const std::vector<uint64_t> stateShape = stateShapeForWeights(spec, weights);
        if (!hasParameter(spec.name)) {
            auto parameter = std::make_shared<PhysicalParameter>(spec.name, false, stateShape, spec.dtype);
            auto initializer = std::make_shared<ZerosInitializer>();
            parameter->setInitializer(initializer->clone());
            addParameter(parameter);
            parameter->compileStorage(weights);
            parameter->compileInitializer();
            THOR_THROW_IF_FALSE(parameter->getStorage().has_value());
            parameter->initialize(gradientUpdateStream);
            continue;
        }

        std::shared_ptr<PhysicalParameter> parameter = getParameter(spec.name);
        if (!parameter->getStorage().has_value()) {
            parameter->compileStorage(weights);
            parameter->compileInitializer();
            THOR_THROW_IF_FALSE(parameter->getStorage().has_value());
            parameter->initialize(gradientUpdateStream);
        }

        Tensor storage = parameter->getStorage().value();
        if (storage.getDimensions() != stateShape) {
            throw std::runtime_error("CustomOptimizer state '" + spec.name + "' storage shape does not match the requested shape.");
        }
        if (storage.getDataType() != spec.dtype) {
            throw std::runtime_error("CustomOptimizer state '" + spec.name + "' storage dtype does not match the requested dtype.");
        }
    }
}

CustomOptimizerUpdateExpression CustomOptimizer::buildAndValidateUpdateExpression(const CustomOptimizerUpdateContext& context) const {
    CustomOptimizerUpdateExpression updateExpression = updateExpressionBuilder_(context);
    if (updateExpression.outputs.empty()) {
        throw std::runtime_error("CustomOptimizer update expression must produce at least the 'weights' output.");
    }

    std::set<std::string> allowedOutputs{"weights"};
    for (const CustomOptimizerStateSpec& spec : stateSpecs_) {
        allowedOutputs.insert(spec.name);
    }

    std::set<std::string> seenOutputs;
    bool hasWeightsOutput = false;
    for (const auto& [name, _] : updateExpression.outputs) {
        if (name.empty()) {
            throw std::runtime_error("CustomOptimizer update expression output name cannot be empty.");
        }
        if (!allowedOutputs.contains(name)) {
            throw std::runtime_error("CustomOptimizer update expression returned unknown output '" + name + "'.");
        }
        if (!seenOutputs.insert(name).second) {
            throw std::runtime_error("CustomOptimizer update expression returned duplicate output '" + name + "'.");
        }
        if (name == "weights") {
            hasWeightsOutput = true;
        }
    }

    if (!hasWeightsOutput) {
        throw std::runtime_error("CustomOptimizer update expression must produce a 'weights' output.");
    }

    return updateExpression;
}

std::unordered_map<std::string, Tensor> CustomOptimizer::stateInputTensors(const std::string& namePrefix) {
    std::unordered_map<std::string, Tensor> inputs;
    inputs.reserve(stateSpecs_.size());
    for (const CustomOptimizerStateSpec& spec : stateSpecs_) {
        THOR_THROW_IF_FALSE(hasParameter(spec.name));
        std::optional<Tensor> storage = getParameter(spec.name)->getStorage();
        THOR_THROW_IF_FALSE(storage.has_value());
        inputs[namePrefix + spec.name + "_in"] = storage.value();
    }
    return inputs;
}

std::unordered_map<std::string, Tensor> CustomOptimizer::preallocatedOutputTensors(
    const CustomOptimizerUpdateExpression& updateExpression,
    const Tensor& weights) {
    std::unordered_map<std::string, Tensor> outputs;
    outputs.reserve(updateExpression.outputs.size());
    for (const auto& [name, _] : updateExpression.outputs) {
        if (name == "weights") {
            outputs[name] = weights;
            continue;
        }
        THOR_THROW_IF_FALSE(hasParameter(name));
        std::optional<Tensor> storage = getParameter(name)->getStorage();
        THOR_THROW_IF_FALSE(storage.has_value());
        outputs[name] = storage.value();
    }
    return outputs;
}

void CustomOptimizer::compile(const Tensor& weights, Stream& gradientUpdateStream, bool materializeDenseGradient) {
    THOR_THROW_IF_FALSE(!compiled);
    THOR_THROW_IF_FALSE(gradientUpdateStream.isInitialized());
    THOR_THROW_IF_FALSE(weights.isInitialized());
    THOR_THROW_IF_FALSE(weights.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);

    this->gradientUpdateStream = gradientUpdateStream;
    this->weights = weights;
    ensureStateParameters(weights);

    if (!materializeDenseGradient) {
        compiled = true;
        return;
    }

    this->weightsGradient = weights.clone();

    Expression gradientInput = Expression::input("gradient", DataType::FP32, DataType::FP32);
    CustomOptimizerUpdateContext context(weights, gradientInput, "", CustomOptimizerUpdateContext::Mode::Dense);
    CustomOptimizerUpdateExpression updateExpression = buildAndValidateUpdateExpression(context);

    std::unordered_map<std::string, Tensor> stampInputs;
    stampInputs[context.weightsInputName()] = weights;
    THOR_THROW_IF_FALSE(weightsGradient.has_value());
    stampInputs["gradient"] = weightsGradient.value();
    for (auto& [name, tensor] : stateInputTensors("")) {
        stampInputs[name] = tensor;
    }

    std::unordered_map<std::string, TensorScalarBinding> tensorScalarInputs;
    std::unordered_map<std::string, Tensor> preallocatedOutputs = preallocatedOutputTensors(updateExpression, weights);

    Outputs outputs = Expression::outputs(updateExpression.outputs);
    FusedEquation updateEquation = FusedEquation::compile(outputs.physicalOutputs(), weights.getPlacement().getDeviceNum());
    updateEquationStamped = std::make_unique<StampedExecutionPlan>(
        updateEquation.stamp(stampInputs, gradientUpdateStream, tensorScalarInputs, preallocatedOutputs));

    compiled = true;
}

DenseOptimizerExpression CustomOptimizer::toDenseUpdateExpression(const Tensor& weights,
                                                                  const Expression& gradient,
                                                                  const std::string& namePrefix) {
    validateReadyToBuildExpression(weights);
    ensureStateParameters(weights);

    CustomOptimizerUpdateContext context(weights, gradient, namePrefix, CustomOptimizerUpdateContext::Mode::Dense);
    CustomOptimizerUpdateExpression updateExpression = buildAndValidateUpdateExpression(context);

    DenseOptimizerExpression result;
    result.inputs[context.weightsInputName()] = weights;
    for (auto& [name, tensor] : stateInputTensors(namePrefix)) {
        result.inputs[name] = tensor;
    }
    result.preallocatedOutputs = preallocatedOutputTensors(updateExpression, weights);

    Outputs outputs = Expression::outputs(updateExpression.outputs);
    result.outputs = outputs.physicalOutputs();
    return result;
}

SparseRowOptimizerExpression CustomOptimizer::toSparseRowUpdateExpression(const Tensor& weights, SparseRowGradient& sparseRowGradient) {
    if (!supportsSparseRowGradients_) {
        throw std::runtime_error("CustomOptimizer does not support sparse row gradients.");
    }

    validateReadyToBuildExpression(weights);
    sparseRowGradient.validate();

    const std::vector<uint64_t> weightDims = weights.getDimensions();
    if (weightDims.size() != 2 || weightDims[0] == 0 || weightDims[1] == 0) {
        throw std::invalid_argument("Sparse CustomOptimizer weights tensor must have shape [vocabulary_size, embedding_dim].");
    }
    if (sparseRowGradient.embeddingDim != weightDims[1]) {
        throw std::invalid_argument("Sparse CustomOptimizer sparse-gradient embedding dimension does not match weights.");
    }
    if (sparseRowGradient.vocabularySize != weightDims[0]) {
        throw std::invalid_argument("Sparse CustomOptimizer sparse-gradient vocabulary size does not match weights.");
    }

    ensureStateParameters(weights);

    Expression gradientInput = Expression::input("gradient", DataType::FP32, DataType::FP32);
    CustomOptimizerUpdateContext context(weights, gradientInput, "", CustomOptimizerUpdateContext::Mode::SparseRows);
    CustomOptimizerUpdateExpression updateExpression = buildAndValidateUpdateExpression(context);

    SparseRowOptimizerExpression result;
    result.inputs[context.weightsInputName()] = SparseRowUpdateTensorBinding{weights, SparseRowUpdateTensorKind::IndexedRows};
    result.inputs["gradient"] = SparseRowUpdateTensorBinding{sparseRowGradient.values, SparseRowUpdateTensorKind::DenseLogicalRows};
    result.indexedOutputs["weights"] = weights;

    for (const CustomOptimizerStateSpec& spec : stateSpecs_) {
        std::optional<Tensor> storage = getParameter(spec.name)->getStorage();
        THOR_THROW_IF_FALSE(storage.has_value());
        result.inputs[context.stateInputName(spec.name)] = SparseRowUpdateTensorBinding{storage.value(), SparseRowUpdateTensorKind::IndexedRows};
        result.indexedOutputs[spec.name] = storage.value();
    }

    Outputs outputs = Expression::outputs(updateExpression.outputs);
    result.outputs = outputs.physicalOutputs();
    return result;
}

SparseRowGradient CustomOptimizer::compileSparseRows(const Tensor& weights, uint64_t maxSparseRows, Stream& gradientUpdateStream) {
    if (!supportsSparseRowGradients_) {
        throw std::runtime_error("CustomOptimizer does not support sparse row gradients.");
    }

    THOR_THROW_IF_FALSE(!compiled);
    THOR_THROW_IF_FALSE(gradientUpdateStream.isInitialized());
    THOR_THROW_IF_FALSE(weights.isInitialized());
    THOR_THROW_IF_FALSE(weights.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);

    const std::vector<uint64_t> weightDims = weights.getDimensions();
    if (weightDims.size() != 2 || weightDims[0] == 0 || weightDims[1] == 0) {
        throw std::invalid_argument("Sparse CustomOptimizer weights tensor must have shape [vocabulary_size, embedding_dim].");
    }
    if (maxSparseRows == 0) {
        throw std::invalid_argument("Sparse CustomOptimizer maxSparseRows must be non-zero.");
    }
    if (maxSparseRows > weightDims[0]) {
        throw std::invalid_argument("Sparse CustomOptimizer maxSparseRows cannot exceed the embedding vocabulary size.");
    }

    this->gradientUpdateStream = gradientUpdateStream;
    this->weights = weights;
    this->sparseRowGradient = SparseRowGradient::allocate(weights.getPlacement(),
                                                          maxSparseRows,
                                                          weightDims[0],
                                                          weightDims[1],
                                                          DataType::FP32,
                                                          SparseRowGradient::chooseRowDataType(weightDims[0]));
    THOR_THROW_IF_FALSE(sparseRowGradient.has_value());

    ensureStateParameters(weights);
    SparseRowOptimizerExpression updateExpression = toSparseRowUpdateExpression(weights, sparseRowGradient.value());

    sparseUpdatePlan = SparseRowUpdatePlan::compile(updateExpression.outputs,
                                                    sparseRowGradient->rows,
                                                    sparseRowGradient->numRows,
                                                    updateExpression.inputs,
                                                    updateExpression.indexedOutputs,
                                                    weights.getPlacement().getDeviceNum());

    compiled = true;
    return sparseRowGradient.value();
}

std::unordered_map<std::string, float> CustomOptimizer::denseUpdateRuntimeScalars(uint32_t batchSize,
                                                                                  const std::string& namePrefix) {
    THOR_THROW_IF_FALSE(batchSize > 0);
    if (!runtimeScalarBuilder_) {
        return {};
    }
    return runtimeScalarBuilder_(batchSize, namePrefix);
}

std::unordered_map<std::string, float> CustomOptimizer::sparseRowUpdateRuntimeScalars(uint32_t batchSize) {
    return denseUpdateRuntimeScalars(batchSize, "");
}

void CustomOptimizer::updateWeights(uint32_t batchSize) {
    THOR_THROW_IF_FALSE(compiled);
    THOR_THROW_IF_FALSE(weightsGradient.has_value());
    THOR_THROW_IF_FALSE(weightsGradient.value().isInitialized());
    THOR_THROW_IF_FALSE(weightsGradient.value().getPlacement() == weights.getPlacement());
    THOR_THROW_IF_FALSE(updateEquationStamped != nullptr);

    updateEquationStamped->run(denseUpdateRuntimeScalars(batchSize, ""));
}

void CustomOptimizer::updateSparseRows(uint32_t batchSize) {
    THOR_THROW_IF_FALSE(compiled);
    THOR_THROW_IF_FALSE(sparseRowGradient.has_value());
    THOR_THROW_IF_FALSE(sparseUpdatePlan != nullptr);

    sparseRowGradient->validate();
    sparseUpdatePlan->run(sparseRowUpdateRuntimeScalars(batchSize), gradientUpdateStream);
}

std::unordered_map<std::string, float> CustomOptimizer::updateHyperParameters(uint64_t epoch,
                                                                              uint64_t batch,
                                                                              uint64_t batchesPerEpoch) {
    (void)epoch;
    (void)batch;
    (void)batchesPerEpoch;
    return {};
}

std::unordered_map<std::string, float> CustomOptimizer::getAllHyperParameters() { return {}; }

}  // namespace ThorImplementation
