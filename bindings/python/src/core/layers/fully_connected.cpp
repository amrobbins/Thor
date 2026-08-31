#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <memory>
#include <exception>
#include <optional>

#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Api/Layers/Learning/FullyConnected.h"
#include "DeepLearning/Api/Layers/Learning/TrainableLayer.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Parameter/ParameterConstraint.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"
#include "DeepLearning/Api/Tensor/Tensor.h"
#include "Utilities/Expression/Expression.h"
#include "bindings/python/src/core/cast.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;
namespace pybind = Thor::PythonBindings;

using DataType = ThorImplementation::DataType;

namespace {
constexpr const char *DEFAULT_ACTIVATION_SENTINEL = "__thor_default_activation__";

bool isDefaultActivationSentinel(const nb::object &activation) {
    if (!nb::isinstance<nb::str>(activation)) {
        return false;
    }
    return pybind::castOrTypeError<std::string>(
               activation, "FullyConnected() argument 'activation'", "thor.activations.Activation, str sentinel, or None", false) ==
           DEFAULT_ACTIVATION_SENTINEL;
}

void applyPythonActivation(FullyConnected::Builder &builder, const nb::object &activation) {
    if (isDefaultActivationSentinel(activation)) {
        // Leave activation unset so the C++ builder applies the learning-layer default.
        return;
    }

    if (activation.is_none()) {
        builder.noActivation();
        return;
    }

    std::shared_ptr<Activation> activationPtr = pybind::castArgument<std::shared_ptr<Activation>>(
        activation, "FullyConnected", "activation", "thor.activations.Activation or None", false);
    if (activationPtr == nullptr) {
        builder.noActivation();
    } else {
        builder.activation(activationPtr);
    }
}

std::optional<DataType> optionalDataTypeFromPython(const nb::object &obj,
                                                   const char *functionName,
                                                   const char *argumentName) {
    if (obj.is_none()) {
        return std::nullopt;
    }
    return pybind::castArgument<DataType>(obj, functionName, argumentName, "thor.DataType or None", false);
}

ThorImplementation::Expression makePythonEpilogueInput(const nb::object &outputDTypeObj, const nb::object &computeDTypeObj) {
    std::optional<DataType> outputDType = optionalDataTypeFromPython(outputDTypeObj, "FullyConnected.epilogue_input", "output_dtype");
    std::optional<DataType> computeDType = optionalDataTypeFromPython(computeDTypeObj, "FullyConnected.epilogue_input", "compute_dtype");
    return FullyConnected::epilogueInput(computeDType, outputDType);
}

ThorImplementation::Expression makePythonEpilogueAuxInput(const std::string &inputName,
                                                          const nb::object &outputDTypeObj,
                                                          const nb::object &computeDTypeObj) {
    std::optional<DataType> outputDType = optionalDataTypeFromPython(outputDTypeObj, "FullyConnected.epilogue_aux_input", "output_dtype");
    std::optional<DataType> computeDType = optionalDataTypeFromPython(computeDTypeObj, "FullyConnected.epilogue_aux_input", "compute_dtype");
    return FullyConnected::epilogueAuxInput(inputName, computeDType, outputDType);
}

void applyPythonEpilogueInputs(FullyConnected::Builder &builder, const nb::object &epilogueInputs) {
    if (epilogueInputs.is_none()) {
        return;
    }
    nb::dict inputsDict = pybind::castOrTypeError<nb::dict>(
        epilogueInputs, "FullyConnected() argument 'epilogue_inputs'", "dict[str, thor.Tensor | thor.RaggedTensor] or None", false);
    size_t index = 0;
    for (auto item : inputsDict) {
        const std::string keyContext = "FullyConnected() argument 'epilogue_inputs' key[" + std::to_string(index) + "]";
        std::string name = pybind::castOrTypeError<std::string>(item.first, keyContext, "str", false);
        const std::string valueContext = "FullyConnected() argument 'epilogue_inputs'[" + name + "]";
        if (nb::isinstance<RaggedTensor>(item.second)) {
            builder.epilogueInput(name, nb::cast<RaggedTensor>(item.second));
        } else if (nb::isinstance<Tensor>(item.second)) {
            builder.epilogueInput(name, nb::cast<Tensor>(item.second));
        } else {
            throw nb::type_error((valueContext + ": expected thor.Tensor or thor.RaggedTensor, got " +
                                  pybind::pythonTypeName(item.second)).c_str());
        }
        ++index;
    }
}

std::vector<std::shared_ptr<ParameterConstraint>> constraintsFromPython(const nb::object& obj, const char* argumentName) {
    std::vector<std::shared_ptr<ParameterConstraint>> constraints;
    if (obj.is_none()) {
        return constraints;
    }

    auto appendConstraint = [&constraints, argumentName](const nb::handle& handle, size_t index) {
        const std::string context = std::string("FullyConnected() argument '") + argumentName + "'[" + std::to_string(index) + "]";
        std::shared_ptr<ParameterConstraint> constraint = pybind::castOrTypeError<std::shared_ptr<ParameterConstraint>>(
            handle, context, "thor.constraints.ParameterConstraint", false);
        if (constraint == nullptr) {
            throw nb::value_error((std::string("FullyConnected() argument '") + argumentName + "' may not contain None").c_str());
        }
        constraints.push_back(constraint->clone());
    };

    std::shared_ptr<ParameterConstraint> single;
    if (pybind::tryCast(obj, single, false)) {
        if (single == nullptr) {
            throw nb::value_error((std::string("FullyConnected() argument '") + argumentName + "' may not be None").c_str());
        }
        constraints.push_back(single->clone());
        return constraints;
    }

    if (!nb::isinstance<nb::sequence>(obj) || nb::isinstance<nb::str>(obj)) {
        throw nb::type_error((std::string("FullyConnected() argument '") + argumentName +
                              "': expected thor.constraints.ParameterConstraint, sequence of constraints, or None, got " +
                              pybind::pythonTypeName(obj)).c_str());
    }

    nb::sequence seq = pybind::castOrTypeError<nb::sequence>(
        obj, std::string("FullyConnected() argument '") + argumentName + "'", "sequence of thor.constraints.ParameterConstraint", false);
    constraints.reserve(nb::len(seq));
    size_t index = 0;
    for (nb::handle item : seq) {
        appendConstraint(item, index++);
    }
    return constraints;
}

void applyConstraints(FullyConnected::Builder& builder, const nb::object& weightsConstraints, const nb::object& biasesConstraints) {
    builder.weightsConstraints(constraintsFromPython(weightsConstraints, "weights_constraints"));
    builder.biasesConstraints(constraintsFromPython(biasesConstraints, "biases_constraints"));
}
void applyPythonEpilogue(FullyConnected::Builder &builder, const nb::object &epilogue) {
    if (epilogue.is_none()) {
        return;
    }
    builder.epilogue(pybind::castArgument<ThorImplementation::Expression>(
        epilogue, "FullyConnected", "epilogue", "thor.physical.Expression or None", false));
}
}  // namespace

void bind_fully_connected(nb::module_ &m) {
    auto fully_connected = nb::class_<FullyConnected, TrainableLayer>(m, "FullyConnected");
    fully_connected.attr("__module__") = "thor.layers";

    fully_connected.def(
        "__init__",
        [](FullyConnected *self,
           Network &network,
           nb::object featureInput,
           uint32_t numOutputFeatures,
           bool hasBias,
           nb::object activation,
           shared_ptr<Initializer> weights_initializer,
           shared_ptr<Initializer> biases_initializer,
           shared_ptr<Optimizer> weights_optimizer,
           shared_ptr<Optimizer> biases_optimizer,
           nb::object epilogue,
           nb::object epilogue_inputs,
           std::optional<bool> preserve_prefix_dimensions,
           nb::object weights_constraints,
           nb::object biases_constraints,
           nb::object weights_data_type,
           nb::object compute_data_type,
           nb::object output_data_type,
           float output_dropout_probability,
           std::optional<int64_t> output_dropout_seed,
           nb::object residual_input) {
            if (numOutputFeatures == 0) {
                throw nb::value_error("FullyConnected instance: num_output_features must be > 0.");
            }

            FullyConnected::Builder builder;
            builder.network(network);
            if (nb::isinstance<RaggedTensor>(featureInput)) {
                builder.featureInput(nb::cast<RaggedTensor>(featureInput));
            } else if (nb::isinstance<Tensor>(featureInput)) {
                builder.featureInput(nb::cast<Tensor>(featureInput));
            } else {
                throw nb::type_error("FullyConnected feature_input must be thor.Tensor or thor.RaggedTensor.");
            }
            builder.numOutputFeatures(numOutputFeatures).hasBias(hasBias);
            if (preserve_prefix_dimensions.has_value()) {
                builder.preserveInputPrefixDimensions(preserve_prefix_dimensions.value());
            }

            applyPythonActivation(builder, activation);
            applyPythonEpilogueInputs(builder, epilogue_inputs);
            applyPythonEpilogue(builder, epilogue);

            if (weights_initializer != nullptr)
                builder.weightsInitializer(weights_initializer);
            if (biases_initializer != nullptr)
                builder.biasInitializer(biases_initializer);

            std::optional<DataType> weightsDataType = optionalDataTypeFromPython(weights_data_type, "FullyConnected", "weights_data_type");
            std::optional<DataType> computeDataType = optionalDataTypeFromPython(compute_data_type, "FullyConnected", "compute_data_type");
            std::optional<DataType> outputDataType = optionalDataTypeFromPython(output_data_type, "FullyConnected", "output_data_type");
            if (weightsDataType.has_value())
                builder.weightsDataType(weightsDataType.value());
            if (computeDataType.has_value())
                builder.computeDataType(computeDataType.value());
            if (outputDataType.has_value())
                builder.outputDataType(outputDataType.value());

            builder.outputDropoutProbability(output_dropout_probability);
            if (output_dropout_seed.has_value()) builder.outputDropoutSeed(output_dropout_seed.value());
            if (!residual_input.is_none()) {
                if (nb::isinstance<RaggedTensor>(residual_input)) {
                    builder.residualInput(nb::cast<RaggedTensor>(residual_input));
                } else if (nb::isinstance<Tensor>(residual_input)) {
                    builder.residualInput(nb::cast<Tensor>(residual_input));
                } else {
                    throw nb::type_error("FullyConnected residual_input must be thor.Tensor, thor.RaggedTensor, or None.");
                }
            }

            builder.weightsOptimizer(weights_optimizer);
            builder.biasesOptimizer(biases_optimizer);
            applyConstraints(builder, weights_constraints, biases_constraints);

            FullyConnected built = builder.build();

            new (self) FullyConnected(std::move(built));
        },
        "network"_a,
        "feature_input"_a,
        "num_output_features"_a,
        "has_bias"_a = true,
        "activation"_a.none() = nb::str(DEFAULT_ACTIVATION_SENTINEL),
        "weights_initializer"_a.none() = nb::none(),
        "biases_initializer"_a.none() = nb::none(),
        "weights_optimizer"_a.none() = nb::none(),
        "biases_optimizer"_a.none() = nb::none(),
        "epilogue"_a.none() = nb::none(),
        "epilogue_inputs"_a.none() = nb::none(),
        "preserve_prefix_dimensions"_a.none() = nb::none(),
        "weights_constraints"_a.none() = nb::none(),
        "biases_constraints"_a.none() = nb::none(),
        "weights_data_type"_a.none() = nb::none(),
        "compute_data_type"_a.none() = nb::none(),
        "output_data_type"_a.none() = nb::none(),
        "output_dropout_probability"_a = 0.0f,
        "output_dropout_seed"_a.none() = nb::none(),
        "residual_input"_a.none() = nb::none());

    fully_connected.def_static(
        "epilogue_input",
        &makePythonEpilogueInput,
        "output_dtype"_a.none() = nb::none(),
        "compute_dtype"_a.none() = nb::none(),
        R"nbdoc(
            Return the single tensor input expression expected by a FullyConnected epilogue.
            )nbdoc");

    fully_connected.def_static(
        "epilogue_aux_input",
        &makePythonEpilogueAuxInput,
        "name"_a,
        "output_dtype"_a.none() = nb::none(),
        "compute_dtype"_a.none() = nb::none(),
        R"nbdoc(
            Return a named auxiliary tensor input expression for a FullyConnected epilogue.
            Bind the same name to a tensor with the ``epilogue_inputs`` constructor argument.
            Ragged FullyConnected requires each auxiliary binding to be a ``thor.RaggedTensor``
            with the exact same row partition as ``feature_input``.
            )nbdoc");

    fully_connected.def("get_weights_data_type", &FullyConnected::getWeightsDataType);
    fully_connected.def("get_compute_data_type", &FullyConnected::getComputeDataType);
    fully_connected.def("get_output_data_type", &FullyConnected::getOutputDataType);

    fully_connected.def(
        "get_feature_output",
        [](FullyConnected &self) -> nb::object {
            if (std::optional<RaggedTensor> raggedOutput = self.getRaggedFeatureOutput(); raggedOutput.has_value()) {
                return nb::cast(raggedOutput.value());
            }
            return nb::cast(self.getFeatureOutput().value());
        },
        R"nbdoc(
            Return the logical output produced by this layer.

            Returns
            -------
            thor.Tensor or thor.RaggedTensor
                The feature output handle. Ragged inputs produce RaggedTensor outputs with the
                same row partition.
            )nbdoc");

    fully_connected.def("get_use_ragged", &FullyConnected::getUseRagged);
    fully_connected.def("get_output_dropout_probability", &FullyConnected::getOutputDropoutProbability);
    fully_connected.def("get_output_dropout_seed", &FullyConnected::getOutputDropoutSeed);
    fully_connected.def("get_use_residual", &FullyConnected::getUseResidual);
    fully_connected.def("get_residual_input", [](FullyConnected &self) -> nb::object {
        if (self.getRaggedResidualInput().has_value()) {
            return nb::cast(self.getRaggedResidualInput().value());
        }
        const std::optional<Tensor> residual = self.getResidualInput();
        return residual.has_value() ? nb::cast(residual.value()) : nb::none();
    });
    fully_connected.def("set_training_dropout_enabled",
                        [](FullyConnected &layer, bool enabled) { layer.setTrainingDropoutEnabled(enabled); },
                        "enabled"_a);
    fully_connected.def("is_training_dropout_enabled",
                        [](const FullyConnected &layer) { return layer.isTrainingDropoutEnabled(); });

    fully_connected.attr("__doc__") = R"nbdoc(
        Fully connected layer.

        Computes an affine projection followed by the optional activation. ``output_dropout_probability``
        applies dropout after that branch. When ``residual_input`` is supplied, the exact training contract is

            residual_input + dropout(activation(W x + b))

        with omitted operations removed. When output dropout is inactive, Thor keeps the residual add adjacent
        to the projection so the compiler can use the existing GEMM residual/beta fusion where legal. When
        output dropout is active, Thor uses the shared native fused dropout+residual post-op. Validation and
        inference always use the deterministic no-dropout branch. Ragged residuals must share the exact input
        row partition.

        Parameters
        ----------
        network : thor.Network
            The network that the layer should be added to.
        feature_input : thor.Tensor or thor.RaggedTensor
            Input feature tensor for this layer. Ragged inputs are projected tokenwise over their
            packed values and preserve the row partition.
        num_output_features : int
            Number of output features (units) produced by this layer.
        has_bias : bool, default True
            Whether to learn an additive bias term.
        preserve_prefix_dimensions : bool or None, default None
            When omitted, dense inputs default to False and ragged inputs default to True.
            If False, all non-batch dense input dimensions are flattened into one dense feature vector.
            If True, only the final input dimension is treated as features and preceding logical
            dimensions are preserved in the output. Ragged inputs require prefix preservation.
        activation : thor.Activation or None, default thor.activations.Gelu()
            Activation to apply after the linear transform. Pass ``None`` to keep the layer purely
            linear. Ragged FullyConnected uses the same activation contract as dense FullyConnected;
            the activation must support standalone ragged execution.
        weights_initializer : thor.initializers.Initializer, default thor.initializers.Glorot()
            Initializer for the weight matrix.
        biases_initializer : thor.initializers.Initializer, default thor.initializers.Glorot()
            Initializer for the bias vector.
        weights_data_type : thor.DataType or None, default None
            Storage type for the weight matrix. When omitted, uses the feature input type.
        compute_data_type : thor.DataType or None, default None
            Arithmetic type for the matrix multiply. When omitted, uses the feature input type, so
            FP32 inputs default to strict FP32 compute. Pass ``thor.DataType.tf32`` to opt into TF32.
        output_data_type : thor.DataType or None, default None
            Storage type for the layer output. When omitted, uses the feature input type.
        output_dropout_probability : float, default 0.0
            Dropout probability on the final FullyConnected branch after bias and activation.
        output_dropout_seed : int or None, default None
            Optional deterministic Philox seed. When omitted with dropout enabled, Thor chooses an independent
            per-layer seed and persists it in the architecture.
        residual_input : thor.Tensor, thor.RaggedTensor, or None, default None
            Optional skip tensor. The exact contract is ``residual_input + dropout(fc_output)`` during training.
            Ragged residuals must share the exact row partition of ``feature_input``.
        epilogue : thor.physical.Expression or None, default None
            Optional expression applied after the affine transform and activation.
            Build it from ``FullyConnected.epilogue_input()``.
        epilogue_inputs : dict[str, thor.Tensor or thor.RaggedTensor] or None, default None
            Named auxiliary tensors referenced by ``FullyConnected.epilogue_aux_input()``.
            Ragged FullyConnected requires every auxiliary to be a RaggedTensor sharing the exact
            row partition of ``feature_input``.
        )nbdoc";
}
