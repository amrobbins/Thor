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
        epilogueInputs, "FullyConnected() argument 'epilogue_inputs'", "dict[str, thor.Tensor] or None", false);
    size_t index = 0;
    for (auto item : inputsDict) {
        const std::string keyContext = "FullyConnected() argument 'epilogue_inputs' key[" + std::to_string(index) + "]";
        std::string name = pybind::castOrTypeError<std::string>(item.first, keyContext, "str", false);
        const std::string valueContext = "FullyConnected() argument 'epilogue_inputs'[" + name + "]";
        Tensor tensor = pybind::castOrTypeError<Tensor>(item.second, valueContext, "thor.Tensor", false);
        builder.epilogueInput(name, tensor);
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
           Tensor featureInput,
           uint32_t numOutputFeatures,
           bool hasBias,
           nb::object activation,
           shared_ptr<Initializer> weights_initializer,
           shared_ptr<Initializer> biases_initializer,
           shared_ptr<Optimizer> weights_optimizer,
           shared_ptr<Optimizer> biases_optimizer,
           nb::object epilogue,
           nb::object epilogue_inputs,
           bool preserve_prefix_dimensions,
           nb::object weights_constraints,
           nb::object biases_constraints) {
            if (numOutputFeatures == 0) {
                throw nb::value_error("FullyConnected instance: num_output_features must be > 0.");
            }

            FullyConnected::Builder builder;
            builder.network(network)
                .featureInput(featureInput)
                .numOutputFeatures(numOutputFeatures)
                .hasBias(hasBias)
                .preserveInputPrefixDimensions(preserve_prefix_dimensions);

            applyPythonActivation(builder, activation);
            applyPythonEpilogueInputs(builder, epilogue_inputs);
            applyPythonEpilogue(builder, epilogue);

            if (weights_initializer != nullptr)
                builder.weightsInitializer(weights_initializer);
            if (biases_initializer != nullptr)
                builder.biasInitializer(biases_initializer);
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
        "preserve_prefix_dimensions"_a = false,
        "weights_constraints"_a.none() = nb::none(),
        "biases_constraints"_a.none() = nb::none());

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
            )nbdoc");

    fully_connected.def(
        "get_feature_output",
        [](FullyConnected &self) -> Tensor {
            std::optional<Tensor> maybeFeatureOutput = self.getFeatureOutput();
            return maybeFeatureOutput.value();
        },
        R"nbdoc(
            Return the output tensor produced by this layer.

            Returns
            -------
            thor.Tensor
                The feature output tensor handle.
            )nbdoc");

    fully_connected.attr("__doc__") = R"nbdoc(
        Fully connected (dense) layer.

        Builds a fully connected layer with optional activation, dropout,
        and batch normalization. This is the standard affine layer

            y = W x + b

        optionally preceded by batch normalization and or drop out,
        and optionally followed by a non-linear activation.

        The connection order of the optional layers, when used, is the following:
        [batch norm] -> [drop out] -> [fully connected] -> [activation]

        Parameters
        ----------
        network : thor.Network
            The network that the layer should be added to.
        feature_input : thor.Tensor
            Input feature tensor for this layer.
        num_output_features : int
            Number of output features (units) produced by this layer.
        has_bias : bool, default True
            Whether to learn an additive bias term.
        preserve_prefix_dimensions : bool, default False
            If False, all non-batch input dimensions are flattened into one dense feature vector.
            If True, only the final input dimension is treated as features and preceding logical
            dimensions are preserved in the output. This is the high-throughput tokenwise projection
            mode for tensors shaped like [sequence, hidden].
        activation : thor.Activation or None, default thor.activations.Gelu()
            Activation to apply after the linear transform (and optional
            batch normalization). Pass ``None`` to not use any activation and
            keep the layer purely linear.
        weights_initializer : thor.initializers.Initializer, default thor.initializers.Glorot()
            Initializer for the weight matrix.
        biases_initializer : thor.initializers.Initializer, default thor.initializers.Glorot()
            Initializer for the bias vector.
        epilogue : thor.physical.Expression or None, default None
            Optional expression applied after the affine transform and activation.
            Build it from ``FullyConnected.epilogue_input()``.
        )nbdoc";
}
