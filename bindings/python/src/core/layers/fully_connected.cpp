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

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

using DataType = ThorImplementation::DataType;

namespace {
constexpr const char *DEFAULT_ACTIVATION_SENTINEL = "__thor_default_activation__";

bool isDefaultActivationSentinel(const nb::object &activation) {
    return nb::isinstance<nb::str>(activation) && nb::cast<std::string>(activation) == DEFAULT_ACTIVATION_SENTINEL;
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

    std::shared_ptr<Activation> activationPtr;
    try {
        activationPtr = nb::cast<std::shared_ptr<Activation>>(activation);
    } catch (const std::exception &) {
        throw nb::type_error("activation must be a thor.activations.Activation instance or None");
    }
    if (activationPtr == nullptr) {
        builder.noActivation();
    } else {
        builder.activation(activationPtr);
    }
}

std::optional<DataType> optionalDataTypeFromPython(const nb::object &obj) {
    if (obj.is_none()) {
        return std::nullopt;
    }
    return nb::cast<DataType>(obj);
}

ThorImplementation::Expression makePythonEpilogueInput(const nb::object &outputDTypeObj, const nb::object &computeDTypeObj) {
    std::optional<DataType> outputDType = optionalDataTypeFromPython(outputDTypeObj);
    std::optional<DataType> computeDType = optionalDataTypeFromPython(computeDTypeObj);
    return FullyConnected::epilogueInput(computeDType, outputDType);
}

ThorImplementation::Expression makePythonEpilogueAuxInput(const std::string &inputName,
                                                          const nb::object &outputDTypeObj,
                                                          const nb::object &computeDTypeObj) {
    std::optional<DataType> outputDType = optionalDataTypeFromPython(outputDTypeObj);
    std::optional<DataType> computeDType = optionalDataTypeFromPython(computeDTypeObj);
    return FullyConnected::epilogueAuxInput(inputName, computeDType, outputDType);
}

void applyPythonEpilogueInputs(FullyConnected::Builder &builder, const nb::object &epilogueInputs) {
    if (epilogueInputs.is_none()) {
        return;
    }
    if (!nb::isinstance<nb::dict>(epilogueInputs)) {
        throw nb::type_error("epilogue_inputs must be a dict[str, thor.Tensor] or None");
    }
    nb::dict inputsDict = nb::cast<nb::dict>(epilogueInputs);
    for (auto item : inputsDict) {
        std::string name = nb::cast<std::string>(item.first);
        Tensor tensor = nb::cast<Tensor>(item.second);
        builder.epilogueInput(name, tensor);
    }
}


std::vector<std::shared_ptr<ParameterConstraint>> constraintsFromPython(const nb::object& obj, const char* argumentName) {
    std::vector<std::shared_ptr<ParameterConstraint>> constraints;
    if (obj.is_none()) {
        return constraints;
    }

    auto appendConstraint = [&constraints, argumentName](const nb::handle& handle) {
        std::shared_ptr<ParameterConstraint> constraint;
        try {
            constraint = nb::cast<std::shared_ptr<ParameterConstraint>>(handle);
        } catch (const std::exception&) {
            throw nb::type_error((std::string(argumentName) + " must contain thor.ParameterConstraint instances").c_str());
        }
        if (constraint == nullptr) {
            throw nb::value_error((std::string(argumentName) + " may not contain None").c_str());
        }
        constraints.push_back(constraint->clone());
    };

    try {
        std::shared_ptr<ParameterConstraint> single = nb::cast<std::shared_ptr<ParameterConstraint>>(obj);
        if (single != nullptr) {
            constraints.push_back(single->clone());
            return constraints;
        }
    } catch (const std::exception&) {
    }

    if (!nb::isinstance<nb::sequence>(obj) || nb::isinstance<nb::str>(obj)) {
        throw nb::type_error((std::string(argumentName) + " must be a thor.ParameterConstraint, a sequence of constraints, or None").c_str());
    }

    nb::sequence seq = nb::cast<nb::sequence>(obj);
    constraints.reserve(nb::len(seq));
    for (nb::handle item : seq) {
        appendConstraint(item);
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
    if (!nb::isinstance<ThorImplementation::Expression>(epilogue)) {
        throw nb::type_error("epilogue must be a thor.physical.Expression instance or None");
    }
    builder.epilogue(nb::cast<ThorImplementation::Expression>(epilogue));
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
