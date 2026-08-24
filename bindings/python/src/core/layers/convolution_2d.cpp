#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <memory>
#include <exception>
#include <optional>
#include <utility>

#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Api/Layers/Learning/Convolution2d.h"
#include "DeepLearning/Api/Layers/Learning/TrainableLayer.h"
#include "DeepLearning/Api/Network/Network.h"
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
               activation, "Convolution2d() argument 'activation'", "thor.activations.Activation, str sentinel, or None", false) ==
           DEFAULT_ACTIVATION_SENTINEL;
}

void applyPythonActivation(Convolution2d::Builder &builder, const nb::object &activation) {
    if (isDefaultActivationSentinel(activation)) {
        // Leave activation unset so the C++ builder applies the learning-layer default.
        return;
    }

    if (activation.is_none()) {
        builder.noActivation();
        return;
    }

    std::shared_ptr<Activation> activationPtr = pybind::castArgument<std::shared_ptr<Activation>>(
        activation, "Convolution2d", "activation", "thor.activations.Activation or None", false);
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
    std::optional<DataType> outputDType = optionalDataTypeFromPython(outputDTypeObj, "Convolution2d.epilogue_input", "output_dtype");
    std::optional<DataType> computeDType = optionalDataTypeFromPython(computeDTypeObj, "Convolution2d.epilogue_input", "compute_dtype");
    return Convolution2d::epilogueInput(computeDType, outputDType);
}

ThorImplementation::Expression makePythonEpilogueAuxInput(const std::string &inputName,
                                                          const nb::object &outputDTypeObj,
                                                          const nb::object &computeDTypeObj) {
    std::optional<DataType> outputDType = optionalDataTypeFromPython(outputDTypeObj, "Convolution2d.epilogue_aux_input", "output_dtype");
    std::optional<DataType> computeDType = optionalDataTypeFromPython(computeDTypeObj, "Convolution2d.epilogue_aux_input", "compute_dtype");
    return Convolution2d::epilogueAuxInput(inputName, computeDType, outputDType);
}

void applyPythonEpilogueInputs(Convolution2d::Builder &builder, const nb::object &epilogueInputs) {
    if (epilogueInputs.is_none()) {
        return;
    }
    nb::dict inputsDict = pybind::castOrTypeError<nb::dict>(
        epilogueInputs, "Convolution2d() argument 'epilogue_inputs'", "dict[str, thor.Tensor] or None", false);
    size_t index = 0;
    for (auto item : inputsDict) {
        const std::string keyContext = "Convolution2d() argument 'epilogue_inputs' key[" + std::to_string(index) + "]";
        std::string name = pybind::castOrTypeError<std::string>(item.first, keyContext, "str", false);
        const std::string valueContext = "Convolution2d() argument 'epilogue_inputs'[" + name + "]";
        Tensor tensor = pybind::castOrTypeError<Tensor>(item.second, valueContext, "thor.Tensor", false);
        builder.epilogueInput(name, tensor);
        ++index;
    }
}

void applyPythonEpilogue(Convolution2d::Builder &builder, const nb::object &epilogue) {
    if (epilogue.is_none()) {
        return;
    }
    builder.epilogue(pybind::castArgument<ThorImplementation::Expression>(
        epilogue, "Convolution2d", "epilogue", "thor.physical.Expression or None", false));
}


struct PythonPaddingSpec {
    ConvolutionPaddingMode mode = ConvolutionPaddingMode::VALID;
    std::array<uint32_t, 4> explicitPadding = {0, 0, 0, 0};
};

PythonPaddingSpec paddingFromPython(const nb::object &padding) {
    if (nb::isinstance<nb::str>(padding)) {
        std::string mode = pybind::castOrTypeError<std::string>(
            padding, "Convolution2d() argument 'padding'", "'valid', 'same', or length-4 sequence[int]", false);
        std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (mode == "valid")
            return {ConvolutionPaddingMode::VALID, {0, 0, 0, 0}};
        if (mode == "same" || mode == "same_upper")
            return {ConvolutionPaddingMode::SAME_UPPER, {0, 0, 0, 0}};
        if (mode == "explicit") {
            throw nb::value_error(
                "Convolution2d() padding='explicit' requires concrete padding; pass (top, bottom, left, right) instead.");
        }
        throw nb::value_error("Convolution2d() argument 'padding' must be 'valid', 'same', or a length-4 sequence[int].");
    }

    if (!nb::isinstance<nb::sequence>(padding)) {
        throw nb::type_error(
            "Convolution2d() argument 'padding': expected 'valid', 'same', or a length-4 sequence[int] "
            "ordered as (top, bottom, left, right).");
    }
    nb::sequence seq = pybind::castOrTypeError<nb::sequence>(
        padding, "Convolution2d() argument 'padding'", "'valid', 'same', or length-4 sequence[int]", false);
    if (nb::len(seq) != 4) {
        throw nb::value_error(
            "Convolution2d instance: explicit padding must contain exactly four values ordered as (top, bottom, left, right).");
    }
    return {
        ConvolutionPaddingMode::EXPLICIT,
        {
            pybind::castOrTypeError<uint32_t>(seq[0], "Convolution2d() argument 'padding'[0]", "non-negative int", false),
            pybind::castOrTypeError<uint32_t>(seq[1], "Convolution2d() argument 'padding'[1]", "non-negative int", false),
            pybind::castOrTypeError<uint32_t>(seq[2], "Convolution2d() argument 'padding'[2]", "non-negative int", false),
            pybind::castOrTypeError<uint32_t>(seq[3], "Convolution2d() argument 'padding'[3]", "non-negative int", false),
        },
    };
}

std::pair<uint32_t, uint32_t> dilationFromPython(const nb::object &dilation) {
    if (nb::isinstance<nb::int_>(dilation)) {
        const uint32_t value = pybind::castOrTypeError<uint32_t>(
            dilation, "Convolution2d() argument 'dilation'", "positive int or length-2 sequence[int]", false);
        if (value == 0)
            throw nb::value_error("Convolution2d instance: dilation must be >= 1.");
        return {value, value};
    }

    if (!nb::isinstance<nb::sequence>(dilation) || nb::isinstance<nb::str>(dilation)) {
        throw nb::type_error("Convolution2d() argument 'dilation': expected positive int or length-2 sequence[int].");
    }
    nb::sequence seq = pybind::castOrTypeError<nb::sequence>(
        dilation, "Convolution2d() argument 'dilation'", "positive int or length-2 sequence[int]", false);
    if (nb::len(seq) != 2)
        throw nb::value_error("Convolution2d instance: dilation sequence must contain exactly two values (height, width).");
    const uint32_t dilationH = pybind::castOrTypeError<uint32_t>(
        seq[0], "Convolution2d() argument 'dilation'[0]", "positive int", false);
    const uint32_t dilationW = pybind::castOrTypeError<uint32_t>(
        seq[1], "Convolution2d() argument 'dilation'[1]", "positive int", false);
    if (dilationH == 0 || dilationW == 0)
        throw nb::value_error("Convolution2d instance: dilation values must be >= 1.");
    return {dilationH, dilationW};
}
}  // namespace

void bind_convolution_2d(nb::module_ &m) {
    auto convolution_2d = nb::class_<Convolution2d, TrainableLayer>(m, "Convolution2d");
    convolution_2d.attr("__module__") = "thor.layers";

    convolution_2d.def(
        "__init__",
        [](Convolution2d *self,
           Network &network,
           Tensor featureInput,
           uint32_t numOutputChannels,
           uint32_t filterHeight,
           uint32_t filterWidth,
           uint32_t verticalStride,
           uint32_t horizontalStride,
           nb::object padding,
           bool hasBias,
           nb::object activation,
           shared_ptr<Initializer> weights_initializer,
           shared_ptr<Initializer> biases_initializer,
           nb::object epilogue,
           nb::object epilogue_inputs,
           nb::object dilation,
           uint32_t groups) {
            const auto &dims = featureInput.getDimensions();
            if (dims.size() != 3) {
                string msg = "Convolution2d instance: feature_input must be a 3D CHW tensor (no batch) but tensor format is " +
                             featureInput.getDescriptorString();
                throw nb::value_error(msg.c_str());
            }

            const uint64_t C = dims[0];
            const uint64_t H = dims[1];
            const uint64_t W = dims[2];

            if (C == 0 || H == 0 || W == 0) {
                string msg = "Convolution2d instance: feature_input dimensions must all be > 0 but tensor format is " +
                             featureInput.getDescriptorString();
                throw nb::value_error(msg.c_str());
            }

            if (numOutputChannels == 0) {
                throw nb::value_error("Convolution2d instance: num_output_channels must be > 0.");
            }
            if (groups == 0 || C % groups != 0 || numOutputChannels % groups != 0)
                throw nb::value_error("Convolution2d instance: groups must divide both input and output channels.");
            if (filterHeight == 0 || filterWidth == 0) {
                string msg =
                    "Convolution2d instance: filter_height and filter_width must be >= 1. "
                    "filter_height=" +
                    to_string(filterHeight) + " filter_width=" + to_string(filterWidth);
                throw nb::value_error(msg.c_str());
            }
            if (verticalStride == 0 || horizontalStride == 0) {
                string msg =
                    "Convolution2d instance: vertical_stride and horizontal_stride must be >= 1. "
                    "vertical_stride=" +
                    to_string(verticalStride) + " horizontal_stride=" + to_string(horizontalStride);
                throw nb::value_error(msg.c_str());
            }

            const auto [dilationH, dilationW] = dilationFromPython(dilation);
            const PythonPaddingSpec paddingSpec = paddingFromPython(padding);

            const uint64_t effectiveFilterH = uint64_t(dilationH) * (uint64_t(filterHeight) - 1ULL) + 1ULL;
            const uint64_t effectiveFilterW = uint64_t(dilationW) * (uint64_t(filterWidth) - 1ULL) + 1ULL;
            if (paddingSpec.mode != ConvolutionPaddingMode::SAME_UPPER) {
                const uint32_t paddingTop = paddingSpec.explicitPadding[0];
                const uint32_t paddingBottom = paddingSpec.explicitPadding[1];
                const uint32_t paddingLeft = paddingSpec.explicitPadding[2];
                const uint32_t paddingRight = paddingSpec.explicitPadding[3];
                const uint64_t paddedH = H + uint64_t(paddingTop) + uint64_t(paddingBottom);
                const uint64_t paddedW = W + uint64_t(paddingLeft) + uint64_t(paddingRight);

                if (effectiveFilterH > paddedH) {
                    string msg = "Convolution2d instance: filter_height effective size " + to_string(effectiveFilterH) +
                                 " is larger than padded input height " + to_string(paddedH) +
                                 ". Input tensor is " + featureInput.getDescriptorString();
                    throw nb::value_error(msg.c_str());
                }
                if (effectiveFilterW > paddedW) {
                    string msg = "Convolution2d instance: filter_width effective size " + to_string(effectiveFilterW) +
                                 " is larger than padded input width " + to_string(paddedW) +
                                 ". Input tensor is " + featureInput.getDescriptorString();
                    throw nb::value_error(msg.c_str());
                }
            }

            Convolution2d::Builder builder;
            builder.network(network)
                .featureInput(featureInput)
                .numOutputChannels(numOutputChannels)
                .filterHeight(filterHeight)
                .filterWidth(filterWidth)
                .groups(groups)
                .verticalStride(verticalStride)
                .horizontalStride(horizontalStride)
                .verticalDilation(dilationH)
                .horizontalDilation(dilationW)
                .hasBias(hasBias);
            switch (paddingSpec.mode) {
                case ConvolutionPaddingMode::VALID:
                    builder.validPadding();
                    break;
                case ConvolutionPaddingMode::SAME_UPPER:
                    builder.samePadding();
                    break;
                case ConvolutionPaddingMode::EXPLICIT:
                    builder.padding(paddingSpec.explicitPadding[0],
                                    paddingSpec.explicitPadding[1],
                                    paddingSpec.explicitPadding[2],
                                    paddingSpec.explicitPadding[3]);
                    break;
            }

            applyPythonActivation(builder, activation);
            applyPythonEpilogueInputs(builder, epilogue_inputs);
            applyPythonEpilogue(builder, epilogue);

            if (weights_initializer != nullptr)
                builder.weightsInitializer(weights_initializer);
            if (biases_initializer != nullptr)
                builder.biasInitializer(biases_initializer);

            Convolution2d built = builder.build();

            new (self) Convolution2d(std::move(built));
        },
        "network"_a,
        "feature_input"_a,
        "num_output_channels"_a,
        "filter_height"_a,
        "filter_width"_a,
        "vertical_stride"_a = 1,
        "horizontal_stride"_a = 1,
        "padding"_a = nb::str("valid"),
        "has_bias"_a = true,
        "activation"_a.none() = nb::str(DEFAULT_ACTIVATION_SENTINEL),
        "weights_initializer"_a = nb::none(),
        "biases_initializer"_a = nb::none(),
        "epilogue"_a.none() = nb::none(),
        "epilogue_inputs"_a.none() = nb::none(),
        "dilation"_a = nb::int_(1),
        "groups"_a = 1);

    convolution_2d.def_static(
        "epilogue_input",
        &makePythonEpilogueInput,
        "output_dtype"_a.none() = nb::none(),
        "compute_dtype"_a.none() = nb::none(),
        R"nbdoc(
            Return the primary tensor input expression expected by a Convolution2d epilogue.
            )nbdoc");

    convolution_2d.def_static(
        "epilogue_aux_input",
        &makePythonEpilogueAuxInput,
        "name"_a,
        "output_dtype"_a.none() = nb::none(),
        "compute_dtype"_a.none() = nb::none(),
        R"nbdoc(
            Return a named auxiliary tensor input expression for a Convolution2d epilogue.
            Bind the same name to a tensor with the ``epilogue_inputs`` constructor argument.
            )nbdoc");

    convolution_2d.def(
        "get_feature_output",
        [](Convolution2d &self) -> Tensor {
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

    convolution_2d.attr("__doc__") = R"nbdoc(
        2D convolution layer.

        Builds a trainable 2D convolutional layer with optional activation,
        dropout, and batch normalization. This layer applies a bank of
        learnable filters over the spatial dimensions of the input tensor.

        Parameters
        ----------
        network : thor.Network
            The network that the layer should be added to. The network
            owns the layer and manages its lifetime.
        feature_input : thor.Tensor
            Input feature tensor for this layer.
            Expected layout matches the underlying Thor tensor convention
            of CHW on the API side. The physical implementation side adds
            the batch layer and uses NCHW.
        num_output_channels : int
            Number of output channels produced by the layer.
        filter_height : int
            Height of each convolution filter (kernel size in the vertical
            dimension).
        filter_width : int
            Width of each convolution filter (kernel size in the horizontal
            dimension).
        vertical_stride : int, default 1
            Stride of the convolution in the vertical direction.
        horizontal_stride : int, default 1
            Stride of the convolution in the horizontal direction.
        padding : {"valid", "same"} or tuple[int, int, int, int], default "valid"
            Padding policy. ``"valid"`` applies no padding. ``"same"`` uses
            SAME_UPPER semantics, choosing output spatial dimensions as
            ``ceil(input / stride)`` and assigning any odd extra padding to
            the bottom/right sides. A length-4 tuple specifies explicit
            ``(top, bottom, left, right)`` zero-padding.
        has_bias : bool, default True
            Whether to learn an additive bias per output channel.
        activation : thor.Activation or None, default thor.activations.Gelu()
            Activation to apply after the convolution
            Pass ``None`` to not use any activation and keep the layer
            purely linear.
        weights_initializer : thor.initializers.Initializer, default thor.initializers.Glorot()
            Initializer for the convolution kernel weights.
        biases_initializer : thor.initializers.Initializer, default thor.initializers.Glorot()
            Initializer for the bias vector.
        epilogue : thor.physical.Expression or None, default None
            Optional expression applied after convolution, bias, and activation.
            Build it from ``Convolution2d.epilogue_input()`` and, when needed,
            ``Convolution2d.epilogue_aux_input(name)``.
        epilogue_inputs : dict[str, thor.Tensor] or None, default None
            Named auxiliary input tensors consumed by the epilogue expression.
            This is intended for residual-style epilogues such as
            ``relu(Convolution2d.epilogue_input() + Convolution2d.epilogue_aux_input("residual"))``.
        dilation : int or tuple[int, int], default 1
            Spacing between kernel elements. An integer applies the same dilation
            vertically and horizontally; a length-2 sequence specifies
            ``(vertical_dilation, horizontal_dilation)``.
        )nbdoc";
}
