#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <memory>
#include <exception>
#include <optional>
#include <tuple>

#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Api/Layers/Learning/Convolution3d.h"
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
               activation, "Convolution3d() argument 'activation'", "thor.activations.Activation, str sentinel, or None", false) ==
           DEFAULT_ACTIVATION_SENTINEL;
}

void applyPythonActivation(Convolution3d::Builder &builder, const nb::object &activation) {
    if (isDefaultActivationSentinel(activation)) {
        // Leave activation unset so the C++ builder applies the learning-layer default.
        return;
    }

    if (activation.is_none()) {
        builder.noActivation();
        return;
    }

    std::shared_ptr<Activation> activationPtr = pybind::castArgument<std::shared_ptr<Activation>>(
        activation, "Convolution3d", "activation", "thor.activations.Activation or None", false);
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
    std::optional<DataType> outputDType = optionalDataTypeFromPython(outputDTypeObj, "Convolution3d.epilogue_input", "output_dtype");
    std::optional<DataType> computeDType = optionalDataTypeFromPython(computeDTypeObj, "Convolution3d.epilogue_input", "compute_dtype");
    return Convolution3d::epilogueInput(computeDType, outputDType);
}

ThorImplementation::Expression makePythonEpilogueAuxInput(const std::string &inputName,
                                                          const nb::object &outputDTypeObj,
                                                          const nb::object &computeDTypeObj) {
    std::optional<DataType> outputDType = optionalDataTypeFromPython(outputDTypeObj, "Convolution3d.epilogue_aux_input", "output_dtype");
    std::optional<DataType> computeDType = optionalDataTypeFromPython(computeDTypeObj, "Convolution3d.epilogue_aux_input", "compute_dtype");
    return Convolution3d::epilogueAuxInput(inputName, computeDType, outputDType);
}

void applyPythonEpilogueInputs(Convolution3d::Builder &builder, const nb::object &epilogueInputs) {
    if (epilogueInputs.is_none()) {
        return;
    }
    nb::dict inputsDict = pybind::castOrTypeError<nb::dict>(
        epilogueInputs, "Convolution3d() argument 'epilogue_inputs'", "dict[str, thor.Tensor] or None", false);
    size_t index = 0;
    for (auto item : inputsDict) {
        const std::string keyContext = "Convolution3d() argument 'epilogue_inputs' key[" + std::to_string(index) + "]";
        std::string name = pybind::castOrTypeError<std::string>(item.first, keyContext, "str", false);
        const std::string valueContext = "Convolution3d() argument 'epilogue_inputs'[" + name + "]";
        Tensor tensor = pybind::castOrTypeError<Tensor>(item.second, valueContext, "thor.Tensor", false);
        builder.epilogueInput(name, tensor);
        ++index;
    }
}

void applyPythonEpilogue(Convolution3d::Builder &builder, const nb::object &epilogue) {
    if (epilogue.is_none()) {
        return;
    }
    builder.epilogue(pybind::castArgument<ThorImplementation::Expression>(
        epilogue, "Convolution3d", "epilogue", "thor.physical.Expression or None", false));
}

struct PythonPaddingSpec {
    Convolution3dPaddingMode mode = Convolution3dPaddingMode::VALID;
    std::array<uint32_t, 6> explicitPadding = {0, 0, 0, 0, 0, 0};
};

PythonPaddingSpec paddingFromPython(const nb::object &padding,
                                    uint32_t legacyDepthPadding,
                                    uint32_t legacyVerticalPadding,
                                    uint32_t legacyHorizontalPadding) {
    const bool hasLegacyPadding = legacyDepthPadding != 0 || legacyVerticalPadding != 0 || legacyHorizontalPadding != 0;
    auto legacyPaddingSpec = [&]() {
        return PythonPaddingSpec{Convolution3dPaddingMode::EXPLICIT,
                                 {legacyDepthPadding,
                                  legacyDepthPadding,
                                  legacyVerticalPadding,
                                  legacyVerticalPadding,
                                  legacyHorizontalPadding,
                                  legacyHorizontalPadding}};
    };

    if (padding.is_none())
        return hasLegacyPadding ? legacyPaddingSpec()
                                : PythonPaddingSpec{Convolution3dPaddingMode::VALID, {0, 0, 0, 0, 0, 0}};

    if (hasLegacyPadding) {
        // ``padding`` defaults to "valid" so legacy callers can keep using the
        // old symmetric padding keywords without having to opt out of the new
        // argument. Any non-default modern policy is ambiguous and rejected.
        if (nb::isinstance<nb::str>(padding)) {
            std::string mode = pybind::castOrTypeError<std::string>(
                padding, "Convolution3d() argument 'padding'", "'valid', 'same', or length-6 sequence[int]", false);
            std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (mode == "valid")
                return legacyPaddingSpec();
        }
        throw nb::value_error(
            "Convolution3d instance: padding cannot be combined with depth_padding, vertical_padding, or horizontal_padding; "
            "use padding=(front, back, top, bottom, left, right) instead.");
    }

    if (nb::isinstance<nb::str>(padding)) {
        std::string mode = pybind::castOrTypeError<std::string>(
            padding, "Convolution3d() argument 'padding'", "'valid', 'same', or length-6 sequence[int]", false);
        std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (mode == "valid")
            return {Convolution3dPaddingMode::VALID, {0, 0, 0, 0, 0, 0}};
        if (mode == "same" || mode == "same_upper")
            return {Convolution3dPaddingMode::SAME_UPPER, {0, 0, 0, 0, 0, 0}};
        if (mode == "explicit") {
            throw nb::value_error(
                "Convolution3d() padding='explicit' requires concrete padding; pass "
                "(front, back, top, bottom, left, right) instead.");
        }
        throw nb::value_error(
            "Convolution3d() argument 'padding' must be 'valid', 'same', or a length-6 sequence[int].");
    }

    if (!nb::isinstance<nb::sequence>(padding)) {
        throw nb::type_error(
            "Convolution3d() argument 'padding': expected 'valid', 'same', or a length-6 sequence[int] "
            "ordered as (front, back, top, bottom, left, right).");
    }
    nb::sequence seq = pybind::castOrTypeError<nb::sequence>(
        padding, "Convolution3d() argument 'padding'", "'valid', 'same', or length-6 sequence[int]", false);
    if (nb::len(seq) != 6) {
        throw nb::value_error(
            "Convolution3d instance: explicit padding must contain exactly six values ordered as "
            "(front, back, top, bottom, left, right).");
    }
    return {Convolution3dPaddingMode::EXPLICIT,
            {pybind::castOrTypeError<uint32_t>(seq[0], "Convolution3d() argument 'padding'[0]", "non-negative int", false),
             pybind::castOrTypeError<uint32_t>(seq[1], "Convolution3d() argument 'padding'[1]", "non-negative int", false),
             pybind::castOrTypeError<uint32_t>(seq[2], "Convolution3d() argument 'padding'[2]", "non-negative int", false),
             pybind::castOrTypeError<uint32_t>(seq[3], "Convolution3d() argument 'padding'[3]", "non-negative int", false),
             pybind::castOrTypeError<uint32_t>(seq[4], "Convolution3d() argument 'padding'[4]", "non-negative int", false),
             pybind::castOrTypeError<uint32_t>(seq[5], "Convolution3d() argument 'padding'[5]", "non-negative int", false)}};
}

std::tuple<uint32_t, uint32_t, uint32_t> dilationFromPython(const nb::object &dilation) {
    if (nb::isinstance<nb::int_>(dilation)) {
        const uint32_t value = pybind::castOrTypeError<uint32_t>(
            dilation, "Convolution3d() argument 'dilation'", "positive int or length-3 sequence[int]", false);
        if (value == 0)
            throw nb::value_error("Convolution3d instance: dilation must be >= 1.");
        return {value, value, value};
    }

    if (!nb::isinstance<nb::sequence>(dilation) || nb::isinstance<nb::str>(dilation)) {
        throw nb::type_error("Convolution3d() argument 'dilation': expected positive int or length-3 sequence[int].");
    }
    nb::sequence seq = pybind::castOrTypeError<nb::sequence>(
        dilation, "Convolution3d() argument 'dilation'", "positive int or length-3 sequence[int]", false);
    if (nb::len(seq) != 3)
        throw nb::value_error(
            "Convolution3d instance: dilation sequence must contain exactly three values (depth, height, width).");
    const uint32_t dilationD = pybind::castOrTypeError<uint32_t>(
        seq[0], "Convolution3d() argument 'dilation'[0]", "positive int", false);
    const uint32_t dilationH = pybind::castOrTypeError<uint32_t>(
        seq[1], "Convolution3d() argument 'dilation'[1]", "positive int", false);
    const uint32_t dilationW = pybind::castOrTypeError<uint32_t>(
        seq[2], "Convolution3d() argument 'dilation'[2]", "positive int", false);
    if (dilationD == 0 || dilationH == 0 || dilationW == 0)
        throw nb::value_error("Convolution3d instance: dilation values must be >= 1.");
    return {dilationD, dilationH, dilationW};
}
}  // namespace

void bind_convolution_3d(nb::module_ &m) {
    auto convolution_3d = nb::class_<Convolution3d, TrainableLayer>(m, "Convolution3d");
    convolution_3d.attr("__module__") = "thor.layers";

    convolution_3d.def(
        "__init__",
        [](Convolution3d *self,
           Network &network,
           Tensor featureInput,
           uint32_t numOutputChannels,
           uint32_t filterDepth,
           uint32_t filterHeight,
           uint32_t filterWidth,
           uint32_t depthStride,
           uint32_t verticalStride,
           uint32_t horizontalStride,
           uint32_t depthPadding,
           uint32_t verticalPadding,
           uint32_t horizontalPadding,
           bool hasBias,
           nb::object activation,
           shared_ptr<Initializer> weights_initializer,
           shared_ptr<Initializer> biases_initializer,
           nb::object epilogue,
           nb::object epilogue_inputs,
           uint32_t groups,
           DataType computeDataType,
           nb::object padding,
           nb::object dilation) {
            const auto &dims = featureInput.getDimensions();
            if (dims.size() != 4) {
                string msg = "Convolution3d instance: feature_input must be a 4D CDHW tensor (no batch) but tensor format is " +
                             featureInput.getDescriptorString();
                throw nb::value_error(msg.c_str());
            }

            const uint64_t C = dims[0];
            const uint64_t D = dims[1];
            const uint64_t H = dims[2];
            const uint64_t W = dims[3];

            if (C == 0 || D == 0 || H == 0 || W == 0) {
                string msg = "Convolution3d instance: feature_input dimensions must all be > 0 but tensor format is " +
                             featureInput.getDescriptorString();
                throw nb::value_error(msg.c_str());
            }
            if (numOutputChannels == 0) {
                throw nb::value_error("Convolution3d instance: num_output_channels must be > 0.");
            }
            if (groups == 0 || C % groups != 0 || numOutputChannels % groups != 0)
                throw nb::value_error("Convolution3d instance: groups must divide both input and output channels.");
            if (computeDataType != DataType::FP32 && computeDataType != DataType::TF32)
                throw nb::value_error("Convolution3d instance: compute_data_type must be thor.DataType.fp32 or thor.DataType.tf32.");
            if (computeDataType == DataType::TF32 && featureInput.getDataType() != DataType::FP32)
                throw nb::value_error("Convolution3d instance: TF32 compute requires FP32 input/weights/output storage.");
            if (filterDepth == 0 || filterHeight == 0 || filterWidth == 0) {
                throw nb::value_error("Convolution3d instance: filter_depth, filter_height, and filter_width must be >= 1.");
            }
            if (depthStride == 0 || verticalStride == 0 || horizontalStride == 0) {
                throw nb::value_error("Convolution3d instance: depth_stride, vertical_stride, and horizontal_stride must be >= 1.");
            }

            const auto [dilationD, dilationH, dilationW] = dilationFromPython(dilation);
            const PythonPaddingSpec paddingSpec =
                paddingFromPython(padding, depthPadding, verticalPadding, horizontalPadding);

            const uint64_t effectiveFilterD = uint64_t(dilationD) * (uint64_t(filterDepth) - 1ULL) + 1ULL;
            const uint64_t effectiveFilterH = uint64_t(dilationH) * (uint64_t(filterHeight) - 1ULL) + 1ULL;
            const uint64_t effectiveFilterW = uint64_t(dilationW) * (uint64_t(filterWidth) - 1ULL) + 1ULL;
            if (paddingSpec.mode != Convolution3dPaddingMode::SAME_UPPER) {
                const uint64_t paddedD = D + uint64_t(paddingSpec.explicitPadding[0]) + paddingSpec.explicitPadding[1];
                const uint64_t paddedH = H + uint64_t(paddingSpec.explicitPadding[2]) + paddingSpec.explicitPadding[3];
                const uint64_t paddedW = W + uint64_t(paddingSpec.explicitPadding[4]) + paddingSpec.explicitPadding[5];
                if (effectiveFilterD > paddedD) {
                    string msg = "Convolution3d instance: filter_depth effective size " + to_string(effectiveFilterD) +
                                 " is larger than padded input depth " + to_string(paddedD) +
                                 ". Input tensor is " + featureInput.getDescriptorString();
                    throw nb::value_error(msg.c_str());
                }
                if (effectiveFilterH > paddedH) {
                    string msg = "Convolution3d instance: filter_height effective size " + to_string(effectiveFilterH) +
                                 " is larger than padded input height " + to_string(paddedH) +
                                 ". Input tensor is " + featureInput.getDescriptorString();
                    throw nb::value_error(msg.c_str());
                }
                if (effectiveFilterW > paddedW) {
                    string msg = "Convolution3d instance: filter_width effective size " + to_string(effectiveFilterW) +
                                 " is larger than padded input width " + to_string(paddedW) +
                                 ". Input tensor is " + featureInput.getDescriptorString();
                    throw nb::value_error(msg.c_str());
                }
            }

            Convolution3d::Builder builder;
            builder.network(network)
                .featureInput(featureInput)
                .numOutputChannels(numOutputChannels)
                .filterDepth(filterDepth)
                .filterHeight(filterHeight)
                .filterWidth(filterWidth)
                .depthStride(depthStride)
                .verticalStride(verticalStride)
                .horizontalStride(horizontalStride)
                .depthDilation(dilationD)
                .verticalDilation(dilationH)
                .horizontalDilation(dilationW)
                .groups(groups)
                .computeDataType(computeDataType)
                .hasBias(hasBias);
            switch (paddingSpec.mode) {
                case Convolution3dPaddingMode::VALID:
                    builder.validPadding();
                    break;
                case Convolution3dPaddingMode::SAME_UPPER:
                    builder.samePadding();
                    break;
                case Convolution3dPaddingMode::EXPLICIT:
                    builder.padding(paddingSpec.explicitPadding[0],
                                    paddingSpec.explicitPadding[1],
                                    paddingSpec.explicitPadding[2],
                                    paddingSpec.explicitPadding[3],
                                    paddingSpec.explicitPadding[4],
                                    paddingSpec.explicitPadding[5]);
                    break;
            }

            applyPythonActivation(builder, activation);
            applyPythonEpilogueInputs(builder, epilogue_inputs);
            applyPythonEpilogue(builder, epilogue);

            if (weights_initializer != nullptr)
                builder.weightsInitializer(weights_initializer);
            if (biases_initializer != nullptr)
                builder.biasInitializer(biases_initializer);

            Convolution3d built = builder.build();
            new (self) Convolution3d(std::move(built));
        },
        "network"_a,
        "feature_input"_a,
        "num_output_channels"_a,
        "filter_depth"_a,
        "filter_height"_a,
        "filter_width"_a,
        "depth_stride"_a = 1,
        "vertical_stride"_a = 1,
        "horizontal_stride"_a = 1,
        "depth_padding"_a = 0,
        "vertical_padding"_a = 0,
        "horizontal_padding"_a = 0,
        "has_bias"_a = true,
        "activation"_a.none() = nb::str(DEFAULT_ACTIVATION_SENTINEL),
        "weights_initializer"_a = nb::none(),
        "biases_initializer"_a = nb::none(),
        "epilogue"_a.none() = nb::none(),
        "epilogue_inputs"_a.none() = nb::none(),
        "groups"_a = 1,
        "compute_data_type"_a = DataType::FP32,
        "padding"_a = nb::str("valid"),
        "dilation"_a = nb::int_(1));

    convolution_3d.def_static(
        "epilogue_input",
        &makePythonEpilogueInput,
        "output_dtype"_a.none() = nb::none(),
        "compute_dtype"_a.none() = nb::none(),
        R"nbdoc(
            Return the single tensor input expression expected by a Convolution3d epilogue.
            )nbdoc");

    convolution_3d.def_static(
        "epilogue_aux_input",
        &makePythonEpilogueAuxInput,
        "name"_a,
        "output_dtype"_a.none() = nb::none(),
        "compute_dtype"_a.none() = nb::none(),
        R"nbdoc(
            Return a named auxiliary tensor input expression for a Convolution3d epilogue.
            Bind the same name to a tensor with the ``epilogue_inputs`` constructor argument.
            )nbdoc");

    convolution_3d.def(
        "get_feature_output",
        [](Convolution3d &self) -> Tensor {
            std::optional<Tensor> maybeFeatureOutput = self.getFeatureOutput();
            return maybeFeatureOutput.value();
        },
        R"nbdoc(
            Return the output tensor produced by this layer.
            )nbdoc");

    convolution_3d.def("get_compute_data_type", &Convolution3d::getComputeDataType);

    convolution_3d.attr("__doc__") = R"nbdoc(
        3D convolution layer.

        Builds a trainable 3D convolutional layer with optional activation.
        Omitted activation defaults to ``thor.activations.Gelu()``; pass
        ``None`` to keep the layer linear.
        The API tensor layout is CDHW; the physical implementation adds the
        batch dimension and uses NCDHW. ``padding`` accepts ``"valid"``,
        ``"same"``/``"same_upper"`` (SAME_UPPER), or explicit
        ``(front, back, top, bottom, left, right)`` padding. ``dilation`` accepts
        either one positive integer or ``(depth, height, width)``. The legacy
        ``depth_padding``, ``vertical_padding``, and ``horizontal_padding``
        arguments remain compatibility aliases for symmetric explicit padding
        when ``padding`` is omitted. ``groups`` partitions input and output
        channels using standard grouped-convolution semantics. Activations are stitched into the
        expression before the implementation CustomLayer is constructed.
        ``epilogue`` may be a ``thor.physical.Expression`` built from
        ``Convolution3d.epilogue_input()`` and is applied after activation.
        ``compute_data_type=thor.DataType.fp32`` requests strict FP32 convolution
        math. ``thor.DataType.tf32`` explicitly permits TensorFloat-32 execution
        for FP32 input, weight, and output storage.
        )nbdoc";
}
