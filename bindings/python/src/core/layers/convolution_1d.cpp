#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <memory>
#include <optional>
#include <string>

#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Api/Layers/Learning/Convolution1d.h"
#include "DeepLearning/Api/Layers/Learning/TrainableLayer.h"
#include "DeepLearning/Api/Network/Network.h"
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
    if (!nb::isinstance<nb::str>(activation))
        return false;
    return pybind::castOrTypeError<string>(
               activation, "Convolution1d() argument 'activation'", "thor.activations.Activation, str sentinel, or None", false) ==
           DEFAULT_ACTIVATION_SENTINEL;
}

void applyPythonActivation(Convolution1d::Builder &builder, const nb::object &activation) {
    if (isDefaultActivationSentinel(activation))
        return;
    if (activation.is_none()) {
        builder.noActivation();
        return;
    }
    shared_ptr<Activation> activationPtr = pybind::castArgument<shared_ptr<Activation>>(
        activation, "Convolution1d", "activation", "thor.activations.Activation or None", false);
    if (activationPtr == nullptr)
        builder.noActivation();
    else
        builder.activation(activationPtr);
}

optional<DataType> optionalDataTypeFromPython(const nb::object &obj, const char *functionName, const char *argumentName) {
    if (obj.is_none())
        return nullopt;
    return pybind::castArgument<DataType>(obj, functionName, argumentName, "thor.DataType or None", false);
}

ThorImplementation::Expression makePythonEpilogueInput(const nb::object &outputDTypeObj, const nb::object &computeDTypeObj) {
    return Convolution1d::epilogueInput(optionalDataTypeFromPython(computeDTypeObj, "Convolution1d.epilogue_input", "compute_dtype"),
                                        optionalDataTypeFromPython(outputDTypeObj, "Convolution1d.epilogue_input", "output_dtype"));
}

ThorImplementation::Expression makePythonEpilogueAuxInput(const string &inputName,
                                                          const nb::object &outputDTypeObj,
                                                          const nb::object &computeDTypeObj) {
    return Convolution1d::epilogueAuxInput(
        inputName,
        optionalDataTypeFromPython(computeDTypeObj, "Convolution1d.epilogue_aux_input", "compute_dtype"),
        optionalDataTypeFromPython(outputDTypeObj, "Convolution1d.epilogue_aux_input", "output_dtype"));
}

void applyPythonEpilogueInputs(Convolution1d::Builder &builder, const nb::object &epilogueInputs) {
    if (epilogueInputs.is_none())
        return;
    nb::dict inputsDict = pybind::castOrTypeError<nb::dict>(
        epilogueInputs, "Convolution1d() argument 'epilogue_inputs'", "dict[str, thor.Tensor] or None", false);
    size_t index = 0;
    for (auto item : inputsDict) {
        const string keyContext = "Convolution1d() argument 'epilogue_inputs' key[" + to_string(index) + "]";
        string name = pybind::castOrTypeError<string>(item.first, keyContext, "str", false);
        const string valueContext = "Convolution1d() argument 'epilogue_inputs'[" + name + "]";
        Tensor tensor = pybind::castOrTypeError<Tensor>(item.second, valueContext, "thor.Tensor", false);
        builder.epilogueInput(name, tensor);
        ++index;
    }
}

void applyPythonEpilogue(Convolution1d::Builder &builder, const nb::object &epilogue) {
    if (!epilogue.is_none()) {
        builder.epilogue(pybind::castArgument<ThorImplementation::Expression>(
            epilogue, "Convolution1d", "epilogue", "thor.physical.Expression or None", false));
    }
}

struct PythonPaddingSpec {
    Convolution1dPaddingMode mode = Convolution1dPaddingMode::VALID;
    array<uint32_t, 2> explicitPadding = {0, 0};
};

PythonPaddingSpec paddingFromPython(const nb::object &padding) {
    if (nb::isinstance<nb::str>(padding)) {
        string mode = pybind::castOrTypeError<string>(
            padding, "Convolution1d() argument 'padding'", "'valid', 'same', 'causal', or length-2 sequence[int]", false);
        transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c) { return static_cast<char>(tolower(c)); });
        if (mode == "valid")
            return {Convolution1dPaddingMode::VALID, {0, 0}};
        if (mode == "same" || mode == "same_upper")
            return {Convolution1dPaddingMode::SAME_UPPER, {0, 0}};
        if (mode == "causal")
            return {Convolution1dPaddingMode::CAUSAL, {0, 0}};
        throw nb::value_error(
            "Convolution1d() argument 'padding' must be 'valid', 'same', 'causal', or a length-2 sequence[int].");
    }
    if (!nb::isinstance<nb::sequence>(padding))
        throw nb::type_error("Convolution1d() argument 'padding': expected string or length-2 sequence[int] (left, right).");
    nb::sequence seq = pybind::castOrTypeError<nb::sequence>(
        padding, "Convolution1d() argument 'padding'", "length-2 sequence[int]", false);
    if (nb::len(seq) != 2)
        throw nb::value_error("Convolution1d explicit padding must contain exactly two values ordered as (left, right).");
    return {Convolution1dPaddingMode::EXPLICIT,
            {pybind::castOrTypeError<uint32_t>(seq[0], "Convolution1d() argument 'padding'[0]", "non-negative int", false),
             pybind::castOrTypeError<uint32_t>(seq[1], "Convolution1d() argument 'padding'[1]", "non-negative int", false)}};
}
}  // namespace

void bind_convolution_1d(nb::module_ &m) {
    auto convolution_1d = nb::class_<Convolution1d, TrainableLayer>(m, "Convolution1d");
    convolution_1d.attr("__module__") = "thor.layers";

    convolution_1d.def(
        "__init__",
        [](Convolution1d *self,
           Network &network,
           nb::object featureInput,
           uint32_t numOutputChannels,
           uint32_t filterWidth,
           uint32_t stride,
           nb::object padding,
           bool hasBias,
           nb::object activation,
           shared_ptr<Initializer> weightsInitializer,
           shared_ptr<Initializer> biasesInitializer,
           nb::object epilogue,
           nb::object epilogueInputs,
           uint32_t dilation,
           uint32_t groups,
           DataType computeDataType) {
            const bool useRagged = nb::isinstance<RaggedTensor>(featureInput);
            const bool useDense = nb::isinstance<Tensor>(featureInput);
            if (!useRagged && !useDense)
                throw nb::type_error("Convolution1d feature_input must be thor.Tensor or thor.RaggedTensor.");

            uint64_t inputChannels = 0;
            optional<Tensor> denseFeatureInput;
            optional<RaggedTensor> raggedFeatureInput;
            if (useRagged) {
                raggedFeatureInput = nb::cast<RaggedTensor>(featureInput);
                const auto &trailingDims = raggedFeatureInput->getTrailingDimensions();
                if (trailingDims.size() != 1 || trailingDims.front() == 0)
                    throw nb::value_error(
                        "Convolution1d instance: ragged feature_input must have exactly one non-zero trailing channel dimension.");
                if (!raggedFeatureInput->hasMaxValuesPerRow())
                    throw nb::value_error("Convolution1d instance: ragged feature_input requires max_values_per_row.");
                inputChannels = trailingDims.front();
            } else {
                denseFeatureInput = nb::cast<Tensor>(featureInput);
                const auto &dims = denseFeatureInput->getDimensions();
                if (dims.size() != 2) {
                    const string msg =
                        "Convolution1d instance: feature_input must be a 2D CW tensor (no batch) but tensor format is " +
                        denseFeatureInput->getDescriptorString();
                    throw nb::value_error(msg.c_str());
                }
                if (dims[0] == 0 || dims[1] == 0)
                    throw nb::value_error("Convolution1d instance: feature_input dimensions must all be > 0.");
                inputChannels = dims[0];
            }

            if (numOutputChannels == 0)
                throw nb::value_error("Convolution1d instance: num_output_channels must be > 0.");
            if (filterWidth == 0)
                throw nb::value_error("Convolution1d instance: filter_width must be >= 1.");
            if (stride == 0)
                throw nb::value_error("Convolution1d instance: stride must be >= 1.");
            if (dilation == 0)
                throw nb::value_error("Convolution1d instance: dilation must be >= 1.");
            if (groups == 0 || inputChannels % groups != 0 || numOutputChannels % groups != 0)
                throw nb::value_error("Convolution1d instance: groups must divide both input and output channels.");
            if (computeDataType != DataType::FP32 && computeDataType != DataType::TF32)
                throw nb::value_error("Convolution1d instance: compute_data_type must be thor.DataType.fp32 or thor.DataType.tf32.");
            const DataType inputStorageDataType = useRagged ? raggedFeatureInput->getValuesDataType() : denseFeatureInput->getDataType();
            if (computeDataType == DataType::TF32 && inputStorageDataType != DataType::FP32)
                throw nb::value_error("Convolution1d instance: TF32 compute requires FP32 input/weights/output storage.");

            const PythonPaddingSpec paddingSpec = paddingFromPython(padding);
            if (useRagged) {
                if (stride != 1)
                    throw nb::value_error("Convolution1d instance: ragged feature_input requires stride=1.");
                if (paddingSpec.mode != Convolution1dPaddingMode::CAUSAL)
                    throw nb::value_error("Convolution1d instance: ragged feature_input requires padding='causal'.");
                if (!epilogue.is_none() || !epilogueInputs.is_none())
                    throw nb::value_error(
                        "Convolution1d instance: ragged feature_input does not support custom epilogues or epilogue_inputs.");
            } else {
                const auto &dims = denseFeatureInput->getDimensions();
                const uint64_t effectiveFilter = uint64_t(dilation) * (uint64_t(filterWidth) - 1ULL) + 1ULL;
                if (paddingSpec.mode != Convolution1dPaddingMode::SAME_UPPER &&
                    paddingSpec.mode != Convolution1dPaddingMode::CAUSAL) {
                    const uint64_t paddedWidth =
                        dims[1] + uint64_t(paddingSpec.explicitPadding[0]) + paddingSpec.explicitPadding[1];
                    if (effectiveFilter > paddedWidth) {
                        const string msg = "Convolution1d instance: effective filter width " + to_string(effectiveFilter) +
                                           " is larger than padded input width " + to_string(paddedWidth) + ".";
                        throw nb::value_error(msg.c_str());
                    }
                }
            }

            Convolution1d::Builder builder;
            builder.network(network);
            if (useRagged)
                builder.featureInput(raggedFeatureInput.value());
            else
                builder.featureInput(denseFeatureInput.value());
            builder.numOutputChannels(numOutputChannels)
                .filterWidth(filterWidth)
                .groups(groups)
                .stride(stride)
                .dilation(dilation)
                .computeDataType(computeDataType)
                .hasBias(hasBias);
            switch (paddingSpec.mode) {
                case Convolution1dPaddingMode::VALID:
                    builder.validPadding();
                    break;
                case Convolution1dPaddingMode::SAME_UPPER:
                    builder.samePadding();
                    break;
                case Convolution1dPaddingMode::CAUSAL:
                    builder.causalPadding();
                    break;
                case Convolution1dPaddingMode::EXPLICIT:
                    builder.padding(paddingSpec.explicitPadding[0], paddingSpec.explicitPadding[1]);
                    break;
            }
            applyPythonActivation(builder, activation);
            applyPythonEpilogueInputs(builder, epilogueInputs);
            applyPythonEpilogue(builder, epilogue);
            if (weightsInitializer != nullptr)
                builder.weightsInitializer(weightsInitializer);
            if (biasesInitializer != nullptr)
                builder.biasInitializer(biasesInitializer);

            Convolution1d built = builder.build();
            new (self) Convolution1d(std::move(built));
        },
        "network"_a,
        "feature_input"_a,
        "num_output_channels"_a,
        "filter_width"_a,
        "stride"_a = 1,
        "padding"_a = nb::str("valid"),
        "has_bias"_a = true,
        "activation"_a.none() = nb::str(DEFAULT_ACTIVATION_SENTINEL),
        "weights_initializer"_a = nb::none(),
        "biases_initializer"_a = nb::none(),
        "epilogue"_a.none() = nb::none(),
        "epilogue_inputs"_a.none() = nb::none(),
        "dilation"_a = 1,
        "groups"_a = 1,
        "compute_data_type"_a = DataType::FP32);

    convolution_1d.def_static("epilogue_input",
                              &makePythonEpilogueInput,
                              "output_dtype"_a.none() = nb::none(),
                              "compute_dtype"_a.none() = nb::none());
    convolution_1d.def_static("epilogue_aux_input",
                              &makePythonEpilogueAuxInput,
                              "name"_a,
                              "output_dtype"_a.none() = nb::none(),
                              "compute_dtype"_a.none() = nb::none());
    convolution_1d.def(
        "get_feature_output",
        [](Convolution1d &self) -> nb::object {
            if (std::optional<RaggedTensor> raggedOutput = self.getRaggedFeatureOutput(); raggedOutput.has_value())
                return nb::cast(raggedOutput.value());
            return nb::cast(self.getFeatureOutput().value());
        });
    convolution_1d.def("get_use_ragged", &Convolution1d::getUseRagged);
    convolution_1d.def("get_compute_data_type", &Convolution1d::getComputeDataType);

    convolution_1d.attr("__doc__") = R"nbdoc(
        Trainable 1D convolution over a dense CW ``thor.Tensor`` or a logical ``thor.RaggedTensor``.

        Dense inputs retain the existing padding surface: ``"valid"``, ``"same"``
        (SAME_UPPER), ``"causal"``, or an explicit ``(left, right)`` pair. Ragged
        inputs deliberately support only stride-1 causal convolution and preserve
        the exact input row partition while changing the trailing channel count.
        The ragged path lowers to Thor's qualified ragged causal Conv1D backend; it
        does not materialize a dense tensor or an im2col/unfold temporary.
        ``compute_data_type=thor.DataType.fp32`` requests strict FP32 convolution
        math. ``thor.DataType.tf32`` explicitly permits TensorFloat-32 execution
        for FP32 input, weight, and output storage.
        )nbdoc";
}
