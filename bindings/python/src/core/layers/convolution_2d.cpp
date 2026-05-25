#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>

#include <memory>
#include <exception>
#include <optional>

#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Api/Layers/Learning/Convolution2d.h"
#include "DeepLearning/Api/Layers/Learning/TrainableLayer.h"
#include "DeepLearning/Api/Network/Network.h"
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

void applyPythonActivation(Convolution2d::Builder &builder, const nb::object &activation) {
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
    return Convolution2d::epilogueInput(computeDType, outputDType);
}

void applyPythonEpilogue(Convolution2d::Builder &builder, const nb::object &epilogue) {
    if (epilogue.is_none()) {
        return;
    }
    if (!nb::isinstance<ThorImplementation::Expression>(epilogue)) {
        throw nb::type_error("epilogue must be a thor.physical.Expression instance or None");
    }
    builder.epilogue(nb::cast<ThorImplementation::Expression>(epilogue));
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
           uint32_t verticalPadding,
           uint32_t horizontalPadding,
           bool hasBias,
           nb::object activation,
           shared_ptr<Initializer> weights_initializer,
           shared_ptr<Initializer> biases_initializer,
           nb::object epilogue) {
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

            // Ensure filter fits within padded input (otherwise output dims go <= 0)
            const uint64_t effH = H + 2ULL * uint64_t(verticalPadding);
            const uint64_t effW = W + 2ULL * uint64_t(horizontalPadding);

            if (uint64_t(filterHeight) > effH) {
                string msg = "Convolution2d instance: filter_height " + to_string(filterHeight) + " is larger than padded input height " +
                             to_string(effH) + ". Input tensor is " + featureInput.getDescriptorString();
                throw nb::value_error(msg.c_str());
            }
            if (uint64_t(filterWidth) > effW) {
                string msg = "Convolution2d instance: filter_width " + to_string(filterWidth) + " is larger than padded input width " +
                             to_string(effW) + ". Input tensor is " + featureInput.getDescriptorString();
                throw nb::value_error(msg.c_str());
            }

            Convolution2d::Builder builder;
            builder.network(network)
                .featureInput(featureInput)
                .numOutputChannels(numOutputChannels)
                .filterHeight(filterHeight)
                .filterWidth(filterWidth)
                .verticalPadding(verticalPadding)
                .horizontalPadding(horizontalPadding)
                .verticalStride(verticalStride)
                .horizontalStride(horizontalStride)
                .hasBias(hasBias);

            applyPythonActivation(builder, activation);
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
        "vertical_padding"_a = 0,
        "horizontal_padding"_a = 0,
        "has_bias"_a = true,
        "activation"_a.none() = nb::str(DEFAULT_ACTIVATION_SENTINEL),
        "weights_initializer"_a = nb::none(),
        "biases_initializer"_a = nb::none(),
        "epilogue"_a.none() = nb::none());

    convolution_2d.def_static(
        "epilogue_input",
        &makePythonEpilogueInput,
        "output_dtype"_a.none() = nb::none(),
        "compute_dtype"_a.none() = nb::none(),
        R"nbdoc(
            Return the single tensor input expression expected by a Convolution2d epilogue.
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
        vertical_padding : int, default 0
            Amount of explicit zero-padding to apply above and below the
            input.
        horizontal_padding : int, default 0
            Amount of explicit zero-padding to apply left and right of the
            input.
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
            Build it from ``Convolution2d.epilogue_input()``.
        )nbdoc";
}
