#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>

#include <memory>
#include <optional>

#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Api/Layers/Learning/Convolution3d.h"
#include "DeepLearning/Api/Layers/Learning/TrainableLayer.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

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
           shared_ptr<Activation> activation,
           shared_ptr<Initializer> weights_initializer,
           shared_ptr<Initializer> biases_initializer) {
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
            if (filterDepth == 0 || filterHeight == 0 || filterWidth == 0) {
                throw nb::value_error("Convolution3d instance: filter_depth, filter_height, and filter_width must be >= 1.");
            }
            if (depthStride == 0 || verticalStride == 0 || horizontalStride == 0) {
                throw nb::value_error("Convolution3d instance: depth_stride, vertical_stride, and horizontal_stride must be >= 1.");
            }

            const uint64_t effD = D + 2ULL * uint64_t(depthPadding);
            const uint64_t effH = H + 2ULL * uint64_t(verticalPadding);
            const uint64_t effW = W + 2ULL * uint64_t(horizontalPadding);
            if (uint64_t(filterDepth) > effD || uint64_t(filterHeight) > effH || uint64_t(filterWidth) > effW) {
                string msg = "Convolution3d instance: filter is larger than padded input. Input tensor is " +
                             featureInput.getDescriptorString();
                throw nb::value_error(msg.c_str());
            }

            Convolution3d::Builder builder;
            builder.network(network)
                .featureInput(featureInput)
                .numOutputChannels(numOutputChannels)
                .filterDepth(filterDepth)
                .filterHeight(filterHeight)
                .filterWidth(filterWidth)
                .depthPadding(depthPadding)
                .verticalPadding(verticalPadding)
                .horizontalPadding(horizontalPadding)
                .depthStride(depthStride)
                .verticalStride(verticalStride)
                .horizontalStride(horizontalStride)
                .hasBias(hasBias);

            if (activation == nullptr) {
                builder.noActivation();
            } else {
                builder.activation(activation);
            }

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
        "activation"_a = nb::none(),
        "weights_initializer"_a = nb::none(),
        "biases_initializer"_a = nb::none());

    convolution_3d.def(
        "get_feature_output",
        [](Convolution3d &self) -> Tensor {
            Optional<Tensor> maybeFeatureOutput = self.getFeatureOutput();
            return maybeFeatureOutput.get();
        },
        R"nbdoc(
            Return the output tensor produced by this layer.
            )nbdoc");

    convolution_3d.attr("__doc__") = R"nbdoc(
        3D convolution layer.

        Builds a trainable 3D convolutional layer with optional activation.
        The API tensor layout is CDHW; the physical implementation adds the
        batch dimension and uses NCDHW. Activations are stitched into the
        expression before the implementation CustomLayer is constructed.
        )nbdoc";
}
