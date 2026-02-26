#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Utility/Pooling.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

#include "core/binding_types.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

void bind_pooling(nb::module_ &layers) {
    auto pooling = nb::class_<Pooling, Layer>(layers, "Pooling");
    pooling.attr("__module__") = "thor.layers";

    auto pooling_type = nb::enum_<Pooling::Type>(pooling, "Type").value("average", Pooling::Type::AVERAGE).value("max", Pooling::Type::MAX);
    pooling_type.attr("__module__") = "thor.layers";
    pooling_type.attr("__qualname__") = "Pooling.Type";
    pooling.attr("Type") = pooling_type;

    pooling
        .def(
            "__init__",
            [](Pooling *self,
               Network &network,
               Tensor feature_input,
               Pooling::Type type,
               uint32_t windowHeight,
               uint32_t windowWidth,
               uint32_t verticalStride,
               uint32_t horizontalStride,
               uint32_t verticalPadding,
               uint32_t horizontalPadding) {
                const auto &dims = feature_input.getDimensions();
                const size_t rank = dims.size();

                // Expect NCHW without batch => [C, H, W]
                if (rank != 3) {
                    string msg =
                        "Pooling instance: feature_input must be a 3D NCHW tensor without batch "
                        "(expected dimensions [C, H, W]) but tensor format is " +
                        feature_input.getDescriptorString();
                    throw nb::value_error(msg.c_str());
                }

                const uint64_t C = dims[0];
                const uint64_t H = dims[1];
                const uint64_t W = dims[2];

                if (C == 0 || H == 0 || W == 0) {
                    string msg = "Pooling instance: feature_input dimensions must all be > 0 but tensor format is " +
                                 feature_input.getDescriptorString();
                    throw nb::value_error(msg.c_str());
                }

                if (windowHeight == 0 || windowWidth == 0) {
                    string msg =
                        "Pooling instance: window_height and window_width must be >= 1. "
                        "window_height=" +
                        to_string(windowHeight) + " window_width=" + to_string(windowWidth);
                    throw nb::value_error(msg.c_str());
                }

                if (verticalStride == 0 || horizontalStride == 0) {
                    string msg =
                        "Pooling instance: vertical_stride and horizontal_stride must be >= 1. "
                        "vertical_stride=" +
                        to_string(verticalStride) + " horizontal_stride=" + to_string(horizontalStride);
                    throw nb::value_error(msg.c_str());
                }

                // Validate pooling type
                if (type != Pooling::Type::AVERAGE && type != Pooling::Type::MAX) {
                    string msg = "Pooling instance: invalid pooling type value " + to_string((int)type);
                    throw nb::value_error(msg.c_str());
                }

                // Ensure window fits within padded input
                const uint64_t effH = H + 2ULL * uint64_t(verticalPadding);
                const uint64_t effW = W + 2ULL * uint64_t(horizontalPadding);

                if (uint64_t(windowHeight) > effH) {
                    string msg = "Pooling instance: window_height " + to_string(windowHeight) + " is larger than padded input height " +
                                 to_string(effH) + ". Input tensor is " + feature_input.getDescriptorString();
                    throw nb::value_error(msg.c_str());
                }

                if (uint64_t(windowWidth) > effW) {
                    string msg = "Pooling instance: window_width " + to_string(windowWidth) + " is larger than padded input width " +
                                 to_string(effW) + ". Input tensor is " + feature_input.getDescriptorString();
                    throw nb::value_error(msg.c_str());
                }

                Pooling::Builder builder;
                builder.network(network)
                    .featureInput(feature_input)
                    .type(type)
                    .windowHeight(windowHeight)
                    .windowWidth(windowWidth)
                    .verticalStride(verticalStride)
                    .horizontalStride(horizontalStride);

                if (verticalPadding != 0)
                    builder.verticalPadding(verticalPadding);
                if (horizontalPadding != 0)
                    builder.horizontalPadding(horizontalPadding);

                Pooling built = builder.build();

                // Move the pooling layer into the pre-allocated but uninitialized memory at self
                new (self) Pooling(std::move(built));
            },
            "network"_a,
            "feature_input"_a,
            "type"_a,
            "window_height"_a,
            "window_width"_a,
            "vertical_stride"_a = 1,
            "horizontal_stride"_a = 1,
            "vertical_padding"_a = 0,
            "horizontal_padding"_a = 0,

            R"nbdoc(
Pooling layer that downsamples its input by applying a pooling operation
(e.g. max or average) over sliding windows.

This layer supports different pooling types via :class:`thor.Pooling.Type`
(such as ``Pooling.Type.MAX`` or ``Pooling.Type.AVERAGE``), and allows
explicit control over window size, stride, and padding in both the
vertical and horizontal directions.

Parameters
----------
network : thor.Network
    The network to add this layer into.
type : thor.Pooling.Type
    The pooling mode to use (e.g. ``Pooling.Type.MAX`` or
    ``Pooling.Type.AVERAGE``).
window_height : int
    Height of the pooling window (in cells).
window_width : int
    Width of the pooling window (in cells).
vertical_stride : int, optional
    Vertical stride of the pooling window. Defaults to 1.
horizontal_stride : int, optional
    Horizontal stride of the pooling window. Defaults to 1.
vertical_padding : int, optional
    Amount of zero-padding added to the top and bottom of the input.
    This amount of padding is added to both the top and bottom of the input,
    so vertical_padding=2 creates 4 rows of padding total.
    Defaults to 0.
horizontal_padding : int, optional
    Amount of zero-padding added to the left and right of the input.
    This amount of padding is added to both the left and right of the input,
    so horizontal_padding=2 creates 4 columns of padding total.
    Defaults to 0.

Notes
-----
The supported tensor layout is NCHW.
)nbdoc")
        .def("get_output_dimensions", &Pooling::getOutputDimensions)
        .def("get_pooling_type", &Pooling::getPoolingType)
        .def("get_window_height", &Pooling::getWindowHeight)
        .def("get_window_width", &Pooling::getWindowWidth)
        .def("get_vertical_stride", &Pooling::getVerticalStride)
        .def("get_horizontal_stride", &Pooling::getHorizontalStride)
        .def("get_vertical_padding", &Pooling::getVerticalPadding)
        .def("get_horizontal_padding", &Pooling::getHorizontalPadding);
}
