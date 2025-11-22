#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Utility/Pooling.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

#include "bindings/python/src/core/binding_types.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

void bind_pooling(nb::module_ &m) {
    auto pooling_class = nb::class_<Pooling, Layer>(m, "Pooling")
                             .def(
                                 "__init__",
                                 [](Pooling *self,
                                    Network &network,
                                    const Pooling::Type &type,
                                    uint32_t window_height,
                                    uint32_t window_width,
                                    uint32_t vertical_stride,
                                    uint32_t horizontal_stride,
                                    uint32_t vertical_padding,
                                    uint32_t horizontal_padding,
                                    bool same_padding,
                                    bool vertical_same_padding,
                                    bool horizontal_same_padding) {
                                     Pooling::Builder builder = builder.network(network)
                                                                    .type(type)
                                                                    .windowHeight(window_height)
                                                                    .windowWidth(window_width)
                                                                    .verticalStride(vertical_stride)
                                                                    .horizontalStride(horizontal_stride);

                                     // These can't all be specified at the same time, but logic to enforce that
                                     // is on the implementation side, not the binding side.
                                     if (vertical_padding != 0)
                                         builder.verticalPadding(vertical_padding);
                                     if (horizontal_padding != 0)
                                         builder.horizontalPadding(horizontal_padding);
                                     if (same_padding)
                                         builder.samePadding();
                                     if (vertical_same_padding != 0)
                                         builder.verticalSamePadding();
                                     if (horizontal_same_padding != 0)
                                         builder.horizontalSamePadding();

                                     Pooling built = builder.build();

                                     // Move the pooling layer into the pre-allocated but uninitialized memory at self
                                     new (self) Pooling(std::move(built));
                                 },
                                 "network"_a,
                                 "type"_a,
                                 "window_height"_a,
                                 "window_width"_a,
                                 "vertical_stride"_a = 1,
                                 "horizontal_stride"_a = 1,
                                 "vertical_padding"_a = 0,
                                 "horizontal_padding"_a = 0,
                                 "same_padding"_a = false,
                                 "vertical_same_padding"_a = false,
                                 "horizontal_same_padding"_a = false,

                                 nb::sig("def __init__(self, "
                                         "network: thor.Network, "
                                         "type: thor.Pooling.Type, "
                                         "window_height: int, "
                                         "window_width: int, "
                                         "vertical_stride: int = 1, "
                                         "horizontal_stride: int = 1, "
                                         "vertical_padding: int = 0, "
                                         "horizontal_padding: int = 0, "
                                         "same_padding: bool = False, "
                                         "vertical_same_padding: bool = False, "
                                         "horizontal_same_padding: bool = False "
                                         ") -> None"),

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
        same_padding: bool, optional
            Compute and apply the amount of padding required for the output height
            and width to match the input height and width. Note that the
            implementation applies the number of padding elements specified to both
            sides of the block, so the amount of padding applied is always even.
            So, when the amount of padding required for same_padding is odd, it is
            not possible to do here and an exception will be raised if same_padding
            is requested.
        vertical_same_padding: bool, optional
            Compute and apply same padding in the vertical direction.
            Note that the limitation described in for same_padding applies here as well.
        horizontal_same_padding: bool, optional
            Compute and apply same padding in the vertical direction.
            Note that the limitation described in for same_padding applies here as well.


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

    nb::enum_<Pooling::Type>(pooling_class, "Type")
        .value("AVERAGE", Pooling::Type::AVERAGE)
        .value("MAX", Pooling::Type::MAX)
        .export_values();
}
