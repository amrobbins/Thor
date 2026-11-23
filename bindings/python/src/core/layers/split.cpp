#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Utility/Split.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;
using namespace Thor;

void bind_split(nb::module_ &m) {
    auto split = nb::class_<Split, Layer>(m, "Split");
    split.attr("__module__") = "thor.layers";

    split.def(
        "__init__",
        [](Split *self,
           Network &network,
           nb::object feature_input,
           uint32_t axis,
           std::vector<uint64_t> sizes,
           std::optional<std::vector<ThorImplementation::DataType>> output_data_types) {
            Split::Builder builder;
            builder.network(network);
            if (nb::isinstance<RaggedTensor>(feature_input))
                builder.featureInput(nb::cast<RaggedTensor>(feature_input));
            else if (nb::isinstance<Tensor>(feature_input))
                builder.featureInput(nb::cast<Tensor>(feature_input));
            else
                throw nb::type_error("Split feature_input must be thor.Tensor or thor.RaggedTensor.");
            builder.axis(axis).sizes(std::move(sizes));
            if (output_data_types.has_value()) builder.outputDataTypes(std::move(output_data_types.value()));
            try {
                Split built = builder.build();
                new (self) Split(std::move(built));
            } catch (const std::logic_error &error) {
                // An argument the layer cannot take: sizes, axis or types.
                throw nb::value_error(error.what());
            }
        },
        "network"_a,
        "feature_input"_a,
        "axis"_a,
        "sizes"_a,
        "output_data_types"_a.none() = nb::none(),
        R"nbdoc(
            Split one tensor along an axis into consecutive pieces, in one kernel launch.

            Parameters
            ----------
            network : thor.Network
                Network the layer should be added to.
            feature_input : thor.Tensor or thor.RaggedTensor
                The tensor to split. A ragged tensor splits along a trailing value axis, and every piece is a ragged
                tensor sharing its row partition; only the active packed values are moved.
            axis : int
                Axis to split along; the batch dimension is not one of the axes (for a ragged tensor, the axes are its
                trailing dimensions).
            sizes : list[int]
                The pieces' extents on the axis, in order: two or more, each positive, summing to the input's extent.
            output_data_types : list[thor.DataType] or None, default None
                One data type per piece; each piece is converted to its type as it is split, in the same kernel (the
                gradients are converted back to the input's type). None gives every piece the input's type. A piece
                of an integer or boolean type takes no gradient, and a piece whose consumers send none contributes
                zeros to the input's gradient.

            For example, an input of dimensions [12, 5] split on axis 0 with sizes [4, 6, 2] gives pieces of
            dimensions [4, 5], [6, 5] and [2, 5]: rows 0-3, 4-9 and 10-11.
            )nbdoc");

    split.def(
        "get_feature_outputs",
        [](const Split &self) -> nb::list {
            nb::list pieces;
            if (self.getUseRagged()) {
                for (const RaggedTensor &piece : self.getRaggedFeatureOutputs()) pieces.append(nb::cast(piece));
            } else {
                for (const Tensor &piece : self.getFeatureOutputs()) pieces.append(nb::cast(piece));
            }
            return pieces;
        },
        R"nbdoc(Return the pieces, in order: thor.Tensor, or thor.RaggedTensor for a ragged input.)nbdoc");
    split.def_prop_ro("use_ragged", &Split::getUseRagged);
    split.def_prop_ro("axis", &Split::getAxis);
    split.def_prop_ro("sizes", &Split::getSizes);
}
