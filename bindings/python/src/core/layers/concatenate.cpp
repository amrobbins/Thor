#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Utility/Concatenate.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

#include "core/binding_types.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

void bind_concatenate(nb::module_ &m) {
    auto concatenate = nb::class_<Concatenate, Layer>(m, "Concatenate");
    concatenate.attr("__module__") = "thor.layers";

    concatenate.def(
        "__init__",
        [](Concatenate *self, Network &network, TensorList feature_inputs, uint32_t concatenation_axis) {
            if (feature_inputs.size() == 0) {
                throw nb::value_error("Concatenate instance: feature_inputs must be a non-empty list of thor.Tensor.");
            }

            // Use first tensor as reference
            nb::handle h0 = feature_inputs[0];
            Tensor &t0 = nb::cast<Tensor &>(h0);

            const auto &ref_dims = t0.getDimensions();
            const size_t ref_rank = ref_dims.size();
            const auto ref_dtype = t0.getDataType();

            if (ref_rank == 0) {
                string msg = "Concatenate instance: tensors must have at least 1 dimension. First tensor is " + t0.getDescriptorString();
                throw nb::value_error(msg.c_str());
            }

            if (concatenation_axis >= ref_rank) {
                string msg = "Concatenate instance: concatenation_axis " + to_string(concatenation_axis) +
                             " is out of range for tensor rank " + to_string(ref_rank) + ". First tensor is " + t0.getDescriptorString();
                throw nb::value_error(msg.c_str());
            }

            // Validate remaining tensors match requirements
            for (size_t i = 0; i < feature_inputs.size(); ++i) {
                Tensor &ti = nb::cast<Tensor &>(feature_inputs[i]);

                const auto &dims = ti.getDimensions();
                const size_t rank = dims.size();

                if (rank != ref_rank) {
                    string msg =
                        "Concatenate instance: all tensors must have the same number of dimensions. "
                        "First tensor rank " +
                        to_string(ref_rank) + " (" + t0.getDescriptorString() +
                        "), "
                        "tensor[" +
                        to_string(i) + "] rank " + to_string(rank) + " (" + ti.getDescriptorString() + ").";
                    throw nb::value_error(msg.c_str());
                }

                if (ti.getDataType() != ref_dtype) {
                    string msg =
                        "Concatenate instance: all tensors must have the same data type. "
                        "First tensor: " +
                        string(t0.getDescriptorString()) + "tensor[" + to_string(i) + "]: " + string(ti.getDescriptorString());
                    throw nb::value_error(msg.c_str());
                }

                for (size_t d = 0; d < ref_rank; ++d) {
                    if (d == concatenation_axis)
                        continue;
                    if (dims[d] != ref_dims[d]) {
                        string msg = "Concatenate instance: all tensor dimensions must match except along concatenation_axis " +
                                     to_string(concatenation_axis) + ". Mismatch at dim " + to_string(d) + ": first tensor has " +
                                     to_string(ref_dims[d]) + ", tensor[" + to_string(i) + "] has " + to_string(dims[d]) + ". tensor[" +
                                     to_string(i) + "] is " + ti.getDescriptorString();
                        throw nb::value_error(msg.c_str());
                    }
                }
            }

            Concatenate::Builder builder;
            builder.network(network).concatenationAxis(concatenation_axis);
            // Iterate Python list and cast each element to Tensor&
            for (nb::handle h : feature_inputs) {
                Tensor &t = nb::cast<Tensor &>(h);
                builder.featureInput(t);
            }

            // Move the concatenate layer into the pre-allocated but uninitialized memory at self
            new (self) Concatenate(std::move(builder.build()));
        },
        "network"_a,
        "feature_inputs"_a,
        "concatenation_axis"_a,

        R"nbdoc(
            Create and attach a Concatenate layer to a Network.

            Parameters
            ----------
            network : thor.Network
                Network the layer should be added to.
            feature_inputs : list[thor.Tensor]
                List of input feature tensors for this layer.
            concatenation_axis : int
                Axis along which to concatenate the input tensors.

            For example, if your input tensors have dimensions:
                1. [2, 4, 5, 7]
                2. [2, 6, 5, 7]
                3. [2, 2, 5, 7]

            with concatenation_axis=1, then your output tensor will have dimensions:
                [2, 12, 5, 7]

            Note that all dimensions must match to perform Contcatenate, except for the concatenation_axis dimension.

            )nbdoc");
}
