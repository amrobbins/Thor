#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

#include "DeepLearning/Api/Tensor/Tensor.h"

using DataType = ThorImplementation::TensorDescriptor::DataType;

// Forward declarations for per-feature binders
void bind_version(nb::module_ &thor);
void bind_network(nb::module_ &thor);
void bind_placed_network(nb::module_ &thor);
void bind_tensor(nb::module_ &thor);

void bind_activations(nb::module_ &activations);
void bind_initializers(nb::module_ &initializers);
void bind_layers(nb::module_ &layers);
void bind_losses(nb::module_ &losses);
void bind_metrics(nb::module_ &metrics);
void bind_optimizers(nb::module_ &optimizers);
void bind_physical(nb::module_ &physical);
void bind_random(nb::module_ &random);

NB_MODULE(_thor, thor) {
    thor.doc() = "Thor Python bindings";

    bind_version(thor);

    auto dt = nb::enum_<DataType>(thor, "DataType")
                  .value("packed_bool", DataType::PACKED_BOOLEAN)
                  .value("bool", DataType::BOOLEAN)
                  .value("int8", DataType::INT8)
                  .value("uint8", DataType::UINT8)
                  .value("int16", DataType::INT16)
                  .value("uint16", DataType::UINT16)
                  .value("int32", DataType::INT32)
                  .value("uint32", DataType::UINT32)
                  .value("int64", DataType::INT64)
                  .value("uint64", DataType::UINT64)
                  // .value("fp8_e4m3", DataType::FP8_E4M3)
                  // .value("fp8_e5m2", DataType::FP8_E5M2)
                  // .value("bf16", DataType::BF16)
                  .value("fp16", DataType::FP16)
                  .value("fp32", DataType::FP32)
                  .value("fp64", DataType::FP64)
                  .value("bf16", DataType::BF16)
                  .value("fp8_e4m3", DataType::FP8_E4M3)
                  .value("fp8_e5m2", DataType::FP8_E5M2);
    dt.attr("__module__") = "thor";

    bind_tensor(thor);
    bind_network(thor);
    bind_placed_network(thor);

    auto activations = thor.def_submodule("activations");
    bind_activations(activations);

    auto initializers = thor.def_submodule("initializers");
    bind_initializers(initializers);

    auto layers = thor.def_submodule("layers");
    bind_layers(layers);

    auto losses = thor.def_submodule("losses");
    bind_losses(losses);

    auto metrics = thor.def_submodule("metrics");
    bind_metrics(metrics);

    auto optimizers = thor.def_submodule("optimizers");
    bind_optimizers(optimizers);

    auto physical = thor.def_submodule("physical");
    bind_physical(physical);

    auto random = thor.def_submodule("random");
    bind_random(random);
}
