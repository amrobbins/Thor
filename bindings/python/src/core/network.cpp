#include <nanobind/nanobind.h>

#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <memory>
#include <utility>

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

namespace {

nb::dict cudaKernelSourceInspectionToPython(const ThorImplementation::CudaKernelSourceInspection& info) {
    nb::dict entry;
    entry["name"] = info.name;
    entry["entrypoint"] = info.entrypoint;
    entry["source"] = info.source;
    entry["compiled_source"] = info.compiled_source;
    entry["compiled_source_hash"] = info.compiled_source_hash;
    entry["loaded_source_compilation_allowed"] = info.loaded_source_compilation_allowed;
    entry["source_encrypted"] = info.source_encrypted;
    if (!info.source_encryption_algorithm.empty()) {
        entry["source_encryption_algorithm"] = info.source_encryption_algorithm;
    }
    if (!info.source_decryption_key_fingerprint.empty()) {
        entry["source_decryption_key_fingerprint"] = info.source_decryption_key_fingerprint;
    }
    if (!info.signature_algorithm.empty()) {
        entry["signature_algorithm"] = info.signature_algorithm;
    }
    if (!info.signing_public_key_fingerprint.empty()) {
        entry["signing_public_key_fingerprint"] = info.signing_public_key_fingerprint;
    }
    if (!info.signature.empty()) {
        entry["signature"] = info.signature;
    }
    return entry;
}

nb::list cudaKernelSourceInspectionListToPython(const std::vector<ThorImplementation::CudaKernelSourceInspection>& infos) {
    nb::list result;
    for (const ThorImplementation::CudaKernelSourceInspection& info : infos) {
        result.append(cudaKernelSourceInspectionToPython(info));
    }
    return result;
}

nb::list cudaKernelOutOfBandKeysToPython(const std::vector<ThorImplementation::CudaKernelOutOfBandKeys>& key_sets) {
    nb::list result;
    for (const ThorImplementation::CudaKernelOutOfBandKeys& keys : key_sets) {
        nb::dict entry;
        entry["signing_public_key"] = keys.signing_public_key;
        entry["source_decryption_key"] = keys.source_decryption_key;
        result.append(std::move(entry));
    }
    return result;
}

}  // namespace

void bind_network(nb::module_ &m) {
    auto network = nb::class_<Network>(m, "Network");
    network.attr("__module__") = "thor";

    auto network_status_code_type = nb::enum_<Network::StatusCode>(m, "StatusCode")
                                        .value("success", Network::StatusCode::SUCCESS)
                                        .value("floating_input", Network::StatusCode::FLOATING_INPUT)
                                        .value("dangling_output", Network::StatusCode::DANGLING_OUTPUT)
                                        .value("gpu_out_of_memory", Network::StatusCode::GPU_OUT_OF_MEMORY)
                                        .value("duplicate_named_network_input", Network::StatusCode::DUPLICATE_NAMED_NETWORK_INPUT)
                                        .value("duplicate_named_network_output", Network::StatusCode::DUPLICATE_NAMED_NETWORK_OUTPUT)
                                        .value("deadlock_cycle", Network::StatusCode::DEADLOCK_CYCLE);
    network_status_code_type.attr("__qualname__") = "Network.StatusCode";
    network.attr("StatusCode") = network_status_code_type;

    network.def_static(
        "__new__",
        [](nb::handle cls, std::string name) -> std::shared_ptr<Network> {
            (void)cls;
            return std::make_shared<Network>(std::move(name));
        },
        "cls"_a,
        "name"_a,
        R"nbdoc(
A Network that contains layers. FIXME.
)nbdoc");
    network.def("__init__", [](Network*, std::string) {}, "name"_a);

    network.def("get_network_name", &Network::getNetworkName);
    network.def("get_num_trainable_layers", &Network::getNumTrainableLayers);
    network.def("status_code_to_string", &Network::statusCodeToString, "status_code"_a);

    network.def("get_architecture_json", &Network::architectureJsonString);
    network.def("save", nb::overload_cast<const std::string &, bool>(&Network::save), "directory"_a, "overwrite"_a = false);
    network.def(
        "_load_in_place",
        [](Network& self,
           const std::string& directory,
           bool allow_unsafe_loaded_cuda_kernel_source,
           const std::string& trusted_cuda_kernel_public_key,
           const std::string& trusted_cuda_kernel_source_decryption_key) {
            self.load(directory,
                      allow_unsafe_loaded_cuda_kernel_source,
                      trusted_cuda_kernel_public_key,
                      trusted_cuda_kernel_source_decryption_key);
        },
        "directory"_a,
        "allow_unsafe_loaded_cuda_kernel_source"_a = false,
        "trusted_cuda_kernel_public_key"_a = "",
        "trusted_cuda_kernel_source_decryption_key"_a = "",
        R"nbdoc(
Load a saved Thor network into this instance.

CudaKernelExpression CUDA source saved by current Thor versions is encrypted in
the model JSON. Loading such a model requires the out-of-band Ed25519 public
signing key and AES-256-GCM source decryption key printed when the model was
saved. Setting ``allow_unsafe_loaded_cuda_kernel_source=True`` additionally
allows the decrypted source to compile/run after signature verification.
)nbdoc");

    network.def_static(
        "_load_from_path",
        [](const std::string& directory,
           const std::string& network_name,
           bool allow_unsafe_loaded_cuda_kernel_source,
           const std::string& trusted_cuda_kernel_public_key,
           const std::string& trusted_cuda_kernel_source_decryption_key) {
            if (network_name.empty()) {
                throw nb::value_error("network_name must be non-empty when loading a Thor Network artifact");
            }
            auto loaded = std::make_shared<Network>(network_name);
            loaded->load(directory,
                         allow_unsafe_loaded_cuda_kernel_source,
                         trusted_cuda_kernel_public_key,
                         trusted_cuda_kernel_source_decryption_key);
            return loaded;
        },
        "directory"_a,
        "network_name"_a,
        "allow_unsafe_loaded_cuda_kernel_source"_a = false,
        "trusted_cuda_kernel_public_key"_a = "",
        "trusted_cuda_kernel_source_decryption_key"_a = "",
        R"nbdoc(
Load and return a saved Thor network with the explicit archive/network name.
)nbdoc");

    network.def("cuda_kernel_source_info", [](const Network& self) { return cudaKernelSourceInspectionListToPython(self.cudaKernelSourceInfo()); });
    network.def("cuda_kernel_sources", &Network::cudaKernelSources);
    network.def("cuda_kernel_source_info_json", &Network::cudaKernelSourceInfoJsonString);
    network.def("has_cuda_kernel_expressions", &Network::hasCudaKernelExpressions);
    network.def("capture_cuda_kernel_save_keys_to_file",
                &Network::captureCudaKernelSaveKeysToFile,
                "path"_a,
                "overwrite"_a = false,
                R"nbdoc(
Configure a required out-of-band key capture file for models containing
CudaKernelExpression CUDA source. Training placement refuses to proceed for
such networks until this is configured. The file is created immediately with a
pending marker and overwritten with the final save-time keys when save() runs.
)nbdoc");
    network.def("clear_cuda_kernel_save_key_capture", &Network::clearCudaKernelSaveKeyCapture);
    network.def("cuda_kernel_save_key_capture_configured", &Network::cudaKernelSaveKeyCaptureConfigured);
    network.def("cuda_kernel_signing_public_keys", &Network::cudaKernelSigningPublicKeys);
    network.def("cuda_kernel_out_of_band_keys", [](const Network& self) { return cudaKernelOutOfBandKeysToPython(self.cudaKernelOutOfBandKeys()); });

    network.def("get_default_optimizer", &Network::getDefaultOptimizer);
    network.def("freeze_training", &Network::freezeTraining);
    network.def("unfreeze_training", &Network::unfreezeTraining);
    network.def("get_trainable_parameter_references", &Network::getTrainableParameterReferences, "training_enabled_only"_a = true);

    network.def(
        "place",
        [](Network &self,
           uint32_t batch_size,
           bool inference_only,
           std::vector<int32_t> forced_devices,
           uint32_t forced_num_stamps_per_gpu,
           bool network_outputs_on_gpu) {
            nb::gil_scoped_release release;
            std::vector<Event> init_done_events;
            shared_ptr<PlacedNetwork> placedNetwork =
                self.place(batch_size, init_done_events, inference_only, forced_devices, forced_num_stamps_per_gpu, network_outputs_on_gpu);
            return placedNetwork;
        },
        "batch_size"_a,
        "inference_only"_a = false,
        "forced_devices"_a = std::vector<int32_t>{},
        "forced_num_stamps_per_gpu"_a = 0,
        "network_outputs_on_gpu"_a = false,
        R"nbdoc(
Place / compile the network for execution.

Parameters
----------
batch_size : int
inference_only : bool, default False
forced_devices : list[int], default []
    Device ids to force placement onto. Use Network.CPU for CPU.
forced_num_stamps_per_gpu : int, default 0
network_outputs_on_gpu : bool, default False
    Stamp NetworkOutput layers to GPU instead of CPU. When the producer tensor is
    already on that GPU, NetworkOutput aliases the producer instead of copying, so
    ensemble runtime can aggregate member outputs on device before one final
    materialization copy.

Returns
-------
thor.Network.StatusCode
)nbdoc");
}
