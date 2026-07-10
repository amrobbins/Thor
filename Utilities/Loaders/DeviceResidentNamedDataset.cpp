#include "Utilities/Loaders/DeviceResidentNamedDataset.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Common/Stream.h"

#include <chrono>
#include <stdexcept>
#include <utility>

using ThorImplementation::Tensor;
using ThorImplementation::TensorPlacement;

namespace {

bool shouldUploadField(const std::set<std::string> &names,
                       const Thor::DatasetField &field) {
    return names.empty() || names.find(field.name) != names.end();
}

}  // namespace

std::shared_ptr<DeviceResidentNamedDataset> DeviceResidentNamedDataset::fromSnapshot(
    const MaterializedNamedDatasetSnapshot &snapshot,
    TensorPlacement devicePlacement) {
    return fromSnapshot(snapshot, devicePlacement, std::set<std::string>{});
}

std::shared_ptr<DeviceResidentNamedDataset> DeviceResidentNamedDataset::fromSnapshot(
    const MaterializedNamedDatasetSnapshot &snapshot,
    TensorPlacement devicePlacement,
    const std::set<std::string> &tensorNamesToUpload) {
    THOR_THROW_IF_FALSE(devicePlacement.getMemDevice() == TensorPlacement::MemDevices::GPU);
    if (snapshot.numExamples == 0) {
        throw std::runtime_error("DeviceResidentNamedDataset requires at least one example.");
    }

    auto dataset = std::shared_ptr<DeviceResidentNamedDataset>(
        new DeviceResidentNamedDataset(
            snapshot.datasetId,
            snapshot.schema,
            snapshot.layout,
            snapshot.numExamples,
            devicePlacement));

    const auto start = std::chrono::steady_clock::now();
    Stream uploadStream(devicePlacement);
    for (const auto &entry : snapshot.fields) {
        const Thor::DatasetField &field = snapshot.schema.getField(entry.first);
        if (!shouldUploadField(tensorNamesToUpload, field)) {
            continue;
        }
        const Tensor &hostTensor = entry.second;
        Tensor deviceTensor(devicePlacement, hostTensor.getDescriptor());
        deviceTensor.copyFromAsync(hostTensor, uploadStream);
        dataset->fields.emplace(entry.first, std::move(deviceTensor));
    }

    if (!tensorNamesToUpload.empty()) {
        for (const std::string &name : tensorNamesToUpload) {
            if (!dataset->hasTensor(name)) {
                throw std::runtime_error(
                    "DeviceResidentNamedDataset requested unknown snapshot tensor '" + name + "'.");
            }
        }
    }

    uploadStream.synchronize();
    dataset->uploadSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    return dataset;
}

uint64_t DeviceResidentNamedDataset::totalBytes() const {
    uint64_t bytes = 0;
    for (const auto &entry : fields) {
        bytes += entry.second.getArraySizeInBytes();
    }
    return bytes;
}

bool DeviceResidentNamedDataset::hasField(Thor::DatasetFieldId id) const {
    return fields.find(id) != fields.end();
}

bool DeviceResidentNamedDataset::hasTensor(const std::string &name) const {
    return schema.contains(name) && hasField(schema.getField(name).id);
}

const Tensor &DeviceResidentNamedDataset::field(Thor::DatasetFieldId id) const {
    const auto found = fields.find(id);
    if (found == fields.end()) {
        throw std::runtime_error("DeviceResidentNamedDataset does not contain requested field id.");
    }
    return found->second;
}

const Tensor &DeviceResidentNamedDataset::tensor(const std::string &name) const {
    return field(schema.getField(name).id);
}
