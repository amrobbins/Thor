#include "Utilities/Loaders/DeviceResidentNamedDataset.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Common/Stream.h"

#include <chrono>
#include <set>
#include <utility>

using ThorImplementation::Tensor;
using ThorImplementation::TensorDescriptor;
using ThorImplementation::TensorPlacement;

namespace {

bool shouldUploadTensor(const std::set<std::string> &names, const std::string &name) {
    return names.empty() || names.find(name) != names.end();
}

}  // namespace

uint64_t DeviceResidentNamedSplit::totalBytes() const {
    uint64_t bytes = 0;
    for (const auto &entry : tensors) {
        bytes += entry.second.getArraySizeInBytes();
    }
    return bytes;
}

const Tensor &DeviceResidentNamedSplit::tensor(const std::string &name) const {
    const auto found = tensors.find(name);
    if (found == tensors.end()) {
        throw std::runtime_error("DeviceResidentNamedSplit does not contain tensor '" + name + "'.");
    }
    return found->second;
}

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
    if (snapshot.batchSize == 0) {
        throw std::runtime_error("DeviceResidentNamedDataset requires batch_size >= 1.");
    }

    auto dataset = std::make_shared<DeviceResidentNamedDataset>();
    dataset->layout = snapshot.layout;
    dataset->numDatasetExamples = snapshot.numDatasetExamples;
    dataset->batchSize = snapshot.batchSize;
    dataset->placement = devicePlacement;
    dataset->splits.reserve(snapshot.splits.size());

    const auto start = std::chrono::steady_clock::now();
    Stream uploadStream(devicePlacement);

    for (const MaterializedNamedSplitSnapshot &sourceSplit : snapshot.splits) {
        DeviceResidentNamedSplit residentSplit;
        residentSplit.exampleType = sourceSplit.exampleType;
        residentSplit.splitName = sourceSplit.splitName;
        residentSplit.sourceIndices = sourceSplit.sourceIndices;
        residentSplit.randomized = sourceSplit.randomized;
        residentSplit.seed = sourceSplit.seed;
        residentSplit.batchesPerEpoch = sourceSplit.batchesPerEpoch;

        if (sourceSplit.numExamples() != 0) {
            for (const auto &entry : sourceSplit.tensors) {
                if (!shouldUploadTensor(tensorNamesToUpload, entry.first)) {
                    continue;
                }
                const Tensor &hostTensor = entry.second;
                Tensor deviceTensor(devicePlacement, hostTensor.getDescriptor());
                deviceTensor.copyFromAsync(hostTensor, uploadStream);
                residentSplit.tensors.emplace(entry.first, deviceTensor);
            }
        }

        dataset->splits.push_back(std::move(residentSplit));
    }

    uploadStream.synchronize();
    dataset->uploadSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return dataset;
}

uint64_t DeviceResidentNamedDataset::totalExamples() const {
    uint64_t examples = 0;
    for (const DeviceResidentNamedSplit &split : splits) {
        examples += split.numExamples();
    }
    return examples;
}

uint64_t DeviceResidentNamedDataset::totalBytes() const {
    uint64_t bytes = 0;
    for (const DeviceResidentNamedSplit &split : splits) {
        bytes += split.totalBytes();
    }
    return bytes;
}

const DeviceResidentNamedSplit *DeviceResidentNamedDataset::findSplit(ExampleType exampleType) const {
    for (const DeviceResidentNamedSplit &candidate : splits) {
        if (candidate.exampleType == exampleType) {
            return &candidate;
        }
    }
    return nullptr;
}

const DeviceResidentNamedSplit &DeviceResidentNamedDataset::split(ExampleType exampleType) const {
    const DeviceResidentNamedSplit *found = findSplit(exampleType);
    if (found == nullptr) {
        throw std::runtime_error("DeviceResidentNamedDataset does not contain requested split.");
    }
    return *found;
}
