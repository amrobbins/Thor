#include "Utilities/Loaders/MaterializedNamedDatasetSnapshot.h"

uint64_t MaterializedNamedSplitSnapshot::totalBytes() const {
    uint64_t total = 0;
    for (const auto &entry : tensors) {
        total += entry.second.getDescriptor().getArraySizeInBytes();
    }
    return total;
}

const ThorImplementation::Tensor &MaterializedNamedSplitSnapshot::tensor(const std::string &name) const {
    const auto it = tensors.find(name);
    if (it == tensors.end()) {
        throw std::runtime_error("Materialized named split snapshot is missing tensor: " + name);
    }
    return it->second;
}

uint64_t MaterializedNamedDatasetSnapshot::totalExamples() const {
    uint64_t total = 0;
    for (const MaterializedNamedSplitSnapshot &split : splits) {
        total += split.numExamples();
    }
    return total;
}

uint64_t MaterializedNamedDatasetSnapshot::totalBytes() const {
    uint64_t total = 0;
    for (const MaterializedNamedSplitSnapshot &split : splits) {
        total += split.totalBytes();
    }
    return total;
}

const MaterializedNamedSplitSnapshot *MaterializedNamedDatasetSnapshot::findSplit(ExampleType exampleType) const {
    for (const MaterializedNamedSplitSnapshot &candidate : splits) {
        if (candidate.exampleType == exampleType) {
            return &candidate;
        }
    }
    return nullptr;
}

const MaterializedNamedSplitSnapshot &MaterializedNamedDatasetSnapshot::split(ExampleType exampleType) const {
    const MaterializedNamedSplitSnapshot *candidate = findSplit(exampleType);
    if (candidate == nullptr) {
        throw std::runtime_error("Materialized named dataset snapshot is missing requested split.");
    }
    return *candidate;
}
