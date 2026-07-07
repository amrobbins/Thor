#include "Utilities/Loaders/NamedDatasetMaterializer.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Loaders/IndexedLocalNamedExampleReader.h"

#include <chrono>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

using ThorImplementation::DataType;
using ThorImplementation::Tensor;
using ThorImplementation::TensorDescriptor;
using ThorImplementation::TensorPlacement;

namespace {

uint64_t checkedAdd(uint64_t left, uint64_t right, const char *context) {
    if (left > std::numeric_limits<uint64_t>::max() - right) {
        throw std::runtime_error(std::string(context) + " overflow while adding.");
    }
    return left + right;
}

uint64_t checkedMul(uint64_t left, uint64_t right, const char *context) {
    if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) {
        throw std::runtime_error(std::string(context) + " overflow while multiplying.");
    }
    return left * right;
}

std::vector<uint64_t> prependExampleDimension(uint64_t numExamples,
                                              const std::vector<uint64_t> &exampleDimensions) {
    if (numExamples == 0) {
        throw std::runtime_error("Cannot build a zero-sized Thor tensor descriptor for an empty materialized split.");
    }
    std::vector<uint64_t> dimensions;
    dimensions.reserve(exampleDimensions.size() + 1);
    dimensions.push_back(numExamples);
    dimensions.insert(dimensions.end(), exampleDimensions.begin(), exampleDimensions.end());
    return dimensions;
}

std::vector<uint64_t> materializedTensorDimensions(uint64_t numExamples,
                                                   const LocalNamedExampleLayout::TensorSpec &spec) {
    return prependExampleDimension(numExamples, spec.dimensions);
}

std::vector<uint64_t> materializedWindowedTensorDimensions(uint64_t numExamples,
                                                           const LocalNamedExampleLayout::WindowedTensorSpec &spec) {
    return prependExampleDimension(numExamples, spec.dimensions);
}

std::vector<uint64_t> materializedWindowMaskDimensions(uint64_t numExamples,
                                                       const LocalNamedExampleLayout::WindowedTensorSpec &spec) {
    return prependExampleDimension(numExamples, std::vector<uint64_t>{spec.windowLength()});
}

MaterializedNamedSplitSnapshot materializeSplit(const DeviceDatasetMaterializationSplitView &split,
                                                const LocalNamedExampleLayout &layout,
                                                const std::shared_ptr<IndexedLocalNamedExampleReader> &reader,
                                                uint64_t readerQueueDepth) {
    MaterializedNamedSplitSnapshot out;
    out.exampleType = split.exampleType;
    out.splitName = split.splitName;
    out.sourceIndices = split.indices;
    out.randomized = split.randomized;
    out.seed = split.seed;
    out.batchesPerEpoch = split.batchesPerEpoch;

    if (split.indices.empty()) {
        return out;
    }

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    std::vector<uint8_t *> tensorBasePointers(reader->getTensorCount(), nullptr);
    for (const LocalNamedExampleLayout::TensorSpec &spec : layout.tensors()) {
        Tensor tensor(cpuPlacement, TensorDescriptor(spec.dataType, materializedTensorDimensions(split.numExamples(), spec)));
        const uint64_t readerOrdinal = reader->getLayoutTensorOrdinal(spec.name);
        THOR_THROW_IF_FALSE(readerOrdinal < tensorBasePointers.size());
        THOR_THROW_IF_FALSE(tensorBasePointers.at(readerOrdinal) == nullptr);
        tensorBasePointers.at(readerOrdinal) = static_cast<uint8_t *>(tensor.getMemPtr());
        out.tensors.emplace(spec.name, std::move(tensor));
    }

    std::vector<uint8_t *> windowedTensorBasePointers(reader->getWindowedTensorCount(), nullptr);
    std::vector<uint8_t *> windowedMaskBasePointers(reader->getWindowedTensorCount(), nullptr);
    for (const LocalNamedExampleLayout::WindowedTensorSpec &spec : layout.windowedTensors()) {
        Tensor tensor(cpuPlacement,
                      TensorDescriptor(spec.dataType, materializedWindowedTensorDimensions(split.numExamples(), spec)));
        const uint64_t readerOrdinal = reader->getLayoutWindowedTensorOrdinal(spec.name);
        THOR_THROW_IF_FALSE(readerOrdinal < windowedTensorBasePointers.size());
        THOR_THROW_IF_FALSE(windowedTensorBasePointers.at(readerOrdinal) == nullptr);
        windowedTensorBasePointers.at(readerOrdinal) = static_cast<uint8_t *>(tensor.getMemPtr());
        out.tensors.emplace(spec.name, std::move(tensor));

        if (spec.maskName.has_value()) {
            Tensor mask(cpuPlacement,
                        TensorDescriptor(DataType::UINT8, materializedWindowMaskDimensions(split.numExamples(), spec)));
            windowedMaskBasePointers.at(readerOrdinal) = static_cast<uint8_t *>(mask.getMemPtr());
            out.tensors.emplace(spec.maskName.value(), std::move(mask));
        }
    }

    for (uint8_t *basePointer : tensorBasePointers) {
        if (basePointer == nullptr) {
            throw std::runtime_error("NamedDatasetMaterializer failed to bind every reader tensor ordinal.");
        }
    }
    for (uint64_t ordinal = 0; ordinal < windowedTensorBasePointers.size(); ++ordinal) {
        if (windowedTensorBasePointers.at(static_cast<size_t>(ordinal)) == nullptr) {
            throw std::runtime_error("NamedDatasetMaterializer failed to bind every reader windowed tensor ordinal.");
        }
        const LocalNamedExampleLayout::WindowedTensorSpec &spec = layout.windowedTensors().at(static_cast<size_t>(ordinal));
        if (spec.maskName.has_value() && windowedMaskBasePointers.at(static_cast<size_t>(ordinal)) == nullptr) {
            throw std::runtime_error("NamedDatasetMaterializer failed to bind every reader windowed mask ordinal.");
        }
    }

    std::unique_ptr<IndexedLocalNamedExampleReader::Session> session = reader->createSession(readerQueueDepth);
    for (uint64_t slot = 0; slot < split.indices.size(); ++slot) {
        reader->validateGlobalIndex(split.indices.at(slot), split.splitName.c_str());
        if (layout.hasWindowedTensors()) {
            session->loadExampleInto(split.indices.at(slot), slot, tensorBasePointers, windowedTensorBasePointers, windowedMaskBasePointers);
        } else {
            session->loadExampleInto(split.indices.at(slot), slot, tensorBasePointers);
        }
    }
    session->drain();
    return out;
}

}  // namespace

NamedDatasetMaterializationSupport checkNamedDatasetSnapshotMaterializationSupport(const DeviceDatasetMaterializationView &view) {
    if (view.datasetPath.empty()) {
        return NamedDatasetMaterializationSupport{false, "missing_dataset_path"};
    }
    if (view.batchSize == 0) {
        return NamedDatasetMaterializationSupport{false, "invalid_batch_size"};
    }
    if (view.layout.tensors().empty() && view.layout.windowedTensors().empty()) {
        return NamedDatasetMaterializationSupport{false, "empty_tensor_layout"};
    }
    for (const LocalNamedExampleLayout::TensorSpec &spec : view.layout.tensors()) {
        if (spec.numBytes == 0) {
            return NamedDatasetMaterializationSupport{false, "zero_sized_tensor"};
        }
    }
    for (const LocalNamedExampleLayout::WindowedTensorSpec &spec : view.layout.windowedTensors()) {
        if (spec.outputNumBytes() == 0 || spec.referenceNumBytes == 0 || spec.sourceStepNumBytes() == 0) {
            return NamedDatasetMaterializationSupport{false, "zero_sized_windowed_tensor"};
        }
        if (!spec.sourceFilename.has_value() || spec.sourceFilename->empty()) {
            return NamedDatasetMaterializationSupport{false, "missing_windowed_tensor_source"};
        }
        if (spec.sourceSequences.empty()) {
            return NamedDatasetMaterializationSupport{false, "missing_windowed_tensor_sequences"};
        }
    }
    return NamedDatasetMaterializationSupport{true, ""};
}

MaterializedNamedDatasetSnapshot materializeNamedDatasetSnapshot(const DeviceDatasetMaterializationView &view,
                                                                uint64_t readerQueueDepth) {
    if (readerQueueDepth == 0) {
        throw std::runtime_error("NamedDatasetMaterializer reader_queue_depth must be >= 1.");
    }

    const NamedDatasetMaterializationSupport support = checkNamedDatasetSnapshotMaterializationSupport(view);
    if (!support.supported) {
        throw std::runtime_error("NamedDatasetMaterializer cannot materialize dataset snapshot: " + support.reason);
    }

    const auto started = std::chrono::steady_clock::now();
    std::shared_ptr<IndexedLocalNamedExampleReader> reader = IndexedLocalNamedExampleReader::openDataset(view.datasetPath, view.layout);

    MaterializedNamedDatasetSnapshot out;
    out.layout = reader->getLayout();
    out.numDatasetExamples = reader->getNumExamples();
    out.batchSize = view.batchSize;
    out.splits.reserve(view.splits.size());

    uint64_t requestedExamples = 0;
    for (const DeviceDatasetMaterializationSplitView &split : view.splits) {
        requestedExamples = checkedAdd(requestedExamples, split.numExamples(), "NamedDatasetMaterializer requested example count");
        for (uint64_t index : split.indices) {
            reader->validateGlobalIndex(index, split.splitName.c_str());
        }
        out.splits.push_back(materializeSplit(split, out.layout, reader, readerQueueDepth));
    }

    // Touch the computed value so overflow checking above is not optimized into dead-looking code in debug builds.
    (void)checkedMul(requestedExamples, uint64_t{1}, "NamedDatasetMaterializer requested example count");

    const auto elapsed = std::chrono::steady_clock::now() - started;
    out.materializationSeconds = std::chrono::duration<double>(elapsed).count();
    return out;
}
