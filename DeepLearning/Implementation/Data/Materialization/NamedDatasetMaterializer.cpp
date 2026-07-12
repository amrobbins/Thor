#include "DeepLearning/Implementation/Data/Materialization/NamedDatasetMaterializer.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Loaders/IndexedLocalNamedExampleReader.h"

#include <chrono>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

using ThorImplementation::DataType;
using ThorImplementation::Tensor;
using ThorImplementation::TensorDescriptor;
using ThorImplementation::TensorPlacement;

namespace {

std::vector<uint64_t> prependExampleDimension(uint64_t numExamples,
                                              const std::vector<uint64_t> &exampleDimensions) {
    if (numExamples == 0) {
        throw std::runtime_error("Cannot materialize a zero-row named dataset.");
    }
    std::vector<uint64_t> dimensions;
    dimensions.reserve(exampleDimensions.size() + 1);
    dimensions.push_back(numExamples);
    dimensions.insert(dimensions.end(), exampleDimensions.begin(), exampleDimensions.end());
    return dimensions;
}

std::vector<uint64_t> materializedTensorDimensions(uint64_t numExamples,
                                                   const DatasetLayout::TensorSpec &spec) {
    return prependExampleDimension(numExamples, spec.dimensions);
}

std::vector<uint64_t> materializedWindowedTensorDimensions(
    uint64_t numExamples,
    const DatasetLayout::WindowedTensorSpec &spec) {
    return prependExampleDimension(numExamples, spec.dimensions);
}

std::vector<uint64_t> materializedWindowMaskDimensions(
    uint64_t numExamples,
    const DatasetLayout::WindowedTensorSpec &spec) {
    return prependExampleDimension(numExamples, std::vector<uint64_t>{spec.windowLength()});
}

}  // namespace

NamedDatasetMaterializationSupport checkNamedDatasetSnapshotMaterializationSupport(
    const Thor::DatasetMaterializationDescription &description) {
    if (description.source == Thor::DatasetMaterializationSource::FILE_DATASET &&
        description.datasetPath.empty()) {
        return NamedDatasetMaterializationSupport{false, "missing_dataset_path"};
    }
    if (description.numExamples == 0) {
        return NamedDatasetMaterializationSupport{false, "empty_dataset"};
    }
    if (description.layout.tensors().empty() && description.layout.windowedTensors().empty()) {
        return NamedDatasetMaterializationSupport{false, "empty_tensor_layout"};
    }
    for (const DatasetLayout::TensorSpec &spec : description.layout.tensors()) {
        if (spec.numBytes == 0) {
            return NamedDatasetMaterializationSupport{false, "zero_sized_tensor"};
        }
    }
    for (const DatasetLayout::WindowedTensorSourceSpec &source : description.layout.windowedTensorSources()) {
        if (source.stepNumBytes() == 0) {
            return NamedDatasetMaterializationSupport{false, "zero_sized_window_source"};
        }
        if (!source.sourceFilename.has_value() || source.sourceFilename->empty()) {
            return NamedDatasetMaterializationSupport{false, "missing_window_source_storage"};
        }
        if (source.sourceSequences.empty()) {
            return NamedDatasetMaterializationSupport{false, "missing_window_source_sequences"};
        }
    }
    for (const DatasetLayout::WindowedTensorSpec &spec : description.layout.windowedTensors()) {
        if (spec.outputNumBytes() == 0 || spec.sourceStepNumBytes() == 0 ||
            (spec.referenceMode == DatasetLayout::WindowedTensorReferenceMode::INDEXED && spec.referenceNumBytes == 0)) {
            return NamedDatasetMaterializationSupport{false, "zero_sized_windowed_tensor"};
        }
    }
    return NamedDatasetMaterializationSupport{true, ""};
}

MaterializedNamedDatasetSnapshot materializeNamedDatasetSnapshot(
    const Thor::DatasetMaterializationDescription &description,
    uint64_t readerQueueDepth) {
    if (readerQueueDepth == 0) {
        throw std::runtime_error("NamedDatasetMaterializer reader_queue_depth must be >= 1.");
    }

    const NamedDatasetMaterializationSupport support =
        checkNamedDatasetSnapshotMaterializationSupport(description);
    if (!support.supported) {
        throw std::runtime_error(
            "NamedDatasetMaterializer cannot materialize dataset snapshot: " + support.reason);
    }

    if (description.source != Thor::DatasetMaterializationSource::FILE_DATASET) {
        throw std::runtime_error(
            "NamedDatasetMaterializer requires the owning dataset backend to materialize an in-memory dataset.");
    }

    const auto started = std::chrono::steady_clock::now();
    std::shared_ptr<IndexedLocalNamedExampleReader> reader =
        IndexedLocalNamedExampleReader::openDataset(description.datasetPath, description.layout);
    if (reader->getNumExamples() != description.numExamples) {
        throw std::runtime_error(
            "NamedDatasetMaterializer dataset row count changed after the description was created.");
    }

    MaterializedNamedDatasetSnapshot out(
        description.datasetId,
        description.schema,
        reader->getLayout(),
        description.numExamples);
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);

    std::vector<uint8_t *> tensorBasePointers(reader->getTensorCount(), nullptr);
    for (const DatasetLayout::TensorSpec &spec : out.layout.tensors()) {
        Tensor tensor(
            cpuPlacement,
            TensorDescriptor(spec.dataType, materializedTensorDimensions(out.numExamples, spec)));
        const uint64_t readerOrdinal = reader->getLayoutTensorOrdinal(spec.name);
        THOR_THROW_IF_FALSE(readerOrdinal < tensorBasePointers.size());
        THOR_THROW_IF_FALSE(tensorBasePointers.at(readerOrdinal) == nullptr);
        tensorBasePointers.at(readerOrdinal) = static_cast<uint8_t *>(tensor.getMemPtr());
        out.fields.emplace(description.schema.getField(spec.name).id, std::move(tensor));
    }

    std::vector<uint8_t *> windowedTensorBasePointers(reader->getWindowedTensorCount(), nullptr);
    std::vector<uint8_t *> windowedMaskBasePointers(reader->getWindowedTensorCount(), nullptr);
    for (const DatasetLayout::WindowedTensorSpec &spec : out.layout.windowedTensors()) {
        Tensor tensor(
            cpuPlacement,
            TensorDescriptor(
                spec.dataType,
                materializedWindowedTensorDimensions(out.numExamples, spec)));
        const uint64_t readerOrdinal = reader->getLayoutWindowedTensorOrdinal(spec.name);
        THOR_THROW_IF_FALSE(readerOrdinal < windowedTensorBasePointers.size());
        THOR_THROW_IF_FALSE(windowedTensorBasePointers.at(readerOrdinal) == nullptr);
        windowedTensorBasePointers.at(readerOrdinal) = static_cast<uint8_t *>(tensor.getMemPtr());
        out.fields.emplace(description.schema.getField(spec.name).id, std::move(tensor));

        if (spec.maskName.has_value()) {
            Tensor mask(
                cpuPlacement,
                TensorDescriptor(
                    DataType::UINT8,
                    materializedWindowMaskDimensions(out.numExamples, spec)));
            windowedMaskBasePointers.at(readerOrdinal) = static_cast<uint8_t *>(mask.getMemPtr());
            out.fields.emplace(description.schema.getField(spec.maskName.value()).id, std::move(mask));
        }
    }

    for (uint8_t *basePointer : tensorBasePointers) {
        if (basePointer == nullptr) {
            throw std::runtime_error(
                "NamedDatasetMaterializer failed to bind every reader tensor ordinal.");
        }
    }
    for (uint64_t ordinal = 0; ordinal < windowedTensorBasePointers.size(); ++ordinal) {
        if (windowedTensorBasePointers.at(static_cast<size_t>(ordinal)) == nullptr) {
            throw std::runtime_error(
                "NamedDatasetMaterializer failed to bind every reader windowed tensor ordinal.");
        }
        const DatasetLayout::WindowedTensorSpec &spec =
            out.layout.windowedTensors().at(static_cast<size_t>(ordinal));
        if (spec.maskName.has_value() &&
            windowedMaskBasePointers.at(static_cast<size_t>(ordinal)) == nullptr) {
            throw std::runtime_error(
                "NamedDatasetMaterializer failed to bind every reader windowed mask ordinal.");
        }
    }

    std::unique_ptr<IndexedLocalNamedExampleReader::Session> session =
        reader->createSession(readerQueueDepth);
    for (uint64_t row = 0; row < out.numExamples; ++row) {
        if (out.layout.hasWindowedTensors()) {
            session->loadExampleInto(
                row,
                row,
                tensorBasePointers,
                windowedTensorBasePointers,
                windowedMaskBasePointers);
        } else {
            session->loadExampleInto(row, row, tensorBasePointers);
        }
    }
    session->drain();

    out.materializationSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    return out;
}
