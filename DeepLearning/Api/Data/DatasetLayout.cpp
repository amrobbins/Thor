#include "DeepLearning/Api/Data/DatasetLayout.h"

#include <algorithm>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

using ThorImplementation::DataType;
using json = nlohmann::json;

namespace {

std::string dataTypeToString(DataType dataType) {
    switch (dataType) {
        case DataType::BOOLEAN: return "boolean";
        case DataType::INT8: return "int8";
        case DataType::UINT8: return "uint8";
        case DataType::INT16: return "int16";
        case DataType::UINT16: return "uint16";
        case DataType::INT32: return "int32";
        case DataType::UINT32: return "uint32";
        case DataType::INT64: return "int64";
        case DataType::UINT64: return "uint64";
        case DataType::FP16: return "fp16";
        case DataType::FP32: return "fp32";
        case DataType::FP64: return "fp64";
        case DataType::BF16: return "bf16";
        case DataType::FP8_E4M3: return "fp8_e4m3";
        case DataType::FP8_E5M2: return "fp8_e5m2";
        default: break;
    }
    throw std::runtime_error("Unsupported dataset storage data type value: " + std::to_string(static_cast<int>(dataType)));
}

std::string windowedTensorReferenceModeToString(DatasetLayout::WindowedTensorReferenceMode mode) {
    switch (mode) {
        case DatasetLayout::WindowedTensorReferenceMode::INDEXED: return "indexed";
        case DatasetLayout::WindowedTensorReferenceMode::AFFINE: return "affine";
    }
    throw std::runtime_error("Unsupported DatasetLayout windowed reference mode.");
}

DatasetLayout::WindowedTensorReferenceMode windowedTensorReferenceModeFromString(const std::string &value) {
    if (value == "indexed") return DatasetLayout::WindowedTensorReferenceMode::INDEXED;
    if (value == "affine") return DatasetLayout::WindowedTensorReferenceMode::AFFINE;
    throw std::runtime_error("Unsupported DatasetLayout windowed reference mode: " + value);
}

DataType dataTypeFromString(const std::string &value) {
    static const std::unordered_map<std::string, DataType> dataTypes = {
        {"boolean", DataType::BOOLEAN}, {"bool", DataType::BOOLEAN}, {"int8", DataType::INT8},
        {"uint8", DataType::UINT8}, {"int16", DataType::INT16}, {"uint16", DataType::UINT16},
        {"int32", DataType::INT32}, {"uint32", DataType::UINT32}, {"int64", DataType::INT64},
        {"uint64", DataType::UINT64}, {"fp16", DataType::FP16}, {"fp32", DataType::FP32},
        {"fp64", DataType::FP64}, {"bf16", DataType::BF16}, {"fp8_e4m3", DataType::FP8_E4M3},
        {"fp8_e5m2", DataType::FP8_E5M2},
    };
    const auto it = dataTypes.find(value);
    if (it == dataTypes.end()) {
        throw std::runtime_error("Unsupported dataset storage data type string: " + value);
    }
    return it->second;
}

uint64_t checkedWholeByteElementSizeBytes(DataType dataType) {
    switch (dataType) {
        case DataType::BOOLEAN:
        case DataType::INT8:
        case DataType::UINT8:
        case DataType::FP8_E4M3:
        case DataType::FP8_E5M2: return 1;
        case DataType::FP16:
        case DataType::BF16:
        case DataType::INT16:
        case DataType::UINT16: return 2;
        case DataType::FP32:
        case DataType::INT32:
        case DataType::UINT32: return 4;
        case DataType::FP64:
        case DataType::INT64:
        case DataType::UINT64: return 8;
        default: break;
    }
    throw std::runtime_error("Unsupported dataset storage data type value: " + std::to_string(static_cast<int>(dataType)));
}

bool isIntegerDataType(DataType dataType) {
    switch (dataType) {
        case DataType::INT8:
        case DataType::UINT8:
        case DataType::INT16:
        case DataType::UINT16:
        case DataType::INT32:
        case DataType::UINT32:
        case DataType::INT64:
        case DataType::UINT64: return true;
        default: return false;
    }
}

uint64_t checkedAdd(uint64_t a, uint64_t b, const std::string &context) {
    if (a > std::numeric_limits<uint64_t>::max() - b) {
        throw std::runtime_error("DatasetLayout overflow while adding " + context + ".");
    }
    return a + b;
}

uint64_t checkedMul(uint64_t a, uint64_t b, const std::string &context) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        throw std::runtime_error("DatasetLayout overflow while multiplying " + context + ".");
    }
    return a * b;
}

uint64_t elementCount(const std::vector<uint64_t> &dimensions, const std::string &name) {
    if (dimensions.empty()) {
        throw std::runtime_error("DatasetLayout tensor '" + name + "' has empty shape.");
    }
    uint64_t product = 1;
    for (uint64_t dim : dimensions) {
        if (dim == 0) {
            throw std::runtime_error("DatasetLayout tensor '" + name + "' has a zero dimension.");
        }
        product = checkedMul(product, dim, "shape for tensor '" + name + "'");
    }
    return product;
}

uint64_t optionalElementCount(const std::vector<uint64_t> &dimensions, const std::string &name) {
    uint64_t product = 1;
    for (uint64_t dim : dimensions) {
        if (dim == 0) {
            throw std::runtime_error("DatasetLayout source '" + name + "' has a zero step dimension.");
        }
        product = checkedMul(product, dim, "step shape for source '" + name + "'");
    }
    return product;
}

uint64_t expectedNumBytes(const std::vector<uint64_t> &dimensions, DataType dataType, const std::string &name) {
    return checkedMul(elementCount(dimensions, name), checkedWholeByteElementSizeBytes(dataType), "bytes for tensor '" + name + "'");
}

uint64_t expectedOptionalShapeNumBytes(const std::vector<uint64_t> &dimensions, DataType dataType, const std::string &name) {
    return checkedMul(optionalElementCount(dimensions, name), checkedWholeByteElementSizeBytes(dataType), "bytes for source '" + name + "'");
}

std::string shapeToString(const std::vector<uint64_t> &shape) {
    std::ostringstream out;
    out << '[';
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) out << ',';
        out << shape[i];
    }
    out << ']';
    return out.str();
}

void validateSourceStorage(const DatasetLayout::WindowedTensorSourceSpec &spec) {
    if (spec.sourceNumBytes == 0 && spec.sourceSequences.empty() && !spec.sourceFilename.has_value()) {
        return;
    }
    if (!spec.sourceFilename.has_value() || spec.sourceFilename->empty()) {
        throw std::runtime_error("DatasetLayout window source '" + spec.name + "' storage is missing source file.");
    }
    uint64_t maxEndBytes = 0;
    std::set<std::string> keys;
    for (const DatasetLayout::WindowedTensorSourceSequence &sequence : spec.sourceSequences) {
        if (sequence.keyHex.empty()) {
            throw std::runtime_error("DatasetLayout window source '" + spec.name + "' sequence has empty key_hex.");
        }
        if (!keys.insert(sequence.keyHex).second) {
            throw std::runtime_error("DatasetLayout window source '" + spec.name + "' has duplicate source key.");
        }
        if (sequence.numSteps == 0) {
            throw std::runtime_error("DatasetLayout window source '" + spec.name + "' sequence has zero steps.");
        }
        if (sequence.endIndexExclusive <= sequence.startIndex) {
            throw std::runtime_error("DatasetLayout window source '" + spec.name + "' sequence has invalid index bounds.");
        }
        const uint64_t expectedSteps = static_cast<uint64_t>(sequence.endIndexExclusive - sequence.startIndex);
        if (sequence.numSteps != expectedSteps) {
            throw std::runtime_error("DatasetLayout window source '" + spec.name + "' sequence num_steps does not match bounds.");
        }
        const uint64_t expectedBytes = checkedMul(sequence.numSteps, spec.stepNumBytes(), "window source sequence bytes");
        if (sequence.numBytes != expectedBytes) {
            throw std::runtime_error("DatasetLayout window source '" + spec.name + "' sequence num_bytes does not match step shape.");
        }
        maxEndBytes = std::max(maxEndBytes, checkedAdd(sequence.offsetBytes, sequence.numBytes, "window source sequence end"));
    }
    if (maxEndBytes != spec.sourceNumBytes) {
        throw std::runtime_error("DatasetLayout window source '" + spec.name + "' bytes do not match source sequences.");
    }
}

}  // namespace

bool DatasetLayout::TensorSpec::operator==(const TensorSpec &rhs) const {
    return name == rhs.name && dataType == rhs.dataType && dimensions == rhs.dimensions &&
           offsetBytes == rhs.offsetBytes && numBytes == rhs.numBytes;
}

DatasetLayout::TensorShape::TensorShape(std::string name, std::vector<uint64_t> dimensions, DataType dataType)
    : name(std::move(name)), dimensions(std::move(dimensions)), dataType(dataType) {}

bool DatasetLayout::WindowedTensorSourceSequence::operator==(const WindowedTensorSourceSequence &rhs) const {
    return keyHex == rhs.keyHex && startIndex == rhs.startIndex && endIndexExclusive == rhs.endIndexExclusive &&
           offsetBytes == rhs.offsetBytes && numSteps == rhs.numSteps && numBytes == rhs.numBytes;
}

uint64_t DatasetLayout::WindowedTensorSourceSpec::stepNumBytes() const {
    return expectedOptionalShapeNumBytes(stepDimensions, dataType, name);
}

uint64_t DatasetLayout::WindowedTensorSourceSpec::keyNumBytes() const {
    return checkedWholeByteElementSizeBytes(keyDataType);
}

bool DatasetLayout::WindowedTensorSourceSpec::operator==(const WindowedTensorSourceSpec &rhs) const {
    return name == rhs.name && dataType == rhs.dataType && stepDimensions == rhs.stepDimensions &&
           keyDataType == rhs.keyDataType && sourceFilename == rhs.sourceFilename &&
           sourceNumBytes == rhs.sourceNumBytes && sourceSequences == rhs.sourceSequences;
}

bool DatasetLayout::WindowedTensorSourceSpec::contractEquals(const WindowedTensorSourceSpec &rhs) const {
    return name == rhs.name && dataType == rhs.dataType && stepDimensions == rhs.stepDimensions &&
           keyDataType == rhs.keyDataType;
}

uint64_t DatasetLayout::WindowedTensorSpec::windowLength() const {
    if (dimensions.empty()) {
        throw std::runtime_error("DatasetLayout windowed tensor '" + name + "' has empty shape.");
    }
    return dimensions.front();
}

std::vector<uint64_t> DatasetLayout::WindowedTensorSpec::sourceStepDimensions() const {
    if (dimensions.empty()) {
        throw std::runtime_error("DatasetLayout windowed tensor '" + name + "' has empty shape.");
    }
    return std::vector<uint64_t>(dimensions.begin() + 1, dimensions.end());
}

uint64_t DatasetLayout::WindowedTensorSpec::sourceStepNumBytes() const {
    return expectedOptionalShapeNumBytes(sourceStepDimensions(), dataType, name);
}

uint64_t DatasetLayout::WindowedTensorSpec::outputNumBytes() const {
    return expectedNumBytes(dimensions, dataType, name);
}

uint64_t DatasetLayout::WindowedTensorSpec::keyNumBytes() const {
    return checkedWholeByteElementSizeBytes(keyDataType);
}

uint64_t DatasetLayout::WindowedTensorSpec::indexNumBytes() const {
    return checkedWholeByteElementSizeBytes(indexDataType);
}

bool DatasetLayout::WindowedTensorSpec::operator==(const WindowedTensorSpec &rhs) const {
    return name == rhs.name && sourceName == rhs.sourceName && dataType == rhs.dataType && dimensions == rhs.dimensions &&
           keyDataType == rhs.keyDataType && indexDataType == rhs.indexDataType && padValue == rhs.padValue &&
           maskName == rhs.maskName && referenceMode == rhs.referenceMode &&
           referenceOffsetBytes == rhs.referenceOffsetBytes && referenceNumBytes == rhs.referenceNumBytes;
}

bool DatasetLayout::WindowedTensorSpec::contractEquals(const WindowedTensorSpec &rhs) const {
    return *this == rhs;
}

DatasetLayout::WindowedTensorSourceShape::WindowedTensorSourceShape(std::string name,
                                                                    std::vector<uint64_t> stepDimensions,
                                                                    DataType dataType,
                                                                    DataType keyDataType)
    : name(std::move(name)), stepDimensions(std::move(stepDimensions)), dataType(dataType), keyDataType(keyDataType) {}

DatasetLayout::WindowedTensorShape::WindowedTensorShape(std::string name,
                                                        std::vector<uint64_t> dimensions,
                                                        std::string sourceName,
                                                        DataType indexDataType,
                                                        double padValue,
                                                        std::optional<std::string> maskName,
                                                        WindowedTensorReferenceMode referenceMode)
    : name(std::move(name)), dimensions(std::move(dimensions)), sourceName(std::move(sourceName)),
      indexDataType(indexDataType), padValue(padValue), maskName(std::move(maskName)), referenceMode(referenceMode) {}

DatasetLayout::DatasetLayout() : layoutRecordSizeBytes(0) {}

DatasetLayout::DatasetLayout(uint64_t recordSizeBytes, std::vector<TensorSpec> tensors)
    : DatasetLayout(recordSizeBytes, std::move(tensors), {}, {}) {}

DatasetLayout::DatasetLayout(uint64_t recordSizeBytes,
                             std::vector<TensorSpec> tensors,
                             std::vector<WindowedTensorSourceSpec> windowedTensorSources,
                             std::vector<WindowedTensorSpec> windowedTensors)
    : layoutRecordSizeBytes(recordSizeBytes), layoutTensors(std::move(tensors)),
      layoutWindowedTensorSources(std::move(windowedTensorSources)), layoutWindowedTensors(std::move(windowedTensors)) {
    validate();
}

uint64_t DatasetLayout::recordSizeBytes() const { return layoutRecordSizeBytes; }

const DatasetLayout::TensorSpec &DatasetLayout::tensor(std::string_view name) const {
    const auto it = std::find_if(layoutTensors.begin(), layoutTensors.end(), [&](const TensorSpec &spec) { return spec.name == name; });
    if (it == layoutTensors.end()) throw std::runtime_error("DatasetLayout tensor not found: " + std::string(name));
    return *it;
}

const std::vector<DatasetLayout::TensorSpec> &DatasetLayout::tensors() const { return layoutTensors; }

bool DatasetLayout::hasWindowedTensors() const { return !layoutWindowedTensors.empty(); }

bool DatasetLayout::hasAffineWindowedTensors() const {
    return std::any_of(layoutWindowedTensors.begin(), layoutWindowedTensors.end(), [](const WindowedTensorSpec &spec) {
        return spec.referenceMode == WindowedTensorReferenceMode::AFFINE;
    });
}

bool DatasetLayout::hasIndexedWindowedTensors() const {
    return std::any_of(layoutWindowedTensors.begin(), layoutWindowedTensors.end(), [](const WindowedTensorSpec &spec) {
        return spec.referenceMode == WindowedTensorReferenceMode::INDEXED;
    });
}

const DatasetLayout::WindowedTensorSourceSpec &DatasetLayout::windowedTensorSource(std::string_view name) const {
    const auto it = std::find_if(layoutWindowedTensorSources.begin(), layoutWindowedTensorSources.end(),
                                 [&](const WindowedTensorSourceSpec &spec) { return spec.name == name; });
    if (it == layoutWindowedTensorSources.end()) {
        throw std::runtime_error("DatasetLayout window source not found: " + std::string(name));
    }
    return *it;
}

const std::vector<DatasetLayout::WindowedTensorSourceSpec> &DatasetLayout::windowedTensorSources() const {
    return layoutWindowedTensorSources;
}

const DatasetLayout::WindowedTensorSpec &DatasetLayout::windowedTensor(std::string_view name) const {
    const auto it = std::find_if(layoutWindowedTensors.begin(), layoutWindowedTensors.end(),
                                 [&](const WindowedTensorSpec &spec) { return spec.name == name; });
    if (it == layoutWindowedTensors.end()) throw std::runtime_error("DatasetLayout windowed tensor not found: " + std::string(name));
    return *it;
}

const std::vector<DatasetLayout::WindowedTensorSpec> &DatasetLayout::windowedTensors() const { return layoutWindowedTensors; }

void DatasetLayout::validate() const {
    if (layoutTensors.empty() && layoutWindowedTensors.empty()) {
        throw std::runtime_error("DatasetLayout must contain at least one tensor or windowed tensor.");
    }

    std::set<std::string> sourceNames;
    for (const WindowedTensorSourceSpec &source : layoutWindowedTensorSources) {
        if (source.name.empty()) throw std::runtime_error("DatasetLayout window source names must be non-empty.");
        if (!sourceNames.insert(source.name).second) throw std::runtime_error("DatasetLayout duplicate window source name: " + source.name);
        (void)source.stepNumBytes();
        if (!isIntegerDataType(source.keyDataType)) {
            throw std::runtime_error("DatasetLayout window source '" + source.name + "' key dtype must be integer.");
        }
        validateSourceStorage(source);
    }

    std::set<std::string> names;
    std::vector<std::pair<uint64_t, uint64_t>> occupiedRanges;
    std::vector<std::string> occupiedNames;
    for (const TensorSpec &spec : layoutTensors) {
        if (spec.name.empty()) throw std::runtime_error("DatasetLayout tensor names must be non-empty.");
        if (!names.insert(spec.name).second) throw std::runtime_error("DatasetLayout duplicate tensor name: " + spec.name);
        const uint64_t expectedBytes = expectedNumBytes(spec.dimensions, spec.dataType, spec.name);
        if (spec.numBytes != expectedBytes) {
            throw std::runtime_error("DatasetLayout tensor '" + spec.name + "' num_bytes " + std::to_string(spec.numBytes) +
                                     " does not match shape " + shapeToString(spec.dimensions) + " and dtype " +
                                     dataTypeToString(spec.dataType) + " expected " + std::to_string(expectedBytes) + ".");
        }
        const uint64_t end = checkedAdd(spec.offsetBytes, spec.numBytes, "offset and num_bytes for tensor '" + spec.name + "'");
        if (end > layoutRecordSizeBytes) throw std::runtime_error("DatasetLayout tensor '" + spec.name + "' extends past record_size_bytes.");
        occupiedRanges.emplace_back(spec.offsetBytes, end);
        occupiedNames.push_back("tensor '" + spec.name + "'");
    }

    std::set<std::string> referencedSources;
    std::set<std::string> maskNames;
    for (const WindowedTensorSpec &spec : layoutWindowedTensors) {
        if (spec.name.empty()) throw std::runtime_error("DatasetLayout windowed tensor names must be non-empty.");
        if (!names.insert(spec.name).second) throw std::runtime_error("DatasetLayout duplicate tensor name: " + spec.name);
        const WindowedTensorSourceSpec &source = windowedTensorSource(spec.sourceName);
        referencedSources.insert(source.name);
        if (spec.dataType != source.dataType || spec.keyDataType != source.keyDataType ||
            spec.sourceStepDimensions() != source.stepDimensions) {
            throw std::runtime_error("DatasetLayout windowed tensor '" + spec.name + "' does not match source '" + source.name + "'.");
        }
        if (!isIntegerDataType(spec.indexDataType)) {
            throw std::runtime_error("DatasetLayout windowed tensor '" + spec.name + "' index dtype must be integer.");
        }
        (void)spec.outputNumBytes();
        if (spec.referenceMode == WindowedTensorReferenceMode::INDEXED) {
            const uint64_t expectedRefBytes = checkedAdd(spec.keyNumBytes(), spec.indexNumBytes(),
                                                         "windowed tensor reference bytes for '" + spec.name + "'");
            if (spec.referenceNumBytes != expectedRefBytes) {
                throw std::runtime_error("DatasetLayout windowed tensor '" + spec.name +
                                         "' reference_num_bytes does not match metadata dtypes.");
            }
            const uint64_t referenceEnd = checkedAdd(spec.referenceOffsetBytes, spec.referenceNumBytes,
                                                     "windowed tensor reference for '" + spec.name + "'");
            if (referenceEnd > layoutRecordSizeBytes) {
                throw std::runtime_error("DatasetLayout windowed tensor reference '" + spec.name +
                                         "' extends past record_size_bytes.");
            }
            occupiedRanges.emplace_back(spec.referenceOffsetBytes, referenceEnd);
            occupiedNames.push_back("windowed tensor reference '" + spec.name + "'");
        } else {
            if (spec.referenceOffsetBytes != 0 || spec.referenceNumBytes != 0) {
                throw std::runtime_error("DatasetLayout affine windowed tensor '" + spec.name +
                                         "' must not reserve indexed reference bytes.");
            }
        }
        if (spec.maskName.has_value()) {
            if (spec.maskName->empty()) throw std::runtime_error("DatasetLayout windowed mask names must be non-empty.");
            if (!names.insert(*spec.maskName).second || !maskNames.insert(*spec.maskName).second) {
                throw std::runtime_error("DatasetLayout duplicate tensor or mask name: " + *spec.maskName);
            }
        }
    }

    if (hasAffineWindowedTensors() && hasIndexedWindowedTensors()) {
        throw std::runtime_error("DatasetLayout does not support mixing indexed and affine window references in one layout.");
    }

    if (referencedSources.size() != layoutWindowedTensorSources.size()) {
        for (const WindowedTensorSourceSpec &source : layoutWindowedTensorSources) {
            if (referencedSources.find(source.name) == referencedSources.end()) {
                throw std::runtime_error("DatasetLayout window source '" + source.name + "' is not referenced by any windowed tensor.");
            }
        }
    }

    std::vector<size_t> order(occupiedRanges.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t left, size_t right) {
        if (occupiedRanges[left].first != occupiedRanges[right].first) return occupiedRanges[left].first < occupiedRanges[right].first;
        return occupiedNames[left] < occupiedNames[right];
    });

    uint64_t maxEndBytes = 0;
    uint64_t previousEndBytes = 0;
    bool havePrevious = false;
    std::string previousName;
    for (size_t index : order) {
        const uint64_t begin = occupiedRanges[index].first;
        const uint64_t end = occupiedRanges[index].second;
        if (havePrevious && begin < previousEndBytes) {
            throw std::runtime_error("DatasetLayout " + occupiedNames[index] + " overlaps " + previousName + ".");
        }
        havePrevious = true;
        previousEndBytes = end;
        previousName = occupiedNames[index];
        maxEndBytes = std::max(maxEndBytes, end);
    }
    if (maxEndBytes != layoutRecordSizeBytes) {
        throw std::runtime_error("DatasetLayout max record entry end " + std::to_string(maxEndBytes) +
                                 " does not equal record_size_bytes " + std::to_string(layoutRecordSizeBytes) + ".");
    }
}

void DatasetLayout::validateRequestedLayoutExact(const DatasetLayout &requested) const {
    validate();
    requested.validate();
    if (layoutRecordSizeBytes != requested.layoutRecordSizeBytes) {
        throw std::runtime_error("DatasetLayout requested record_size_bytes does not match manifest record_size_bytes.");
    }
    if (layoutTensors.size() != requested.layoutTensors.size()) {
        throw std::runtime_error("DatasetLayout requested tensor count does not match manifest tensor count.");
    }
    if (layoutWindowedTensorSources.size() != requested.layoutWindowedTensorSources.size()) {
        throw std::runtime_error("DatasetLayout requested window source count does not match manifest window source count.");
    }
    if (layoutWindowedTensors.size() != requested.layoutWindowedTensors.size()) {
        throw std::runtime_error("DatasetLayout requested windowed tensor count does not match manifest windowed tensor count.");
    }
    for (const TensorSpec &manifestSpec : layoutTensors) {
        if (manifestSpec != requested.tensor(manifestSpec.name)) {
            throw std::runtime_error("DatasetLayout requested tensor '" + manifestSpec.name + "' does not match manifest.");
        }
    }
    for (const WindowedTensorSourceSpec &manifestSpec : layoutWindowedTensorSources) {
        if (!manifestSpec.contractEquals(requested.windowedTensorSource(manifestSpec.name))) {
            throw std::runtime_error("DatasetLayout requested window source '" + manifestSpec.name + "' does not match manifest.");
        }
    }
    for (const WindowedTensorSpec &manifestSpec : layoutWindowedTensors) {
        if (!manifestSpec.contractEquals(requested.windowedTensor(manifestSpec.name))) {
            throw std::runtime_error("DatasetLayout requested windowed tensor '" + manifestSpec.name + "' does not match manifest.");
        }
    }
}

json DatasetLayout::toJson() const {
    validate();
    json root;
    root["format"] = FORMAT;
    root["record_size_bytes"] = layoutRecordSizeBytes;
    root["tensors"] = json::object();

    std::vector<const TensorSpec *> sortedTensors;
    for (const TensorSpec &spec : layoutTensors) sortedTensors.push_back(&spec);
    std::sort(sortedTensors.begin(), sortedTensors.end(), [](const TensorSpec *a, const TensorSpec *b) {
        if (a->offsetBytes != b->offsetBytes) return a->offsetBytes < b->offsetBytes;
        return a->name < b->name;
    });
    for (const TensorSpec *spec : sortedTensors) {
        root["tensors"][spec->name] = json{{"shape", spec->dimensions}, {"data_type", dataTypeToString(spec->dataType)},
                                                   {"offset_bytes", spec->offsetBytes}, {"num_bytes", spec->numBytes}};
    }

    if (!layoutWindowedTensorSources.empty()) {
        root["window_sources"] = json::object();
        std::vector<const WindowedTensorSourceSpec *> sortedSources;
        for (const WindowedTensorSourceSpec &spec : layoutWindowedTensorSources) sortedSources.push_back(&spec);
        std::sort(sortedSources.begin(), sortedSources.end(), [](const auto *a, const auto *b) { return a->name < b->name; });
        for (const WindowedTensorSourceSpec *spec : sortedSources) {
            json sourceJson{{"step_shape", spec->stepDimensions}, {"data_type", dataTypeToString(spec->dataType)},
                            {"key_data_type", dataTypeToString(spec->keyDataType)}};
            if (spec->sourceFilename.has_value()) {
                json storage{{"file", *spec->sourceFilename}, {"num_bytes", spec->sourceNumBytes}, {"sequences", json::array()}};
                for (const WindowedTensorSourceSequence &sequence : spec->sourceSequences) {
                    storage["sequences"].push_back(json{{"key_hex", sequence.keyHex}, {"start_index", sequence.startIndex},
                                                         {"end_index_exclusive", sequence.endIndexExclusive},
                                                         {"offset_bytes", sequence.offsetBytes}, {"num_steps", sequence.numSteps},
                                                         {"num_bytes", sequence.numBytes}});
                }
                sourceJson["storage"] = std::move(storage);
            }
            root["window_sources"][spec->name] = std::move(sourceJson);
        }
    }

    if (!layoutWindowedTensors.empty()) {
        root["windowed_tensors"] = json::object();
        std::vector<const WindowedTensorSpec *> sortedWindowed;
        for (const WindowedTensorSpec &spec : layoutWindowedTensors) sortedWindowed.push_back(&spec);
        std::sort(sortedWindowed.begin(), sortedWindowed.end(), [](const auto *a, const auto *b) {
            if (a->referenceOffsetBytes != b->referenceOffsetBytes) return a->referenceOffsetBytes < b->referenceOffsetBytes;
            return a->name < b->name;
        });
        for (const WindowedTensorSpec *spec : sortedWindowed) {
            json specJson{{"shape", spec->dimensions}, {"source", spec->sourceName},
                          {"index_data_type", dataTypeToString(spec->indexDataType)},
                          {"reference_mode", windowedTensorReferenceModeToString(spec->referenceMode)},
                          {"pad", json{{"mode", "constant"}, {"value", spec->padValue}}}};
            if (spec->referenceMode == WindowedTensorReferenceMode::INDEXED) {
                specJson["reference_offset_bytes"] = spec->referenceOffsetBytes;
                specJson["reference_num_bytes"] = spec->referenceNumBytes;
            }
            if (spec->maskName.has_value()) specJson["mask_name"] = *spec->maskName;
            root["windowed_tensors"][spec->name] = std::move(specJson);
        }
    }
    return root;
}

DatasetLayout DatasetLayout::fromJson(const json &j) {
    if (!j.is_object()) throw std::runtime_error("DatasetLayout manifest must be a JSON object.");
    if (!j.contains("format") || !j.at("format").is_string()) {
        throw std::runtime_error("DatasetLayout manifest is missing required string field 'format'.");
    }
    const std::string format = j.at("format").get<std::string>();
    if (format != FORMAT) throw std::runtime_error("DatasetLayout unsupported manifest format: " + format);
    const uint64_t recordSizeBytes = j.at("record_size_bytes").get<uint64_t>();
    const json &tensorJson = j.at("tensors");
    if (!tensorJson.is_object()) throw std::runtime_error("DatasetLayout manifest tensors field must be an object.");

    std::vector<TensorSpec> tensors;
    for (const auto &item : tensorJson.items()) {
        const json &specJson = item.value();
        tensors.push_back(TensorSpec{.name = item.key(), .dataType = dataTypeFromString(specJson.at("data_type").get<std::string>()),
                                     .dimensions = specJson.at("shape").get<std::vector<uint64_t>>(),
                                     .offsetBytes = specJson.at("offset_bytes").get<uint64_t>(),
                                     .numBytes = specJson.at("num_bytes").get<uint64_t>()});
    }

    std::vector<WindowedTensorSourceSpec> sources;
    if (j.contains("window_sources")) {
        const json &sourceJson = j.at("window_sources");
        if (!sourceJson.is_object()) throw std::runtime_error("DatasetLayout manifest window_sources field must be an object.");
        for (const auto &item : sourceJson.items()) {
            const json &specJson = item.value();
            WindowedTensorSourceSpec spec{.name = item.key(),
                                          .dataType = dataTypeFromString(specJson.at("data_type").get<std::string>()),
                                          .stepDimensions = specJson.at("step_shape").get<std::vector<uint64_t>>(),
                                          .keyDataType = dataTypeFromString(specJson.at("key_data_type").get<std::string>())};
            if (specJson.contains("storage")) {
                const json &storage = specJson.at("storage");
                spec.sourceFilename = storage.at("file").get<std::string>();
                spec.sourceNumBytes = storage.at("num_bytes").get<uint64_t>();
                const json &sequences = storage.at("sequences");
                if (!sequences.is_array()) throw std::runtime_error("DatasetLayout window source sequences must be an array.");
                for (const json &sequence : sequences) {
                    spec.sourceSequences.push_back(WindowedTensorSourceSequence{
                        .keyHex = sequence.at("key_hex").get<std::string>(),
                        .startIndex = sequence.at("start_index").get<int64_t>(),
                        .endIndexExclusive = sequence.at("end_index_exclusive").get<int64_t>(),
                        .offsetBytes = sequence.at("offset_bytes").get<uint64_t>(),
                        .numSteps = sequence.at("num_steps").get<uint64_t>(),
                        .numBytes = sequence.at("num_bytes").get<uint64_t>()});
                }
            }
            sources.push_back(std::move(spec));
        }
    }

    std::map<std::string, const WindowedTensorSourceSpec *> sourceByName;
    for (const WindowedTensorSourceSpec &source : sources) sourceByName.emplace(source.name, &source);

    std::vector<WindowedTensorSpec> windowed;
    if (j.contains("windowed_tensors")) {
        const json &windowedJson = j.at("windowed_tensors");
        if (!windowedJson.is_object()) throw std::runtime_error("DatasetLayout manifest windowed_tensors field must be an object.");
        for (const auto &item : windowedJson.items()) {
            const std::string name = item.key();
            const json &specJson = item.value();
            const std::string sourceName = specJson.at("source").get<std::string>();
            const auto sourceIt = sourceByName.find(sourceName);
            if (sourceIt == sourceByName.end()) {
                throw std::runtime_error("DatasetLayout windowed tensor '" + name + "' references unknown source '" + sourceName + "'.");
            }
            double padValue = 0.0;
            if (specJson.contains("pad")) {
                const json &pad = specJson.at("pad");
                if (pad.at("mode").get<std::string>() != "constant") {
                    throw std::runtime_error("DatasetLayout windowed tensor '" + name + "' has unsupported pad mode.");
                }
                padValue = pad.at("value").get<double>();
            }
            std::optional<std::string> maskName;
            if (specJson.contains("mask_name")) maskName = specJson.at("mask_name").get<std::string>();
            if (!specJson.contains("reference_mode") || !specJson.at("reference_mode").is_string()) {
                throw std::runtime_error("DatasetLayout windowed tensor '" + name +
                                         "' is missing required string field 'reference_mode'. "
                                         "Retired windowed dataset manifests are unsupported; rewrite the dataset using the current V1 writer.");
            }
            windowed.push_back(WindowedTensorSpec{
                .name = name, .sourceName = sourceName, .dataType = sourceIt->second->dataType,
                .dimensions = specJson.at("shape").get<std::vector<uint64_t>>(),
                .keyDataType = sourceIt->second->keyDataType,
                .indexDataType = dataTypeFromString(specJson.at("index_data_type").get<std::string>()),
                .padValue = padValue, .maskName = std::move(maskName),
                .referenceMode = windowedTensorReferenceModeFromString(specJson.at("reference_mode").get<std::string>()),
                .referenceOffsetBytes = specJson.value("reference_offset_bytes", uint64_t{0}),
                .referenceNumBytes = specJson.value("reference_num_bytes", uint64_t{0})});
        }
    }

    std::sort(tensors.begin(), tensors.end(), [](const TensorSpec &a, const TensorSpec &b) {
        if (a.offsetBytes != b.offsetBytes) return a.offsetBytes < b.offsetBytes;
        return a.name < b.name;
    });
    std::sort(sources.begin(), sources.end(), [](const auto &a, const auto &b) { return a.name < b.name; });
    std::sort(windowed.begin(), windowed.end(), [](const WindowedTensorSpec &a, const WindowedTensorSpec &b) {
        if (a.referenceOffsetBytes != b.referenceOffsetBytes) return a.referenceOffsetBytes < b.referenceOffsetBytes;
        return a.name < b.name;
    });
    return DatasetLayout(recordSizeBytes, std::move(tensors), std::move(sources), std::move(windowed));
}

void DatasetLayout::writeManifest(const std::filesystem::path &path) const {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) throw std::runtime_error("DatasetLayout failed to open manifest for writing: " + path.string());
    out << toJson().dump(2) << '\n';
    if (!out.good()) throw std::runtime_error("DatasetLayout failed while writing manifest: " + path.string());
}

DatasetLayout DatasetLayout::readManifest(const std::filesystem::path &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) throw std::runtime_error("DatasetLayout failed to open manifest for reading: " + path.string());
    json j;
    in >> j;
    if (!in.good() && !in.eof()) throw std::runtime_error("DatasetLayout failed while reading manifest: " + path.string());
    return fromJson(j);
}

DatasetLayout DatasetLayout::fromTensorShapes(const std::vector<TensorShape> &tensors) {
    return fromTensorShapes(tensors, {}, {});
}

DatasetLayout DatasetLayout::fromTensorShapes(const std::vector<TensorShape> &tensors,
                                              const std::vector<WindowedTensorSourceShape> &windowedTensorSources,
                                              const std::vector<WindowedTensorShape> &windowedTensors) {
    uint64_t offsetBytes = 0;
    std::vector<TensorSpec> tensorSpecs;
    tensorSpecs.reserve(tensors.size());
    for (const TensorShape &entry : tensors) {
        const uint64_t numBytes = expectedNumBytes(entry.dimensions, entry.dataType, entry.name);
        tensorSpecs.push_back(TensorSpec{.name = entry.name, .dataType = entry.dataType, .dimensions = entry.dimensions,
                                         .offsetBytes = offsetBytes, .numBytes = numBytes});
        offsetBytes = checkedAdd(offsetBytes, numBytes, "record size for tensor '" + entry.name + "'");
    }

    std::vector<WindowedTensorSourceSpec> sourceSpecs;
    sourceSpecs.reserve(windowedTensorSources.size());
    std::map<std::string, const WindowedTensorSourceSpec *> sourceByName;
    for (const WindowedTensorSourceShape &entry : windowedTensorSources) {
        sourceSpecs.push_back(WindowedTensorSourceSpec{.name = entry.name, .dataType = entry.dataType,
                                                       .stepDimensions = entry.stepDimensions,
                                                       .keyDataType = entry.keyDataType});
    }
    for (const WindowedTensorSourceSpec &source : sourceSpecs) {
        if (!sourceByName.emplace(source.name, &source).second) {
            throw std::runtime_error("DatasetLayout duplicate window source name: " + source.name);
        }
    }

    std::vector<WindowedTensorSpec> windowedSpecs;
    windowedSpecs.reserve(windowedTensors.size());
    for (const WindowedTensorShape &entry : windowedTensors) {
        const auto sourceIt = sourceByName.find(entry.sourceName);
        if (sourceIt == sourceByName.end()) {
            throw std::runtime_error("DatasetLayout windowed tensor '" + entry.name + "' references unknown source '" + entry.sourceName + "'.");
        }
        const WindowedTensorSourceSpec &source = *sourceIt->second;
        WindowedTensorSpec spec{.name = entry.name, .sourceName = entry.sourceName, .dataType = source.dataType,
                                .dimensions = entry.dimensions, .keyDataType = source.keyDataType,
                                .indexDataType = entry.indexDataType, .padValue = entry.padValue,
                                .maskName = entry.maskName, .referenceMode = entry.referenceMode};
        if (entry.referenceMode == WindowedTensorReferenceMode::INDEXED) {
            spec.referenceOffsetBytes = offsetBytes;
            spec.referenceNumBytes = checkedAdd(source.keyNumBytes(),
                                                checkedWholeByteElementSizeBytes(entry.indexDataType),
                                                "windowed tensor reference bytes for '" + entry.name + "'");
            offsetBytes = checkedAdd(offsetBytes, spec.referenceNumBytes,
                                     "record size for windowed tensor reference '" + entry.name + "'");
        }
        windowedSpecs.push_back(std::move(spec));
    }
    return DatasetLayout(offsetBytes, std::move(tensorSpecs), std::move(sourceSpecs), std::move(windowedSpecs));
}
