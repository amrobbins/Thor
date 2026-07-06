#include "Utilities/Loaders/LocalNamedExampleLayout.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <limits>
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
        case DataType::BOOLEAN:
            return "boolean";
        case DataType::INT8:
            return "int8";
        case DataType::UINT8:
            return "uint8";
        case DataType::INT16:
            return "int16";
        case DataType::UINT16:
            return "uint16";
        case DataType::INT32:
            return "int32";
        case DataType::UINT32:
            return "uint32";
        case DataType::INT64:
            return "int64";
        case DataType::UINT64:
            return "uint64";
        case DataType::FP16:
            return "fp16";
        case DataType::FP32:
            return "fp32";
        case DataType::FP64:
            return "fp64";
        case DataType::BF16:
            return "bf16";
        case DataType::FP8_E4M3:
            return "fp8_e4m3";
        case DataType::FP8_E5M2:
            return "fp8_e5m2";
        default:
            break;
    }
    throw std::runtime_error("Unsupported local named example data type value: " + std::to_string(static_cast<int>(dataType)));
}

DataType dataTypeFromString(const std::string &value) {
    static const std::unordered_map<std::string, DataType> dataTypes = {
        {"boolean", DataType::BOOLEAN}, {"bool", DataType::BOOLEAN},       {"int8", DataType::INT8},
        {"uint8", DataType::UINT8},     {"int16", DataType::INT16},        {"uint16", DataType::UINT16},
        {"int32", DataType::INT32},     {"uint32", DataType::UINT32},      {"int64", DataType::INT64},
        {"uint64", DataType::UINT64},   {"fp16", DataType::FP16},          {"fp32", DataType::FP32},
        {"fp64", DataType::FP64},       {"bf16", DataType::BF16},          {"fp8_e4m3", DataType::FP8_E4M3},
        {"fp8_e5m2", DataType::FP8_E5M2},
    };
    const auto it = dataTypes.find(value);
    if (it == dataTypes.end()) {
        throw std::runtime_error("Unsupported local named example data type string: " + value);
    }
    return it->second;
}

uint64_t checkedWholeByteElementSizeBytes(DataType dataType) {
    switch (dataType) {
        case DataType::BOOLEAN:
        case DataType::INT8:
        case DataType::UINT8:
        case DataType::FP8_E4M3:
        case DataType::FP8_E5M2:
            return 1;
        case DataType::FP16:
        case DataType::BF16:
        case DataType::INT16:
        case DataType::UINT16:
            return 2;
        case DataType::FP32:
        case DataType::INT32:
        case DataType::UINT32:
            return 4;
        case DataType::FP64:
        case DataType::INT64:
        case DataType::UINT64:
            return 8;
        default:
            break;
    }
    throw std::runtime_error("Unsupported local named example data type value: " + std::to_string(static_cast<int>(dataType)));
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
        case DataType::UINT64:
            return true;
        default:
            return false;
    }
}

uint64_t checkedAdd(uint64_t a, uint64_t b, const std::string &context) {
    if (a > std::numeric_limits<uint64_t>::max() - b) {
        throw std::runtime_error("LocalNamedExampleLayout overflow while adding " + context + ".");
    }
    return a + b;
}

uint64_t checkedMul(uint64_t a, uint64_t b, const std::string &context) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        throw std::runtime_error("LocalNamedExampleLayout overflow while multiplying " + context + ".");
    }
    return a * b;
}

uint64_t elementCount(const std::vector<uint64_t> &dimensions, const std::string &name) {
    if (dimensions.empty()) {
        throw std::runtime_error("LocalNamedExampleLayout tensor '" + name + "' has empty shape.");
    }
    uint64_t product = 1;
    for (uint64_t dim : dimensions) {
        if (dim == 0) {
            throw std::runtime_error("LocalNamedExampleLayout tensor '" + name + "' has a zero dimension.");
        }
        product = checkedMul(product, dim, "shape for tensor '" + name + "'");
    }
    return product;
}

uint64_t optionalElementCount(const std::vector<uint64_t> &dimensions, const std::string &name) {
    uint64_t product = 1;
    for (uint64_t dim : dimensions) {
        if (dim == 0) {
            throw std::runtime_error("LocalNamedExampleLayout tensor '" + name + "' has a zero dimension.");
        }
        product = checkedMul(product, dim, "shape for tensor '" + name + "'");
    }
    return product;
}

uint64_t expectedNumBytes(const std::vector<uint64_t> &dimensions, DataType dataType, const std::string &name) {
    return checkedMul(elementCount(dimensions, name), checkedWholeByteElementSizeBytes(dataType), "bytes for tensor '" + name + "'");
}

uint64_t expectedOptionalShapeNumBytes(const std::vector<uint64_t> &dimensions, DataType dataType, const std::string &name) {
    return checkedMul(optionalElementCount(dimensions, name), checkedWholeByteElementSizeBytes(dataType), "bytes for tensor '" + name + "'");
}

std::string shapeToString(const std::vector<uint64_t> &shape) {
    std::ostringstream out;
    out << '[';
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            out << ',';
        }
        out << shape[i];
    }
    out << ']';
    return out.str();
}

void validateSourceStorage(const LocalNamedExampleLayout::WindowedTensorSpec &spec) {
    if (spec.sourceNumBytes == 0 && spec.sourceSequences.empty() && !spec.sourceFilename.has_value()) {
        return;
    }
    if (!spec.sourceFilename.has_value() || spec.sourceFilename->empty()) {
        throw std::runtime_error("LocalNamedExampleLayout windowed tensor '" + spec.name + "' source storage is missing source file.");
    }
    uint64_t maxEndBytes = 0;
    std::set<std::string> keys;
    for (const LocalNamedExampleLayout::WindowedTensorSourceSequence &sequence : spec.sourceSequences) {
        if (sequence.keyHex.empty()) {
            throw std::runtime_error("LocalNamedExampleLayout windowed tensor '" + spec.name + "' source sequence has empty key_hex.");
        }
        if (!keys.insert(sequence.keyHex).second) {
            throw std::runtime_error("LocalNamedExampleLayout windowed tensor '" + spec.name + "' has duplicate source key.");
        }
        if (sequence.numSteps == 0) {
            throw std::runtime_error("LocalNamedExampleLayout windowed tensor '" + spec.name + "' source sequence has zero steps.");
        }
        if (sequence.endIndexExclusive <= sequence.startIndex) {
            throw std::runtime_error("LocalNamedExampleLayout windowed tensor '" + spec.name + "' source sequence has invalid index bounds.");
        }
        const uint64_t expectedSteps = static_cast<uint64_t>(sequence.endIndexExclusive - sequence.startIndex);
        if (sequence.numSteps != expectedSteps) {
            throw std::runtime_error("LocalNamedExampleLayout windowed tensor '" + spec.name + "' source sequence num_steps does not match bounds.");
        }
        const uint64_t expectedBytes = checkedMul(sequence.numSteps, spec.sourceStepNumBytes(), "windowed source sequence bytes");
        if (sequence.numBytes != expectedBytes) {
            throw std::runtime_error("LocalNamedExampleLayout windowed tensor '" + spec.name + "' source sequence num_bytes does not match step shape.");
        }
        maxEndBytes = std::max(maxEndBytes, checkedAdd(sequence.offsetBytes, sequence.numBytes, "windowed source sequence end"));
    }
    if (maxEndBytes != spec.sourceNumBytes) {
        throw std::runtime_error("LocalNamedExampleLayout windowed tensor '" + spec.name + "' source bytes do not match source sequences.");
    }
}

}  // namespace

bool LocalNamedExampleLayout::TensorSpec::operator==(const TensorSpec &rhs) const {
    return name == rhs.name && dataType == rhs.dataType && dimensions == rhs.dimensions && offsetBytes == rhs.offsetBytes &&
           numBytes == rhs.numBytes;
}

bool LocalNamedExampleLayout::WindowedTensorSourceSequence::operator==(const WindowedTensorSourceSequence &rhs) const {
    return keyHex == rhs.keyHex && startIndex == rhs.startIndex && endIndexExclusive == rhs.endIndexExclusive &&
           offsetBytes == rhs.offsetBytes && numSteps == rhs.numSteps && numBytes == rhs.numBytes;
}

uint64_t LocalNamedExampleLayout::WindowedTensorSpec::windowLength() const {
    if (dimensions.empty()) {
        throw std::runtime_error("LocalNamedExampleLayout windowed tensor '" + name + "' has empty shape.");
    }
    return dimensions.front();
}

std::vector<uint64_t> LocalNamedExampleLayout::WindowedTensorSpec::sourceStepDimensions() const {
    if (dimensions.empty()) {
        throw std::runtime_error("LocalNamedExampleLayout windowed tensor '" + name + "' has empty shape.");
    }
    return std::vector<uint64_t>(dimensions.begin() + 1, dimensions.end());
}

uint64_t LocalNamedExampleLayout::WindowedTensorSpec::sourceStepNumBytes() const {
    return expectedOptionalShapeNumBytes(sourceStepDimensions(), dataType, name);
}

uint64_t LocalNamedExampleLayout::WindowedTensorSpec::outputNumBytes() const { return expectedNumBytes(dimensions, dataType, name); }

uint64_t LocalNamedExampleLayout::WindowedTensorSpec::keyNumBytes() const { return checkedWholeByteElementSizeBytes(keyDataType); }

uint64_t LocalNamedExampleLayout::WindowedTensorSpec::indexNumBytes() const { return checkedWholeByteElementSizeBytes(indexDataType); }

bool LocalNamedExampleLayout::WindowedTensorSpec::operator==(const WindowedTensorSpec &rhs) const {
    return name == rhs.name && dataType == rhs.dataType && dimensions == rhs.dimensions && keyDataType == rhs.keyDataType &&
           indexDataType == rhs.indexDataType && padValue == rhs.padValue && maskName == rhs.maskName &&
           referenceOffsetBytes == rhs.referenceOffsetBytes && referenceNumBytes == rhs.referenceNumBytes &&
           sourceFilename == rhs.sourceFilename && sourceNumBytes == rhs.sourceNumBytes && sourceSequences == rhs.sourceSequences;
}

bool LocalNamedExampleLayout::WindowedTensorSpec::contractEquals(const WindowedTensorSpec &rhs) const {
    return name == rhs.name && dataType == rhs.dataType && dimensions == rhs.dimensions && keyDataType == rhs.keyDataType &&
           indexDataType == rhs.indexDataType && padValue == rhs.padValue && maskName == rhs.maskName &&
           referenceOffsetBytes == rhs.referenceOffsetBytes && referenceNumBytes == rhs.referenceNumBytes;
}

LocalNamedExampleLayout::WindowedTensorShape::WindowedTensorShape(std::string name,
                                                                  std::vector<uint64_t> dimensions,
                                                                  DataType keyDataType,
                                                                  DataType indexDataType,
                                                                  double padValue,
                                                                  std::optional<std::string> maskName,
                                                                  DataType dataType)
    : name(std::move(name)),
      dimensions(std::move(dimensions)),
      keyDataType(keyDataType),
      indexDataType(indexDataType),
      padValue(padValue),
      maskName(std::move(maskName)),
      dataType(dataType) {}

LocalNamedExampleLayout::LocalNamedExampleLayout() : layoutDataType(DataType::FP32), layoutRecordSizeBytes(0) {}

LocalNamedExampleLayout::LocalNamedExampleLayout(DataType dataType, uint64_t recordSizeBytes, std::vector<TensorSpec> tensors)
    : LocalNamedExampleLayout(dataType, recordSizeBytes, std::move(tensors), {}) {}

LocalNamedExampleLayout::LocalNamedExampleLayout(DataType dataType,
                                                 uint64_t recordSizeBytes,
                                                 std::vector<TensorSpec> tensors,
                                                 std::vector<WindowedTensorSpec> windowedTensors)
    : layoutDataType(dataType),
      layoutRecordSizeBytes(recordSizeBytes),
      layoutTensors(std::move(tensors)),
      layoutWindowedTensors(std::move(windowedTensors)) {
    validate();
}

DataType LocalNamedExampleLayout::dataType() const { return layoutDataType; }

uint64_t LocalNamedExampleLayout::recordSizeBytes() const { return layoutRecordSizeBytes; }

const LocalNamedExampleLayout::TensorSpec &LocalNamedExampleLayout::tensor(std::string_view name) const {
    const auto it = std::find_if(layoutTensors.begin(), layoutTensors.end(), [&](const TensorSpec &spec) { return spec.name == name; });
    if (it == layoutTensors.end()) {
        throw std::runtime_error("LocalNamedExampleLayout tensor not found: " + std::string(name));
    }
    return *it;
}

const std::vector<LocalNamedExampleLayout::TensorSpec> &LocalNamedExampleLayout::tensors() const { return layoutTensors; }

bool LocalNamedExampleLayout::hasWindowedTensors() const { return !layoutWindowedTensors.empty(); }

const LocalNamedExampleLayout::WindowedTensorSpec &LocalNamedExampleLayout::windowedTensor(std::string_view name) const {
    const auto it = std::find_if(layoutWindowedTensors.begin(), layoutWindowedTensors.end(), [&](const WindowedTensorSpec &spec) {
        return spec.name == name;
    });
    if (it == layoutWindowedTensors.end()) {
        throw std::runtime_error("LocalNamedExampleLayout windowed tensor not found: " + std::string(name));
    }
    return *it;
}

const std::vector<LocalNamedExampleLayout::WindowedTensorSpec> &LocalNamedExampleLayout::windowedTensors() const {
    return layoutWindowedTensors;
}

void LocalNamedExampleLayout::validate() const {
    if (layoutRecordSizeBytes == 0) {
        throw std::runtime_error("LocalNamedExampleLayout record_size_bytes must be non-zero.");
    }
    if (layoutTensors.empty() && layoutWindowedTensors.empty()) {
        throw std::runtime_error("LocalNamedExampleLayout must contain at least one tensor or windowed tensor.");
    }

    const uint64_t elementSizeBytes = checkedWholeByteElementSizeBytes(layoutDataType);
    std::set<std::string> names;
    std::vector<std::pair<uint64_t, uint64_t>> occupiedRanges;
    std::vector<std::string> occupiedNames;
    std::vector<const TensorSpec *> sortedSpecs;
    sortedSpecs.reserve(layoutTensors.size());

    for (const TensorSpec &spec : layoutTensors) {
        if (spec.name.empty()) {
            throw std::runtime_error("LocalNamedExampleLayout tensor names must be non-empty.");
        }
        if (!names.insert(spec.name).second) {
            throw std::runtime_error("LocalNamedExampleLayout duplicate tensor name: " + spec.name);
        }
        if (spec.dataType != layoutDataType) {
            throw std::runtime_error("LocalNamedExampleLayout tensor '" + spec.name + "' dtype " + dataTypeToString(spec.dataType) +
                                     " does not match layout dtype " + dataTypeToString(layoutDataType) + ".");
        }
        if (spec.offsetBytes % elementSizeBytes != 0) {
            throw std::runtime_error("LocalNamedExampleLayout tensor '" + spec.name + "' offset is not aligned to dtype size.");
        }
        if (spec.numBytes % elementSizeBytes != 0) {
            throw std::runtime_error("LocalNamedExampleLayout tensor '" + spec.name + "' byte count is not aligned to dtype size.");
        }
        const uint64_t expectedBytes = expectedNumBytes(spec.dimensions, spec.dataType, spec.name);
        if (spec.numBytes != expectedBytes) {
            throw std::runtime_error("LocalNamedExampleLayout tensor '" + spec.name + "' num_bytes " + std::to_string(spec.numBytes) +
                                     " does not match shape " + shapeToString(spec.dimensions) + " and dtype " + dataTypeToString(spec.dataType) +
                                     " expected " + std::to_string(expectedBytes) + ".");
        }
        const uint64_t end = checkedAdd(spec.offsetBytes, spec.numBytes, "offset and num_bytes for tensor '" + spec.name + "'");
        if (end > layoutRecordSizeBytes) {
            throw std::runtime_error("LocalNamedExampleLayout tensor '" + spec.name + "' extends past record_size_bytes.");
        }
        sortedSpecs.push_back(&spec);
        occupiedRanges.emplace_back(spec.offsetBytes, end);
        occupiedNames.push_back("tensor '" + spec.name + "'");
    }

    std::set<std::string> maskNames;
    for (const WindowedTensorSpec &spec : layoutWindowedTensors) {
        if (spec.name.empty()) {
            throw std::runtime_error("LocalNamedExampleLayout windowed tensor names must be non-empty.");
        }
        if (!names.insert(spec.name).second) {
            throw std::runtime_error("LocalNamedExampleLayout duplicate tensor/windowed tensor name: " + spec.name);
        }
        if (spec.dataType != layoutDataType) {
            throw std::runtime_error("LocalNamedExampleLayout windowed tensor '" + spec.name + "' dtype " + dataTypeToString(spec.dataType) +
                                     " does not match layout dtype " + dataTypeToString(layoutDataType) + ".");
        }
        if (!isIntegerDataType(spec.keyDataType)) {
            throw std::runtime_error("LocalNamedExampleLayout windowed tensor '" + spec.name + "' key dtype must be an integer type.");
        }
        if (!isIntegerDataType(spec.indexDataType)) {
            throw std::runtime_error("LocalNamedExampleLayout windowed tensor '" + spec.name + "' index dtype must be an integer type.");
        }
        const uint64_t expectedReferenceBytes = checkedAdd(spec.keyNumBytes(), spec.indexNumBytes(),
                                                          "windowed tensor reference bytes for '" + spec.name + "'");
        if (spec.referenceNumBytes != expectedReferenceBytes) {
            throw std::runtime_error("LocalNamedExampleLayout windowed tensor '" + spec.name + "' reference_num_bytes " +
                                     std::to_string(spec.referenceNumBytes) + " does not match key+index byte count " +
                                     std::to_string(expectedReferenceBytes) + ".");
        }
        (void)spec.outputNumBytes();
        const uint64_t referenceEnd = checkedAdd(spec.referenceOffsetBytes,
                                                 spec.referenceNumBytes,
                                                 "reference offset and num_bytes for windowed tensor '" + spec.name + "'");
        if (referenceEnd > layoutRecordSizeBytes) {
            throw std::runtime_error("LocalNamedExampleLayout windowed tensor '" + spec.name + "' reference extends past record_size_bytes.");
        }
        if (spec.maskName.has_value()) {
            if (spec.maskName->empty()) {
                throw std::runtime_error("LocalNamedExampleLayout windowed tensor '" + spec.name + "' mask name must be non-empty.");
            }
            if (names.find(*spec.maskName) != names.end()) {
                throw std::runtime_error("LocalNamedExampleLayout windowed tensor '" + spec.name + "' mask name collides with tensor name: " +
                                         *spec.maskName);
            }
            if (!maskNames.insert(*spec.maskName).second) {
                throw std::runtime_error("LocalNamedExampleLayout duplicate windowed tensor mask name: " + *spec.maskName);
            }
        }
        validateSourceStorage(spec);
        occupiedRanges.emplace_back(spec.referenceOffsetBytes, referenceEnd);
        occupiedNames.push_back("windowed tensor reference '" + spec.name + "'");
    }

    std::sort(sortedSpecs.begin(), sortedSpecs.end(), [](const TensorSpec *a, const TensorSpec *b) {
        if (a->offsetBytes != b->offsetBytes) {
            return a->offsetBytes < b->offsetBytes;
        }
        return a->name < b->name;
    });

    std::vector<size_t> order(occupiedRanges.size());
    for (size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(), [&](size_t left, size_t right) {
        if (occupiedRanges[left].first != occupiedRanges[right].first) {
            return occupiedRanges[left].first < occupiedRanges[right].first;
        }
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
            throw std::runtime_error("LocalNamedExampleLayout " + occupiedNames[index] + " overlaps " + previousName + ".");
        }
        havePrevious = true;
        previousEndBytes = end;
        previousName = occupiedNames[index];
        maxEndBytes = std::max(maxEndBytes, end);
    }

    if (maxEndBytes != layoutRecordSizeBytes) {
        throw std::runtime_error("LocalNamedExampleLayout max record entry end " + std::to_string(maxEndBytes) +
                                 " does not equal record_size_bytes " + std::to_string(layoutRecordSizeBytes) + ".");
    }
}

void LocalNamedExampleLayout::validateRequestedLayoutExact(const LocalNamedExampleLayout &requested) const {
    validate();
    requested.validate();

    if (layoutDataType != requested.layoutDataType) {
        throw std::runtime_error("LocalNamedExampleLayout requested dtype " + dataTypeToString(requested.layoutDataType) +
                                 " does not match manifest dtype " + dataTypeToString(layoutDataType) + ".");
    }
    if (layoutRecordSizeBytes != requested.layoutRecordSizeBytes) {
        throw std::runtime_error("LocalNamedExampleLayout requested record_size_bytes " + std::to_string(requested.layoutRecordSizeBytes) +
                                 " does not match manifest record_size_bytes " + std::to_string(layoutRecordSizeBytes) + ".");
    }
    if (layoutTensors.size() != requested.layoutTensors.size()) {
        throw std::runtime_error("LocalNamedExampleLayout requested tensor count " + std::to_string(requested.layoutTensors.size()) +
                                 " does not match manifest tensor count " + std::to_string(layoutTensors.size()) + ".");
    }
    if (layoutWindowedTensors.size() != requested.layoutWindowedTensors.size()) {
        throw std::runtime_error("LocalNamedExampleLayout requested windowed tensor count " +
                                 std::to_string(requested.layoutWindowedTensors.size()) + " does not match manifest windowed tensor count " +
                                 std::to_string(layoutWindowedTensors.size()) + ".");
    }

    for (const TensorSpec &manifestSpec : layoutTensors) {
        const TensorSpec &requestedSpec = requested.tensor(manifestSpec.name);
        if (manifestSpec != requestedSpec) {
            std::ostringstream message;
            message << "LocalNamedExampleLayout requested tensor '" << manifestSpec.name << "' does not match manifest.";
            if (manifestSpec.dimensions != requestedSpec.dimensions) {
                message << " shape manifest=" << shapeToString(manifestSpec.dimensions)
                        << " requested=" << shapeToString(requestedSpec.dimensions) << '.';
            }
            if (manifestSpec.offsetBytes != requestedSpec.offsetBytes) {
                message << " offset manifest=" << manifestSpec.offsetBytes << " requested=" << requestedSpec.offsetBytes << '.';
            }
            if (manifestSpec.numBytes != requestedSpec.numBytes) {
                message << " num_bytes manifest=" << manifestSpec.numBytes << " requested=" << requestedSpec.numBytes << '.';
            }
            if (manifestSpec.dataType != requestedSpec.dataType) {
                message << " dtype manifest=" << dataTypeToString(manifestSpec.dataType)
                        << " requested=" << dataTypeToString(requestedSpec.dataType) << '.';
            }
            throw std::runtime_error(message.str());
        }
    }

    for (const WindowedTensorSpec &manifestSpec : layoutWindowedTensors) {
        const WindowedTensorSpec &requestedSpec = requested.windowedTensor(manifestSpec.name);
        if (!manifestSpec.contractEquals(requestedSpec)) {
            std::ostringstream message;
            message << "LocalNamedExampleLayout requested windowed tensor '" << manifestSpec.name << "' does not match manifest.";
            if (manifestSpec.dimensions != requestedSpec.dimensions) {
                message << " shape manifest=" << shapeToString(manifestSpec.dimensions)
                        << " requested=" << shapeToString(requestedSpec.dimensions) << '.';
            }
            if (manifestSpec.referenceOffsetBytes != requestedSpec.referenceOffsetBytes) {
                message << " reference_offset manifest=" << manifestSpec.referenceOffsetBytes
                        << " requested=" << requestedSpec.referenceOffsetBytes << '.';
            }
            if (manifestSpec.referenceNumBytes != requestedSpec.referenceNumBytes) {
                message << " reference_num_bytes manifest=" << manifestSpec.referenceNumBytes
                        << " requested=" << requestedSpec.referenceNumBytes << '.';
            }
            if (manifestSpec.dataType != requestedSpec.dataType) {
                message << " dtype manifest=" << dataTypeToString(manifestSpec.dataType)
                        << " requested=" << dataTypeToString(requestedSpec.dataType) << '.';
            }
            if (manifestSpec.keyDataType != requestedSpec.keyDataType) {
                message << " key_dtype manifest=" << dataTypeToString(manifestSpec.keyDataType)
                        << " requested=" << dataTypeToString(requestedSpec.keyDataType) << '.';
            }
            if (manifestSpec.indexDataType != requestedSpec.indexDataType) {
                message << " index_dtype manifest=" << dataTypeToString(manifestSpec.indexDataType)
                        << " requested=" << dataTypeToString(requestedSpec.indexDataType) << '.';
            }
            throw std::runtime_error(message.str());
        }
    }
}

json LocalNamedExampleLayout::toJson() const {
    validate();

    json root;
    root["format"] = FORMAT;
    root["data_type"] = dataTypeToString(layoutDataType);
    root["record_size_bytes"] = layoutRecordSizeBytes;
    root["tensors"] = json::object();

    std::vector<const TensorSpec *> sortedSpecs;
    sortedSpecs.reserve(layoutTensors.size());
    for (const TensorSpec &spec : layoutTensors) {
        sortedSpecs.push_back(&spec);
    }
    std::sort(sortedSpecs.begin(), sortedSpecs.end(), [](const TensorSpec *a, const TensorSpec *b) {
        if (a->offsetBytes != b->offsetBytes) {
            return a->offsetBytes < b->offsetBytes;
        }
        return a->name < b->name;
    });

    for (const TensorSpec *spec : sortedSpecs) {
        root["tensors"][spec->name] = json{{"shape", spec->dimensions}, {"offset_bytes", spec->offsetBytes}, {"num_bytes", spec->numBytes}};
    }

    if (!layoutWindowedTensors.empty()) {
        root["windowed_tensors"] = json::object();
        std::vector<const WindowedTensorSpec *> sortedWindowedSpecs;
        sortedWindowedSpecs.reserve(layoutWindowedTensors.size());
        for (const WindowedTensorSpec &spec : layoutWindowedTensors) {
            sortedWindowedSpecs.push_back(&spec);
        }
        std::sort(sortedWindowedSpecs.begin(), sortedWindowedSpecs.end(), [](const WindowedTensorSpec *a, const WindowedTensorSpec *b) {
            if (a->referenceOffsetBytes != b->referenceOffsetBytes) {
                return a->referenceOffsetBytes < b->referenceOffsetBytes;
            }
            return a->name < b->name;
        });
        for (const WindowedTensorSpec *spec : sortedWindowedSpecs) {
            json specJson{{"shape", spec->dimensions},
                          {"data_type", dataTypeToString(spec->dataType)},
                          {"key_data_type", dataTypeToString(spec->keyDataType)},
                          {"index_data_type", dataTypeToString(spec->indexDataType)},
                          {"reference_offset_bytes", spec->referenceOffsetBytes},
                          {"reference_num_bytes", spec->referenceNumBytes},
                          {"pad", json{{"mode", "constant"}, {"value", spec->padValue}}}};
            if (spec->maskName.has_value()) {
                specJson["mask_name"] = spec->maskName.value();
            }
            if (spec->sourceFilename.has_value()) {
                json sourceStorage{{"file", spec->sourceFilename.value()}, {"num_bytes", spec->sourceNumBytes}, {"sequences", json::array()}};
                for (const WindowedTensorSourceSequence &sequence : spec->sourceSequences) {
                    sourceStorage["sequences"].push_back(json{{"key_hex", sequence.keyHex},
                                                               {"start_index", sequence.startIndex},
                                                               {"end_index_exclusive", sequence.endIndexExclusive},
                                                               {"offset_bytes", sequence.offsetBytes},
                                                               {"num_steps", sequence.numSteps},
                                                               {"num_bytes", sequence.numBytes}});
                }
                specJson["source_storage"] = std::move(sourceStorage);
            }
            root["windowed_tensors"][spec->name] = std::move(specJson);
        }
    }
    return root;
}

LocalNamedExampleLayout LocalNamedExampleLayout::fromJson(const json &j) {
    if (!j.is_object()) {
        throw std::runtime_error("LocalNamedExampleLayout manifest must be a JSON object.");
    }
    const std::string format = j.at("format").get<std::string>();
    if (format != FORMAT) {
        throw std::runtime_error("LocalNamedExampleLayout unsupported manifest format: " + format);
    }
    const DataType dataType = dataTypeFromString(j.at("data_type").get<std::string>());
    const uint64_t recordSizeBytes = j.at("record_size_bytes").get<uint64_t>();
    const json &tensorJson = j.at("tensors");
    if (!tensorJson.is_object()) {
        throw std::runtime_error("LocalNamedExampleLayout manifest tensors field must be an object.");
    }

    std::vector<TensorSpec> specs;
    specs.reserve(tensorJson.size());
    for (const auto &item : tensorJson.items()) {
        const std::string name = item.key();
        const json &specJson = item.value();
        specs.push_back(TensorSpec{.name = name,
                                   .dataType = dataType,
                                   .dimensions = specJson.at("shape").get<std::vector<uint64_t>>(),
                                   .offsetBytes = specJson.at("offset_bytes").get<uint64_t>(),
                                   .numBytes = specJson.at("num_bytes").get<uint64_t>()});
    }

    std::vector<WindowedTensorSpec> windowedSpecs;
    if (j.contains("windowed_tensors")) {
        const json &windowedTensorJson = j.at("windowed_tensors");
        if (!windowedTensorJson.is_object()) {
            throw std::runtime_error("LocalNamedExampleLayout manifest windowed_tensors field must be an object.");
        }
        windowedSpecs.reserve(windowedTensorJson.size());
        for (const auto &item : windowedTensorJson.items()) {
            const std::string name = item.key();
            const json &specJson = item.value();
            double padValue = 0.0;
            if (specJson.contains("pad")) {
                const json &padJson = specJson.at("pad");
                if (padJson.at("mode").get<std::string>() != "constant") {
                    throw std::runtime_error("LocalNamedExampleLayout windowed tensor '" + name + "' has unsupported pad mode.");
                }
                padValue = padJson.at("value").get<double>();
            }
            std::optional<std::string> maskName;
            if (specJson.contains("mask_name")) {
                maskName = specJson.at("mask_name").get<std::string>();
            }
            WindowedTensorSpec spec{.name = name,
                                    .dataType = specJson.contains("data_type")
                                                    ? dataTypeFromString(specJson.at("data_type").get<std::string>())
                                                    : dataType,
                                    .dimensions = specJson.at("shape").get<std::vector<uint64_t>>(),
                                    .keyDataType = dataTypeFromString(specJson.at("key_data_type").get<std::string>()),
                                    .indexDataType = dataTypeFromString(specJson.at("index_data_type").get<std::string>()),
                                    .padValue = padValue,
                                    .maskName = std::move(maskName),
                                    .referenceOffsetBytes = specJson.at("reference_offset_bytes").get<uint64_t>(),
                                    .referenceNumBytes = specJson.at("reference_num_bytes").get<uint64_t>()};
            if (specJson.contains("source_storage")) {
                const json &sourceJson = specJson.at("source_storage");
                spec.sourceFilename = sourceJson.at("file").get<std::string>();
                spec.sourceNumBytes = sourceJson.at("num_bytes").get<uint64_t>();
                const json &sequencesJson = sourceJson.at("sequences");
                if (!sequencesJson.is_array()) {
                    throw std::runtime_error("LocalNamedExampleLayout windowed tensor '" + name + "' source sequences field must be an array.");
                }
                spec.sourceSequences.reserve(sequencesJson.size());
                for (const json &sequenceJson : sequencesJson) {
                    spec.sourceSequences.push_back(WindowedTensorSourceSequence{
                        .keyHex = sequenceJson.at("key_hex").get<std::string>(),
                        .startIndex = sequenceJson.at("start_index").get<int64_t>(),
                        .endIndexExclusive = sequenceJson.at("end_index_exclusive").get<int64_t>(),
                        .offsetBytes = sequenceJson.at("offset_bytes").get<uint64_t>(),
                        .numSteps = sequenceJson.at("num_steps").get<uint64_t>(),
                        .numBytes = sequenceJson.at("num_bytes").get<uint64_t>()});
                }
            }
            windowedSpecs.push_back(std::move(spec));
        }
    }

    std::sort(specs.begin(), specs.end(), [](const TensorSpec &a, const TensorSpec &b) {
        if (a.offsetBytes != b.offsetBytes) {
            return a.offsetBytes < b.offsetBytes;
        }
        return a.name < b.name;
    });
    std::sort(windowedSpecs.begin(), windowedSpecs.end(), [](const WindowedTensorSpec &a, const WindowedTensorSpec &b) {
        if (a.referenceOffsetBytes != b.referenceOffsetBytes) {
            return a.referenceOffsetBytes < b.referenceOffsetBytes;
        }
        return a.name < b.name;
    });

    return LocalNamedExampleLayout(dataType, recordSizeBytes, std::move(specs), std::move(windowedSpecs));
}

void LocalNamedExampleLayout::writeManifest(const std::filesystem::path &path) const {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error("LocalNamedExampleLayout failed to open manifest for writing: " + path.string());
    }
    out << toJson().dump(2) << '\n';
    if (!out.good()) {
        throw std::runtime_error("LocalNamedExampleLayout failed while writing manifest: " + path.string());
    }
}

LocalNamedExampleLayout LocalNamedExampleLayout::readManifest(const std::filesystem::path &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("LocalNamedExampleLayout failed to open manifest for reading: " + path.string());
    }
    json j;
    in >> j;
    if (!in.good() && !in.eof()) {
        throw std::runtime_error("LocalNamedExampleLayout failed while reading manifest: " + path.string());
    }
    return fromJson(j);
}

LocalNamedExampleLayout LocalNamedExampleLayout::fromTensorShapes(const std::map<std::string, std::vector<uint64_t>> &tensors,
                                                                  DataType dataType) {
    std::vector<std::pair<std::string, std::vector<uint64_t>>> ordered;
    ordered.reserve(tensors.size());
    for (const auto &entry : tensors) {
        ordered.emplace_back(entry.first, entry.second);
    }
    return fromTensorShapes(ordered, dataType);
}

LocalNamedExampleLayout LocalNamedExampleLayout::fromTensorShapes(const std::vector<std::pair<std::string, std::vector<uint64_t>>> &tensors,
                                                                  DataType dataType) {
    return fromTensorShapes(tensors, {}, dataType);
}

LocalNamedExampleLayout LocalNamedExampleLayout::fromTensorShapes(
    const std::vector<std::pair<std::string, std::vector<uint64_t>>> &tensors,
    const std::vector<WindowedTensorShape> &windowedTensors,
    DataType dataType) {
    uint64_t offsetBytes = 0;
    std::vector<TensorSpec> specs;
    specs.reserve(tensors.size());
    for (const auto &entry : tensors) {
        const uint64_t numBytes = expectedNumBytes(entry.second, dataType, entry.first);
        specs.push_back(TensorSpec{.name = entry.first,
                                   .dataType = dataType,
                                   .dimensions = entry.second,
                                   .offsetBytes = offsetBytes,
                                   .numBytes = numBytes});
        offsetBytes = checkedAdd(offsetBytes, numBytes, "record size for tensor '" + entry.first + "'");
    }

    std::vector<WindowedTensorSpec> windowedSpecs;
    windowedSpecs.reserve(windowedTensors.size());
    for (const WindowedTensorShape &entry : windowedTensors) {
        const DataType tensorDataType = entry.dataType == DataType::FP32 ? dataType : entry.dataType;
        WindowedTensorSpec spec{.name = entry.name,
                                .dataType = tensorDataType,
                                .dimensions = entry.dimensions,
                                .keyDataType = entry.keyDataType,
                                .indexDataType = entry.indexDataType,
                                .padValue = entry.padValue,
                                .maskName = entry.maskName,
                                .referenceOffsetBytes = offsetBytes,
                                .referenceNumBytes = checkedAdd(checkedWholeByteElementSizeBytes(entry.keyDataType),
                                                                checkedWholeByteElementSizeBytes(entry.indexDataType),
                                                                "windowed tensor reference bytes for '" + entry.name + "'")};
        offsetBytes = checkedAdd(offsetBytes, spec.referenceNumBytes, "record size for windowed tensor reference '" + entry.name + "'");
        windowedSpecs.push_back(std::move(spec));
    }
    return LocalNamedExampleLayout(dataType, offsetBytes, std::move(specs), std::move(windowedSpecs));
}
