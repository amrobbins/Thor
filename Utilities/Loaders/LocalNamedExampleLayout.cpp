#include "Utilities/Loaders/LocalNamedExampleLayout.h"

#include <algorithm>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

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

uint64_t expectedNumBytes(const std::vector<uint64_t> &dimensions, DataType dataType, const std::string &name) {
    return checkedMul(elementCount(dimensions, name), checkedWholeByteElementSizeBytes(dataType), "bytes for tensor '" + name + "'");
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

}  // namespace

bool LocalNamedExampleLayout::TensorSpec::operator==(const TensorSpec &rhs) const {
    return name == rhs.name && dataType == rhs.dataType && dimensions == rhs.dimensions && offsetBytes == rhs.offsetBytes &&
           numBytes == rhs.numBytes;
}

LocalNamedExampleLayout::LocalNamedExampleLayout() : layoutDataType(DataType::FP32), layoutRecordSizeBytes(0) {}

LocalNamedExampleLayout::LocalNamedExampleLayout(DataType dataType, uint64_t recordSizeBytes, std::vector<TensorSpec> tensors)
    : layoutDataType(dataType), layoutRecordSizeBytes(recordSizeBytes), layoutTensors(std::move(tensors)) {
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

void LocalNamedExampleLayout::validate() const {
    if (layoutRecordSizeBytes == 0) {
        throw std::runtime_error("LocalNamedExampleLayout record_size_bytes must be non-zero.");
    }
    if (layoutTensors.empty()) {
        throw std::runtime_error("LocalNamedExampleLayout must contain at least one tensor.");
    }

    const uint64_t elementSizeBytes = checkedWholeByteElementSizeBytes(layoutDataType);
    std::set<std::string> names;
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
    }

    std::sort(sortedSpecs.begin(), sortedSpecs.end(), [](const TensorSpec *a, const TensorSpec *b) {
        if (a->offsetBytes != b->offsetBytes) {
            return a->offsetBytes < b->offsetBytes;
        }
        return a->name < b->name;
    });

    uint64_t maxEndBytes = 0;
    uint64_t previousEndBytes = 0;
    bool havePrevious = false;
    std::string previousName;
    for (const TensorSpec *spec : sortedSpecs) {
        const uint64_t end = spec->offsetBytes + spec->numBytes;
        if (havePrevious && spec->offsetBytes < previousEndBytes) {
            throw std::runtime_error("LocalNamedExampleLayout tensor '" + spec->name + "' overlaps tensor '" + previousName + "'.");
        }
        havePrevious = true;
        previousEndBytes = end;
        previousName = spec->name;
        maxEndBytes = std::max(maxEndBytes, end);
    }

    if (maxEndBytes != layoutRecordSizeBytes) {
        throw std::runtime_error("LocalNamedExampleLayout max tensor end " + std::to_string(maxEndBytes) +
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

    std::sort(specs.begin(), specs.end(), [](const TensorSpec &a, const TensorSpec &b) {
        if (a.offsetBytes != b.offsetBytes) {
            return a.offsetBytes < b.offsetBytes;
        }
        return a.name < b.name;
    });

    return LocalNamedExampleLayout(dataType, recordSizeBytes, std::move(specs));
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
    return LocalNamedExampleLayout(dataType, offsetBytes, std::move(specs));
}
