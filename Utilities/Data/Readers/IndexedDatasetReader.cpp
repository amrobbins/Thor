#include "Utilities/Data/Readers/IndexedDatasetReader.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Api/Data/DatasetWriter.h"
#include "Utilities/Data/Storage/DatasetShard.h"
#include "Utilities/TarFile/UringDirect.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <fcntl.h>
#include <sys/uio.h>
#include <unistd.h>
#include <utility>

using json = nlohmann::json;

namespace {

uint64_t checkedAdd(uint64_t left, uint64_t right, const char *context) {
    if (left > std::numeric_limits<uint64_t>::max() - right) {
        throw std::runtime_error(std::string(context) + " overflow while adding byte offsets.");
    }
    return left + right;
}

uint64_t checkedMul(uint64_t left, uint64_t right, const char *context) {
    if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) {
        throw std::runtime_error(std::string(context) + " overflow while multiplying byte offsets.");
    }
    return left * right;
}

int64_t checkedAffineStart(int64_t base, int64_t stride, int64_t fieldOffset, uint64_t row, const std::string &name) {
    const __int128 value = static_cast<__int128>(base) +
                           static_cast<__int128>(row) * static_cast<__int128>(stride) +
                           static_cast<__int128>(fieldOffset);
    if (value < static_cast<__int128>(std::numeric_limits<int64_t>::min()) ||
        value > static_cast<__int128>(std::numeric_limits<int64_t>::max())) {
        throw std::runtime_error("IndexedDatasetReader affine window reference '" + name +
                                 "' overflows int64.");
    }
    return static_cast<int64_t>(value);
}

uint64_t runtimeIovMax() {
    const long value = ::sysconf(_SC_IOV_MAX);
    if (value <= 0) {
        return 1024;
    }
    return static_cast<uint64_t>(value);
}

unsigned checkedQueueDepthForUring(uint64_t queueDepth) {
    if (queueDepth == 0 || queueDepth > static_cast<uint64_t>(std::numeric_limits<unsigned>::max())) {
        throw std::runtime_error("IndexedDatasetReader queue depth is outside unsigned range.");
    }
    return static_cast<unsigned>(queueDepth);
}

uint32_t checkedUint32(uint64_t value, const char *context) {
    if (value == 0 || value > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::runtime_error(std::string(context) + " is outside uint32_t range.");
    }
    return static_cast<uint32_t>(value);
}

using SteadyClock = std::chrono::steady_clock;

bool diagnosticsTimingEnabled() {
    static const bool enabled = [] {
        const char *specific = std::getenv("THOR_INDEXED_DATASET_READER_DIAGNOSTICS");
        if (specific != nullptr && specific[0] != '\0') {
            return !(specific[0] == '0' && specific[1] == '\0');
        }
        const char *shared = std::getenv("THOR_TRAINING_QUEUE_DIAGNOSTICS");
        return shared != nullptr && shared[0] != '\0' && !(shared[0] == '0' && shared[1] == '\0');
    }();
    return enabled;
}

SteadyClock::time_point diagnosticNow() {
    return diagnosticsTimingEnabled() ? SteadyClock::now() : SteadyClock::time_point{};
}

uint64_t diagnosticElapsedNanoseconds(SteadyClock::time_point start) {
    if (!diagnosticsTimingEnabled()) {
        return 0;
    }
    const auto elapsed = SteadyClock::now() - start;
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
}

std::string bytesToHex(const void *data, uint64_t numBytes) {
    if (data == nullptr) {
        throw std::runtime_error("IndexedDatasetReader cannot hex encode a null byte pointer.");
    }
    const auto *bytes = static_cast<const uint8_t *>(data);
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (uint64_t i = 0; i < numBytes; ++i) {
        out << std::setw(2) << static_cast<unsigned>(bytes[i]);
    }
    return out.str();
}

uint64_t checkedElementSizeBytes(ThorImplementation::DataType dataType) {
    switch (dataType) {
        case ThorImplementation::DataType::BOOLEAN:
        case ThorImplementation::DataType::INT8:
        case ThorImplementation::DataType::UINT8:
        case ThorImplementation::DataType::FP8_E4M3:
        case ThorImplementation::DataType::FP8_E5M2:
            return 1;
        case ThorImplementation::DataType::FP16:
        case ThorImplementation::DataType::BF16:
        case ThorImplementation::DataType::INT16:
        case ThorImplementation::DataType::UINT16:
            return 2;
        case ThorImplementation::DataType::FP32:
        case ThorImplementation::DataType::INT32:
        case ThorImplementation::DataType::UINT32:
            return 4;
        case ThorImplementation::DataType::FP64:
        case ThorImplementation::DataType::INT64:
        case ThorImplementation::DataType::UINT64:
            return 8;
        default:
            break;
    }
    throw std::runtime_error("IndexedDatasetReader unsupported data type value: " +
                             std::to_string(static_cast<int>(dataType)));
}

template <typename T>
T readScalar(const uint8_t *data) {
    T out{};
    std::memcpy(&out, data, sizeof(T));
    return out;
}

int64_t readIndexScalar(const uint8_t *data, ThorImplementation::DataType dataType) {
    switch (dataType) {
        case ThorImplementation::DataType::INT8:
            return readScalar<int8_t>(data);
        case ThorImplementation::DataType::INT16:
            return readScalar<int16_t>(data);
        case ThorImplementation::DataType::INT32:
            return readScalar<int32_t>(data);
        case ThorImplementation::DataType::INT64:
            return readScalar<int64_t>(data);
        case ThorImplementation::DataType::UINT8:
            return readScalar<uint8_t>(data);
        case ThorImplementation::DataType::UINT16:
            return readScalar<uint16_t>(data);
        case ThorImplementation::DataType::UINT32:
            return readScalar<uint32_t>(data);
        case ThorImplementation::DataType::UINT64: {
            const uint64_t value = readScalar<uint64_t>(data);
            if (value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
                throw std::runtime_error("IndexedDatasetReader window start is outside int64_t range.");
            }
            return static_cast<int64_t>(value);
        }
        default:
            break;
    }
    throw std::runtime_error("IndexedDatasetReader window index dtype must be integer.");
}

template <typename T>
void fillTyped(uint8_t *destination, uint64_t count, double value) {
    T typedValue = static_cast<T>(value);
    T *typedDestination = reinterpret_cast<T *>(destination);
    for (uint64_t i = 0; i < count; ++i) {
        typedDestination[i] = typedValue;
    }
}

void fillConstant(uint8_t *destination, uint64_t numBytes, ThorImplementation::DataType dataType, double value) {
    if (destination == nullptr) {
        throw std::runtime_error("IndexedDatasetReader cannot pad into a null destination pointer.");
    }
    if (numBytes == 0) {
        return;
    }
    if (value == 0.0) {
        std::memset(destination, 0, static_cast<size_t>(numBytes));
        return;
    }
    const uint64_t elementSize = checkedElementSizeBytes(dataType);
    if ((numBytes % elementSize) != 0) {
        throw std::runtime_error("IndexedDatasetReader window padding byte count is not dtype aligned.");
    }
    const uint64_t count = numBytes / elementSize;
    switch (dataType) {
        case ThorImplementation::DataType::BOOLEAN:
            fillTyped<bool>(destination, count, value != 0.0 ? 1.0 : 0.0);
            return;
        case ThorImplementation::DataType::INT8:
            fillTyped<int8_t>(destination, count, value);
            return;
        case ThorImplementation::DataType::UINT8:
            fillTyped<uint8_t>(destination, count, value);
            return;
        case ThorImplementation::DataType::INT16:
            fillTyped<int16_t>(destination, count, value);
            return;
        case ThorImplementation::DataType::UINT16:
            fillTyped<uint16_t>(destination, count, value);
            return;
        case ThorImplementation::DataType::INT32:
            fillTyped<int32_t>(destination, count, value);
            return;
        case ThorImplementation::DataType::UINT32:
            fillTyped<uint32_t>(destination, count, value);
            return;
        case ThorImplementation::DataType::INT64:
            fillTyped<int64_t>(destination, count, value);
            return;
        case ThorImplementation::DataType::UINT64:
            fillTyped<uint64_t>(destination, count, value);
            return;
        case ThorImplementation::DataType::FP32:
            fillTyped<float>(destination, count, value);
            return;
        case ThorImplementation::DataType::FP64:
            fillTyped<double>(destination, count, value);
            return;
        default:
            break;
    }
    throw std::runtime_error(
        "IndexedDatasetReader non-zero window padding currently supports only whole-byte scalar CPU types except fp16/bf16/fp8.");
}

void checkedSignedWindowEnd(int64_t start, uint64_t length, const std::string &name) {
    if (length > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        throw std::runtime_error("IndexedDatasetReader windowed tensor '" + name + "' length is outside int64_t range.");
    }
    const int64_t signedLength = static_cast<int64_t>(length);
    if (start > std::numeric_limits<int64_t>::max() - signedLength) {
        throw std::runtime_error("IndexedDatasetReader windowed tensor '" + name + "' requested range overflows int64_t.");
    }
    (void)signedLength;
}

void preadExact(int fd, void *destination, uint64_t numBytes, uint64_t offsetBytes, const std::string &context) {
    if (destination == nullptr) {
        throw std::runtime_error(context + " has null destination.");
    }
    uint8_t *out = static_cast<uint8_t *>(destination);
    uint64_t remaining = numBytes;
    uint64_t offset = offsetBytes;
    while (remaining != 0) {
        const size_t request = remaining > static_cast<uint64_t>(std::numeric_limits<size_t>::max())
                                   ? std::numeric_limits<size_t>::max()
                                   : static_cast<size_t>(remaining);
        if (offset > static_cast<uint64_t>(std::numeric_limits<off_t>::max())) {
            throw std::runtime_error(context + " offset is outside off_t range.");
        }
        const ssize_t n = ::pread(fd, out, request, static_cast<off_t>(offset));
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw std::runtime_error(context + " failed: " + std::strerror(errno));
        }
        if (n == 0) {
            throw std::runtime_error(context + " ended before requested bytes were read.");
        }
        const uint64_t consumed = static_cast<uint64_t>(n);
        out += consumed;
        offset = checkedAdd(offset, consumed, context.c_str());
        remaining -= consumed;
    }
}

}  // namespace

class IndexedDatasetReader::Impl {
   public:
    struct ShardInfo {
        std::filesystem::path path;
        std::string filename;
        uint64_t globalStart = 0;
        uint64_t numExamples = 0;
        std::shared_ptr<DatasetShard> shard;
    };

    enum class RecordReadSpanKind {
        Tensor,
        WindowedReference,
    };

    struct RecordReadSpan {
        RecordReadSpanKind kind = RecordReadSpanKind::Tensor;
        uint64_t sourceOffsetBytes = 0;
        uint64_t numBytes = 0;
        uint64_t ordinal = 0;
    };

    struct WindowedSourceReadSpec {
        DatasetLayout::WindowedTensorSourceSpec spec;
        std::filesystem::path sourcePath;
        std::map<std::string, DatasetLayout::WindowedTensorSourceSequence> sequenceByKeyHex;
    };

    struct WindowedTensorReadSpec {
        DatasetLayout::WindowedTensorSpec spec;
        uint64_t sourceOrdinal = 0;
        uint64_t referenceBufferOffsetBytes = 0;
    };

    struct AffineWindowReference {
        std::string keyHex;
        int64_t base = 0;
        int64_t stride = 1;
        int64_t fieldOffset = 0;
    };

    struct AffineWindowReferenceSegment {
        uint64_t rowStart = 0;
        uint64_t count = 0;
        std::vector<AffineWindowReference> references;
    };

    std::filesystem::path datasetPath;
    DatasetLayout layout;
    std::vector<ShardInfo> shards;
    std::vector<RecordReadSpan> recordReadSpans;
    std::vector<DatasetLayout::TensorSpec> directTensorSpecs;
    std::vector<WindowedSourceReadSpec> windowedSourceReadSpecs;
    std::vector<WindowedTensorReadSpec> windowedReadSpecs;
    std::vector<AffineWindowReferenceSegment> affineWindowReferenceSegments;
    std::map<std::string, uint64_t> tensorOrdinalByName;
    std::map<std::string, uint64_t> windowedTensorOrdinalByName;
    uint64_t totalWindowedReferenceBytes = 0;
    uint64_t numExamples = 0;

    static std::unique_ptr<Impl> openDataset(const std::filesystem::path &datasetPath,
                                             const DatasetLayout *requestedLayout) {
        const std::filesystem::path manifestPath = datasetPath / DatasetWriter::MANIFEST_FILENAME;
        std::ifstream in(manifestPath, std::ios::binary);
        if (!in.is_open()) {
            throw std::runtime_error("IndexedDatasetReader failed to open manifest: " + manifestPath.string());
        }

        json manifest;
        in >> manifest;
        if (!in.good() && !in.eof()) {
            throw std::runtime_error("IndexedDatasetReader failed while reading manifest: " + manifestPath.string());
        }

        if (!manifest.contains("storage_mode")) {
            throw std::runtime_error(
                "IndexedDatasetReader rejected a legacy split dataset manifest without storage_mode. "
                "Rewrite the dataset with DatasetWriter and provide splits through DatasetSplitManifest.");
        }
        const std::string storageMode = manifest.at("storage_mode").get<std::string>();
        if (storageMode != DatasetWriter::STORAGE_MODE_INDEXED) {
            throw std::runtime_error(
                "IndexedDatasetReader rejected legacy dataset storage_mode='" + storageMode +
                "'. Rewrite the dataset with DatasetWriter and provide splits through DatasetSplitManifest.");
        }

        auto out = std::unique_ptr<Impl>(new Impl());
        out->datasetPath = datasetPath;
        out->layout = DatasetLayout::fromJson(manifest);
        if (requestedLayout != nullptr) {
            out->layout.validateRequestedLayoutExact(*requestedLayout);
        }
        out->numExamples = manifest.at("num_examples").get<uint64_t>();

        const json &shardsJson = manifest.at("shards");
        if (!shardsJson.is_array()) {
            throw std::runtime_error("IndexedDatasetReader manifest shards field must be an array.");
        }

        out->shards.reserve(shardsJson.size());
        for (const json &shardJson : shardsJson) {
            ShardInfo info;
            info.filename = shardJson.at("file").get<std::string>();
            info.path = datasetPath / info.filename;
            info.globalStart = shardJson.at("global_start").get<uint64_t>();
            info.numExamples = shardJson.at("num_examples").get<uint64_t>();
            info.shard = std::make_shared<DatasetShard>();
            info.shard->openShard(info.path.string());
            if (info.shard->getExampleSizeInBytes() != out->layout.recordSizeBytes()) {
                throw std::runtime_error("IndexedDatasetReader shard record size does not match manifest layout for: " +
                                         info.path.string());
            }
            if (info.shard->getNumExamples(ExampleType::TRAIN) != info.numExamples) {
                throw std::runtime_error("IndexedDatasetReader shard train example count does not match manifest for: " +
                                         info.path.string());
            }
            out->shards.push_back(std::move(info));
        }

        std::sort(out->shards.begin(), out->shards.end(), [](const ShardInfo &left, const ShardInfo &right) {
            return left.globalStart < right.globalStart;
        });

        uint64_t expectedGlobalStart = 0;
        for (const ShardInfo &info : out->shards) {
            if (info.globalStart != expectedGlobalStart) {
                throw std::runtime_error("IndexedDatasetReader indexed shard global_start values are not contiguous.");
            }
            expectedGlobalStart = checkedAdd(expectedGlobalStart, info.numExamples, "IndexedDatasetReader shard coverage");
        }
        if (out->layout.recordSizeBytes() == 0) {
            if (!out->shards.empty()) {
                throw std::runtime_error("IndexedDatasetReader zero-record affine datasets must not contain shards.");
            }
        } else if (expectedGlobalStart != out->numExamples) {
            throw std::runtime_error("IndexedDatasetReader shard coverage does not match manifest num_examples.");
        }

        out->directTensorSpecs.reserve(out->layout.tensors().size());
        uint64_t ordinal = 0;
        for (const DatasetLayout::TensorSpec &spec : out->layout.tensors()) {
            const auto [insertIt, inserted] = out->tensorOrdinalByName.emplace(spec.name, ordinal);
            (void)insertIt;
            THOR_THROW_IF_FALSE(inserted);
            out->directTensorSpecs.push_back(spec);
            out->recordReadSpans.push_back(RecordReadSpan{.kind = RecordReadSpanKind::Tensor,
                                                          .sourceOffsetBytes = spec.offsetBytes,
                                                          .numBytes = spec.numBytes,
                                                          .ordinal = ordinal});
            ordinal += 1;
        }

        std::map<std::string, uint64_t> windowedSourceOrdinalByName;
        uint64_t sourceOrdinal = 0;
        for (const DatasetLayout::WindowedTensorSourceSpec &source : out->layout.windowedTensorSources()) {
            if (!source.sourceFilename.has_value()) {
                throw std::runtime_error("IndexedDatasetReader window source '" + source.name + "' has no storage.");
            }
            if (source.sourceSequences.empty()) {
                throw std::runtime_error("IndexedDatasetReader window source '" + source.name + "' has no sequences.");
            }
            WindowedSourceReadSpec readSpec;
            readSpec.spec = source;
            readSpec.sourcePath = datasetPath / source.sourceFilename.value();
            for (const DatasetLayout::WindowedTensorSourceSequence &sequence : source.sourceSequences) {
                const auto [sequenceIt, sequenceInserted] = readSpec.sequenceByKeyHex.emplace(sequence.keyHex, sequence);
                (void)sequenceIt;
                if (!sequenceInserted) {
                    throw std::runtime_error("IndexedDatasetReader window source '" + source.name +
                                             "' has duplicate key in manifest.");
                }
            }
            windowedSourceOrdinalByName.emplace(source.name, sourceOrdinal++);
            out->windowedSourceReadSpecs.push_back(std::move(readSpec));
        }

        uint64_t windowedOrdinal = 0;
        uint64_t referenceBufferOffsetBytes = 0;
        for (const DatasetLayout::WindowedTensorSpec &spec : out->layout.windowedTensors()) {
            const auto sourceIt = windowedSourceOrdinalByName.find(spec.sourceName);
            if (sourceIt == windowedSourceOrdinalByName.end()) {
                throw std::runtime_error("IndexedDatasetReader windowed tensor '" + spec.name +
                                         "' references an unknown source.");
            }
            WindowedTensorReadSpec readSpec;
            readSpec.spec = spec;
            readSpec.sourceOrdinal = sourceIt->second;
            readSpec.referenceBufferOffsetBytes = referenceBufferOffsetBytes;
            const auto [insertIt, inserted] = out->windowedTensorOrdinalByName.emplace(spec.name, windowedOrdinal);
            (void)insertIt;
            THOR_THROW_IF_FALSE(inserted);
            out->windowedReadSpecs.push_back(std::move(readSpec));
            if (spec.referenceMode == DatasetLayout::WindowedTensorReferenceMode::INDEXED) {
                out->recordReadSpans.push_back(RecordReadSpan{.kind = RecordReadSpanKind::WindowedReference,
                                                              .sourceOffsetBytes = spec.referenceOffsetBytes,
                                                              .numBytes = spec.referenceNumBytes,
                                                              .ordinal = windowedOrdinal});
                referenceBufferOffsetBytes = checkedAdd(referenceBufferOffsetBytes,
                                                         spec.referenceNumBytes,
                                                         "IndexedDatasetReader windowed reference buffer");
            }
            windowedOrdinal += 1;
        }
        out->totalWindowedReferenceBytes = referenceBufferOffsetBytes;

        if (out->recordReadSpans.size() > runtimeIovMax()) {
            throw std::runtime_error("IndexedDatasetReader layout record span count exceeds system IOV_MAX for readv.");
        }

        if (out->layout.hasAffineWindowedTensors()) {
            if (!manifest.contains("affine_window_reference_segments") ||
                !manifest.at("affine_window_reference_segments").is_array()) {
                throw std::runtime_error("IndexedDatasetReader affine dataset is missing reference segments.");
            }
            uint64_t expectedRowStart = 0;
            for (const json &segmentJson : manifest.at("affine_window_reference_segments")) {
                AffineWindowReferenceSegment segment;
                segment.rowStart = segmentJson.at("row_start").get<uint64_t>();
                segment.count = segmentJson.at("count").get<uint64_t>();
                if (segment.rowStart != expectedRowStart || segment.count == 0) {
                    throw std::runtime_error("IndexedDatasetReader affine reference segments must be contiguous and non-empty.");
                }
                const json &references = segmentJson.at("references");
                if (!references.is_object() || references.size() != out->windowedReadSpecs.size()) {
                    throw std::runtime_error("IndexedDatasetReader affine segment reference set does not match layout.");
                }
                segment.references.reserve(out->windowedReadSpecs.size());
                for (const WindowedTensorReadSpec &readSpec : out->windowedReadSpecs) {
                    if (!references.contains(readSpec.spec.name)) {
                        throw std::runtime_error("IndexedDatasetReader affine segment is missing field '" +
                                                 readSpec.spec.name + "'.");
                    }
                    const json &reference = references.at(readSpec.spec.name);
                    AffineWindowReference parsed{.keyHex = reference.at("key_hex").get<std::string>(),
                                                 .base = reference.at("base").get<int64_t>(),
                                                 .stride = reference.at("stride").get<int64_t>(),
                                                 .fieldOffset = reference.at("field_offset").get<int64_t>()};
                    if (parsed.stride <= 0) {
                        throw std::runtime_error("IndexedDatasetReader affine reference stride must be >= 1.");
                    }
                    (void)checkedAffineStart(parsed.base, parsed.stride, parsed.fieldOffset, 0, readSpec.spec.name);
                    (void)checkedAffineStart(
                        parsed.base, parsed.stride, parsed.fieldOffset, segment.count - 1, readSpec.spec.name);
                    const WindowedSourceReadSpec &source =
                        out->windowedSourceReadSpecs.at(static_cast<size_t>(readSpec.sourceOrdinal));
                    if (!source.sequenceByKeyHex.contains(parsed.keyHex)) {
                        throw std::runtime_error("IndexedDatasetReader affine reference key has no source sequence for '" +
                                                 readSpec.spec.name + "'.");
                    }
                    segment.references.push_back(std::move(parsed));
                }
                out->affineWindowReferenceSegments.push_back(std::move(segment));
                expectedRowStart = checkedAdd(expectedRowStart,
                                              out->affineWindowReferenceSegments.back().count,
                                              "IndexedDatasetReader affine segment coverage");
            }
            if (expectedRowStart != out->numExamples) {
                throw std::runtime_error("IndexedDatasetReader affine reference segments do not cover dataset rows.");
            }
        } else if (manifest.contains("affine_window_reference_segments")) {
            throw std::runtime_error("IndexedDatasetReader non-affine dataset contains affine reference segments.");
        }

        std::sort(out->recordReadSpans.begin(), out->recordReadSpans.end(), [](const RecordReadSpan &left, const RecordReadSpan &right) {
            if (left.sourceOffsetBytes != right.sourceOffsetBytes) {
                return left.sourceOffsetBytes < right.sourceOffsetBytes;
            }
            return static_cast<int>(left.kind) < static_cast<int>(right.kind);
        });

        uint64_t expectedRecordOffsetBytes = 0;
        for (const RecordReadSpan &span : out->recordReadSpans) {
            if (span.sourceOffsetBytes != expectedRecordOffsetBytes) {
                throw std::runtime_error("IndexedDatasetReader readv layout requires dense contiguous tensor/reference records "
                                         "in source-offset order.");
            }
            expectedRecordOffsetBytes = checkedAdd(expectedRecordOffsetBytes,
                                                   span.numBytes,
                                                   "IndexedDatasetReader direct-read record layout");
        }
        if (expectedRecordOffsetBytes != out->layout.recordSizeBytes()) {
            throw std::runtime_error("IndexedDatasetReader readv tensor/reference specs do not cover record_size_bytes.");
        }

        return out;
    }

    const AffineWindowReferenceSegment &resolveAffineSegment(uint64_t globalExampleIndex) const {
        if (globalExampleIndex >= numExamples || affineWindowReferenceSegments.empty()) {
            throw std::runtime_error("IndexedDatasetReader failed to resolve affine dataset row.");
        }
        const auto it = std::upper_bound(
            affineWindowReferenceSegments.begin(),
            affineWindowReferenceSegments.end(),
            globalExampleIndex,
            [](uint64_t index, const AffineWindowReferenceSegment &segment) { return index < segment.rowStart; });
        if (it == affineWindowReferenceSegments.begin()) {
            throw std::runtime_error("IndexedDatasetReader failed to resolve affine dataset row.");
        }
        const AffineWindowReferenceSegment &segment = *(it - 1);
        if (globalExampleIndex - segment.rowStart >= segment.count) {
            throw std::runtime_error("IndexedDatasetReader affine segment does not cover dataset row.");
        }
        return segment;
    }

    const ShardInfo &resolveShard(uint64_t globalExampleIndex, uint64_t &localExampleIndex) const {
        if (globalExampleIndex >= numExamples) {
            throw std::runtime_error("IndexedDatasetReader global index outside dataset row count.");
        }
        const auto it = std::upper_bound(shards.begin(),
                                         shards.end(),
                                         globalExampleIndex,
                                         [](uint64_t index, const ShardInfo &info) { return index < info.globalStart; });
        if (it == shards.begin()) {
            throw std::runtime_error("IndexedDatasetReader failed to resolve global index to a shard.");
        }
        const ShardInfo &info = *(it - 1);
        THOR_THROW_IF_FALSE(globalExampleIndex >= info.globalStart);
        localExampleIndex = globalExampleIndex - info.globalStart;
        THOR_THROW_IF_FALSE(localExampleIndex < info.numExamples);
        return info;
    }
};

class IndexedDatasetReader::Session::Impl {
   public:
    struct PendingReadRequest {
        bool active = false;
        bool materializeWindowedTensors = false;
        uint64_t batchSlot = 0;
        uint64_t globalExampleIndex = 0;
        std::vector<uint8_t *> windowedTensorBasePointers;
        std::vector<uint8_t *> windowedMaskBasePointers;
    };

    struct IoContext {
        std::string filename;
        UringDirect io;
        std::vector<std::vector<iovec>> iovecSlots;
        std::vector<std::vector<uint8_t>> windowedReferenceBuffers;
        std::vector<PendingReadRequest> pendingRequests;
        std::deque<uint64_t> freeIovecSlots;
        std::deque<uint64_t> submittedIovecSlotsInOrder;
        uint64_t submittedNotCompleted = 0;

        IoContext(uint64_t queueDepth, uint64_t recordSpanCount, uint64_t windowedReferenceBytes)
            : io(checkedQueueDepthForUring(queueDepth)) {
            THOR_THROW_IF_FALSE(queueDepth > 0);
            THOR_THROW_IF_FALSE(recordSpanCount > 0);
            iovecSlots.resize(static_cast<size_t>(queueDepth));
            windowedReferenceBuffers.resize(static_cast<size_t>(queueDepth));
            pendingRequests.resize(static_cast<size_t>(queueDepth));
            for (uint64_t slot = 0; slot < queueDepth; ++slot) {
                iovecSlots.at(static_cast<size_t>(slot)).resize(static_cast<size_t>(recordSpanCount));
                windowedReferenceBuffers.at(static_cast<size_t>(slot)).resize(static_cast<size_t>(windowedReferenceBytes));
                freeIovecSlots.push_back(slot);
            }
        }

        IoContext(const IoContext &) = delete;
        IoContext &operator=(const IoContext &) = delete;
        IoContext(IoContext &&) noexcept = default;
        IoContext &operator=(IoContext &&) noexcept = default;
    };

    struct SourceFileContext {
        std::string filename;
        int fd = -1;

        SourceFileContext() = default;
        SourceFileContext(std::string filename, int fd) : filename(std::move(filename)), fd(fd) {}
        ~SourceFileContext() {
            if (fd >= 0) {
                ::close(fd);
                fd = -1;
            }
        }

        SourceFileContext(const SourceFileContext &) = delete;
        SourceFileContext &operator=(const SourceFileContext &) = delete;
        SourceFileContext(SourceFileContext &&other) noexcept : filename(std::move(other.filename)), fd(other.fd) { other.fd = -1; }
        SourceFileContext &operator=(SourceFileContext &&other) noexcept {
            if (this != &other) {
                if (fd >= 0) {
                    ::close(fd);
                }
                filename = std::move(other.filename);
                fd = other.fd;
                other.fd = -1;
            }
            return *this;
        }
    };

    std::shared_ptr<IndexedDatasetReader> owner;
    std::map<uint64_t, IoContext> ioContextsByShardIndex;
    std::map<uint64_t, SourceFileContext> sourceContextsBySourceOrdinal;
    IndexedDatasetReaderSessionStats stats;
    std::set<std::string> resolvedIoBackends;
    const uint64_t queueDepth;

    Impl(std::shared_ptr<IndexedDatasetReader> owner, uint64_t queueDepth)
        : owner(std::move(owner)), queueDepth(std::max<uint64_t>(queueDepth, 1)) {
        THOR_THROW_IF_FALSE(this->owner != nullptr);
    }

    IoContext &contextFor(uint64_t shardIndex, const IndexedDatasetReader::Impl::ShardInfo &shardInfo) {
        stats.shardContextLookupCalls += 1;
        auto it = ioContextsByShardIndex.find(shardIndex);
        if (it != ioContextsByShardIndex.end()) {
            stats.shardContextCacheHits += 1;
            return it->second;
        }

        stats.shardContextCacheMisses += 1;
        const uint64_t recordSpanCount = static_cast<uint64_t>(owner->impl->recordReadSpans.size());
        IoContext context(queueDepth, recordSpanCount, owner->impl->totalWindowedReferenceBytes);
        context.filename = shardInfo.path.string();
        context.io.registerCachedLoadFile(context.filename);
        resolvedIoBackends.insert(std::string(context.io.activeBackendName()) + "_readv");

        auto [insertIt, inserted] = ioContextsByShardIndex.emplace(shardIndex, std::move(context));
        THOR_THROW_IF_FALSE(inserted);
        stats.shardContextOpenCount += 1;
        stats.maxOpenShardContexts = std::max<uint64_t>(stats.maxOpenShardContexts,
                                                        static_cast<uint64_t>(ioContextsByShardIndex.size()));
        return insertIt->second;
    }

    SourceFileContext &sourceContextFor(uint64_t sourceOrdinal) {
        auto it = sourceContextsBySourceOrdinal.find(sourceOrdinal);
        if (it != sourceContextsBySourceOrdinal.end()) {
            return it->second;
        }
        THOR_THROW_IF_FALSE(owner != nullptr);
        const auto &source = owner->impl->windowedSourceReadSpecs.at(static_cast<size_t>(sourceOrdinal));
        const std::string filename = source.sourcePath.string();
        int fd = ::open(filename.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) {
            throw std::runtime_error("IndexedDatasetReader failed to open window source '" + filename +
                                     "': " + std::strerror(errno));
        }
        auto [insertIt, inserted] = sourceContextsBySourceOrdinal.emplace(sourceOrdinal, SourceFileContext(filename, fd));
        THOR_THROW_IF_FALSE(inserted);
        return insertIt->second;
    }

    void materializeWindowedTensorsForRow(
        uint64_t globalExampleIndex,
        uint64_t batchSlot,
        const std::vector<uint8_t *> &windowedTensorBasePointers,
        const std::vector<uint8_t *> &windowedMaskBasePointers,
        const std::vector<uint8_t> *referenceBuffer) {
        THOR_THROW_IF_FALSE(owner != nullptr);
        const IndexedDatasetReader::Impl &reader = *owner->impl;
        if (reader.windowedReadSpecs.empty()) {
            return;
        }
        if (windowedTensorBasePointers.size() != reader.windowedReadSpecs.size()) {
            throw std::runtime_error("IndexedDatasetReader windowed destination tensor count does not match layout.");
        }
        if (windowedMaskBasePointers.size() != reader.windowedReadSpecs.size()) {
            throw std::runtime_error("IndexedDatasetReader windowed mask destination count does not match layout.");
        }

        const IndexedDatasetReader::Impl::AffineWindowReferenceSegment *affineSegment = nullptr;
        uint64_t affineSegmentRow = 0;
        if (reader.layout.hasAffineWindowedTensors()) {
            affineSegment = &reader.resolveAffineSegment(globalExampleIndex);
            affineSegmentRow = globalExampleIndex - affineSegment->rowStart;
        }

        for (uint64_t ordinal = 0; ordinal < reader.windowedReadSpecs.size(); ++ordinal) {
            const IndexedDatasetReader::Impl::WindowedTensorReadSpec &readSpec =
                reader.windowedReadSpecs.at(static_cast<size_t>(ordinal));
            const DatasetLayout::WindowedTensorSpec &spec = readSpec.spec;
            const IndexedDatasetReader::Impl::WindowedSourceReadSpec &sourceReadSpec =
                reader.windowedSourceReadSpecs.at(static_cast<size_t>(readSpec.sourceOrdinal));
            uint8_t *const outputBase = windowedTensorBasePointers.at(static_cast<size_t>(ordinal));
            if (outputBase == nullptr) {
                throw std::runtime_error("IndexedDatasetReader received a null windowed tensor destination for: " + spec.name);
            }
            uint8_t *const maskBase = windowedMaskBasePointers.at(static_cast<size_t>(ordinal));
            if (spec.maskName.has_value() && maskBase == nullptr) {
                throw std::runtime_error("IndexedDatasetReader received a null windowed mask destination for: " + spec.name);
            }

            std::string keyHex;
            int64_t requestedStart = 0;
            if (spec.referenceMode == DatasetLayout::WindowedTensorReferenceMode::INDEXED) {
                if (referenceBuffer == nullptr ||
                    readSpec.referenceBufferOffsetBytes + spec.referenceNumBytes > referenceBuffer->size()) {
                    throw std::runtime_error("IndexedDatasetReader indexed window reference buffer is missing or truncated.");
                }
                const uint8_t *const reference = referenceBuffer->data() + readSpec.referenceBufferOffsetBytes;
                keyHex = bytesToHex(reference, spec.keyNumBytes());
                requestedStart = readIndexScalar(reference + spec.keyNumBytes(), spec.indexDataType);
            } else {
                THOR_THROW_IF_FALSE(affineSegment != nullptr);
                const IndexedDatasetReader::Impl::AffineWindowReference &reference =
                    affineSegment->references.at(static_cast<size_t>(ordinal));
                keyHex = reference.keyHex;
                requestedStart = checkedAffineStart(reference.base,
                                                    reference.stride,
                                                    reference.fieldOffset,
                                                    affineSegmentRow,
                                                    spec.name);
            }

            const uint64_t windowLength = spec.windowLength();
            checkedSignedWindowEnd(requestedStart, windowLength, spec.name);
            const int64_t requestedEnd = requestedStart + static_cast<int64_t>(windowLength);
            const uint64_t stepBytes = spec.sourceStepNumBytes();
            const uint64_t outputSlotBytes = spec.outputNumBytes();
            uint8_t *const outputSlot = outputBase + checkedMul(batchSlot,
                                                                 outputSlotBytes,
                                                                 "IndexedDatasetReader windowed tensor batch slot");
            fillConstant(outputSlot, outputSlotBytes, spec.dataType, spec.padValue);

            uint8_t *maskSlot = nullptr;
            if (spec.maskName.has_value()) {
                maskSlot = maskBase + checkedMul(batchSlot,
                                                 windowLength,
                                                 "IndexedDatasetReader windowed mask batch slot");
                std::memset(maskSlot, 0, static_cast<size_t>(windowLength));
            }

            const auto sequenceIt = sourceReadSpec.sequenceByKeyHex.find(keyHex);
            if (sequenceIt == sourceReadSpec.sequenceByKeyHex.end()) {
                throw std::runtime_error("IndexedDatasetReader windowed tensor '" + spec.name +
                                         "' reference key has no sequence in source '" + spec.sourceName + "'.");
            }
            const DatasetLayout::WindowedTensorSourceSequence &sequence = sequenceIt->second;
            const int64_t readStart = std::max<int64_t>(requestedStart, sequence.startIndex);
            const int64_t readEnd = std::min<int64_t>(requestedEnd, sequence.endIndexExclusive);
            if (readStart >= readEnd) {
                continue;
            }
            const uint64_t validSteps = static_cast<uint64_t>(readEnd - readStart);
            const uint64_t padLeftSteps = static_cast<uint64_t>(readStart - requestedStart);
            const uint64_t sourceStepOffset = static_cast<uint64_t>(readStart - sequence.startIndex);
            const uint64_t sourceOffset = checkedAdd(sequence.offsetBytes,
                                                     checkedMul(sourceStepOffset,
                                                                stepBytes,
                                                                "IndexedDatasetReader windowed source offset"),
                                                     "IndexedDatasetReader windowed source offset");
            const uint64_t destinationOffset = checkedMul(padLeftSteps,
                                                          stepBytes,
                                                          "IndexedDatasetReader windowed destination offset");
            const uint64_t readBytes = checkedMul(validSteps, stepBytes, "IndexedDatasetReader windowed source read bytes");
            SourceFileContext &sourceContext = sourceContextFor(readSpec.sourceOrdinal);
            preadExact(sourceContext.fd,
                       outputSlot + destinationOffset,
                       readBytes,
                       sourceOffset,
                       "IndexedDatasetReader windowed tensor source read for '" + spec.name + "'");
            stats.windowedSourceReadCalls += 1;
            stats.windowedSourceReadBytes += readBytes;
            if (maskSlot != nullptr) {
                std::memset(maskSlot + padLeftSteps, 1, static_cast<size_t>(validSteps));
            }
        }
    }

    void materializeWindowedTensors(IoContext &context, uint64_t iovecSlot) {
        PendingReadRequest &request = context.pendingRequests.at(static_cast<size_t>(iovecSlot));
        if (!request.active) {
            throw std::runtime_error("IndexedDatasetReader missing pending request for completed read.");
        }
        if (!request.materializeWindowedTensors) {
            return;
        }
        const std::vector<uint8_t> &referenceBuffer = context.windowedReferenceBuffers.at(static_cast<size_t>(iovecSlot));
        materializeWindowedTensorsForRow(request.globalExampleIndex,
                                         request.batchSlot,
                                         request.windowedTensorBasePointers,
                                         request.windowedMaskBasePointers,
                                         &referenceBuffer);
    }

    void drainOne(IoContext &context) {
        if (context.submittedNotCompleted == 0) {
            return;
        }

        stats.drainMaxInflightReads = std::max<uint64_t>(stats.drainMaxInflightReads, context.submittedNotCompleted);

        const SteadyClock::time_point waitStart = diagnosticNow();
        UringDirect::Completion completion = context.io.waitCompletionInOrder();
        stats.readvCompletionWaitCalls += 1;
        stats.readvCompletionWaitNanoseconds += diagnosticElapsedNanoseconds(waitStart);
        if (completion.responseCode < 0) {
            throw std::runtime_error("IndexedDatasetReader async readv failed for shard '" + context.filename +
                                     "': " + std::strerror(-completion.responseCode));
        }

        if (context.submittedIovecSlotsInOrder.empty()) {
            throw std::runtime_error("IndexedDatasetReader async readv completion without an iovec slot.");
        }

        const SteadyClock::time_point processStart = diagnosticNow();
        const uint64_t iovecSlot = context.submittedIovecSlotsInOrder.front();
        context.submittedIovecSlotsInOrder.pop_front();
        if (static_cast<uint64_t>(completion.responseCode) != owner->impl->layout.recordSizeBytes()) {
            throw std::runtime_error("IndexedDatasetReader async readv completed with an unexpected byte count for shard '" +
                                     context.filename + "'.");
        }
        materializeWindowedTensors(context, iovecSlot);
        context.pendingRequests.at(static_cast<size_t>(iovecSlot)) = PendingReadRequest{};
        context.freeIovecSlots.push_back(iovecSlot);
        --context.submittedNotCompleted;

        stats.readCallsCompleted += 1;
        stats.readBytesCompleted += static_cast<uint64_t>(completion.responseCode);
        stats.drainCompletions += 1;
        stats.drainCompletionProcessNanoseconds += diagnosticElapsedNanoseconds(processStart);
    }

    void readRecord(IoContext &context,
                    uint64_t fileOffsetBytes,
                    uint64_t globalExampleIndex,
                    uint64_t batchSlot,
                    const std::vector<uint8_t *> &tensorBasePointers,
                    const std::vector<uint8_t *> &windowedTensorBasePointers,
                    const std::vector<uint8_t *> &windowedMaskBasePointers,
                    bool materializeWindowedTensors) {
        THOR_THROW_IF_FALSE(owner != nullptr);
        const IndexedDatasetReader::Impl &reader = *owner->impl;
        const uint64_t tensorCount = static_cast<uint64_t>(reader.directTensorSpecs.size());
        const uint64_t windowedCount = static_cast<uint64_t>(reader.windowedReadSpecs.size());
        THOR_THROW_IF_FALSE(tensorBasePointers.size() == tensorCount);
        if (materializeWindowedTensors) {
            THOR_THROW_IF_FALSE(windowedTensorBasePointers.size() == windowedCount);
            THOR_THROW_IF_FALSE(windowedMaskBasePointers.size() == windowedCount);
        } else {
            THOR_THROW_IF_FALSE(windowedTensorBasePointers.empty());
            THOR_THROW_IF_FALSE(windowedMaskBasePointers.empty());
        }

        const SteadyClock::time_point slotAcquireStart = diagnosticNow();
        if (context.freeIovecSlots.empty()) {
            context.io.submit();
            drainOne(context);
        }
        if (context.freeIovecSlots.empty()) {
            throw std::runtime_error("IndexedDatasetReader async readv iovec slot pool is empty after draining.");
        }

        const uint64_t iovecSlot = context.freeIovecSlots.front();
        context.freeIovecSlots.pop_front();
        stats.iovecSlotAcquireNanoseconds += diagnosticElapsedNanoseconds(slotAcquireStart);

        std::vector<iovec> &iovecs = context.iovecSlots.at(static_cast<size_t>(iovecSlot));
        THOR_THROW_IF_FALSE(iovecs.size() == reader.recordReadSpans.size());
        PendingReadRequest &pendingRequest = context.pendingRequests.at(static_cast<size_t>(iovecSlot));
        THOR_THROW_IF_FALSE(!pendingRequest.active);
        pendingRequest.active = true;
        pendingRequest.materializeWindowedTensors = materializeWindowedTensors;
        pendingRequest.batchSlot = batchSlot;
        pendingRequest.globalExampleIndex = globalExampleIndex;
        pendingRequest.windowedTensorBasePointers = windowedTensorBasePointers;
        pendingRequest.windowedMaskBasePointers = windowedMaskBasePointers;

        const SteadyClock::time_point fillStart = diagnosticNow();
        iovec *const readIovecs = iovecs.data();
        std::vector<uint8_t> &referenceBuffer = context.windowedReferenceBuffers.at(static_cast<size_t>(iovecSlot));

        for (uint64_t spanOrdinal = 0; spanOrdinal < reader.recordReadSpans.size(); ++spanOrdinal) {
            const IndexedDatasetReader::Impl::RecordReadSpan &span = reader.recordReadSpans.at(static_cast<size_t>(spanOrdinal));
            if (span.kind == IndexedDatasetReader::Impl::RecordReadSpanKind::Tensor) {
                const DatasetLayout::TensorSpec &spec = reader.directTensorSpecs.at(static_cast<size_t>(span.ordinal));
                THOR_THROW_IF_FALSE(span.numBytes == spec.numBytes);
                uint8_t *const basePointer = tensorBasePointers.at(static_cast<size_t>(span.ordinal));
                if (basePointer == nullptr) {
                    throw std::runtime_error("IndexedDatasetReader::Session received a null destination tensor pointer.");
                }
                const uint64_t destinationOffset = checkedMul(batchSlot,
                                                              spec.numBytes,
                                                              "IndexedDatasetReader destination tensor slot");
                readIovecs[spanOrdinal].iov_base = basePointer + destinationOffset;
                readIovecs[spanOrdinal].iov_len = static_cast<size_t>(spec.numBytes);
            } else {
                const IndexedDatasetReader::Impl::WindowedTensorReadSpec &spec =
                    reader.windowedReadSpecs.at(static_cast<size_t>(span.ordinal));
                THOR_THROW_IF_FALSE(span.numBytes == spec.spec.referenceNumBytes);
                readIovecs[spanOrdinal].iov_base = referenceBuffer.data() + spec.referenceBufferOffsetBytes;
                readIovecs[spanOrdinal].iov_len = static_cast<size_t>(spec.spec.referenceNumBytes);
            }
        }
        stats.iovecFillNanoseconds += diagnosticElapsedNanoseconds(fillStart);

        const uint32_t recordSizeBytes = checkedUint32(reader.layout.recordSizeBytes(),
                                                       "IndexedDatasetReader record size");
        const SteadyClock::time_point submitStart = diagnosticNow();
        while (true) {
            const SteadyClock::time_point submitCallStart = diagnosticNow();
            const bool submitted = context.io.submitReadvCached(iovecs.data(),
                                                                static_cast<unsigned>(iovecs.size()),
                                                                fileOffsetBytes,
                                                                recordSizeBytes);
            stats.readvSubmitCallNanoseconds += diagnosticElapsedNanoseconds(submitCallStart);
            if (submitted) {
                break;
            }
            stats.readvSubmitBackpressureCount += 1;
            const SteadyClock::time_point backpressureStart = diagnosticNow();
            context.io.submit();
            if (context.submittedNotCompleted == 0) {
                context.pendingRequests.at(static_cast<size_t>(iovecSlot)) = PendingReadRequest{};
                context.freeIovecSlots.push_front(iovecSlot);
                throw std::runtime_error("IndexedDatasetReader async readv submission failed with no in-flight reads to drain.");
            }
            drainOne(context);
            stats.readvSubmitBackpressureNanoseconds += diagnosticElapsedNanoseconds(backpressureStart);
        }
        stats.readvSubmitNanoseconds += diagnosticElapsedNanoseconds(submitStart);

        context.submittedIovecSlotsInOrder.push_back(iovecSlot);
        ++context.submittedNotCompleted;
        stats.drainMaxInflightReads = std::max<uint64_t>(stats.drainMaxInflightReads, context.submittedNotCompleted);
        stats.readCallsSubmitted += 1;
        stats.readBytesSubmitted += reader.layout.recordSizeBytes();
    }

    void drain() {
        const SteadyClock::time_point drainStart = diagnosticNow();
        stats.drainCalls += 1;

        // Submit every active shard context first, then wait/recycle completions.
        // The previous implementation submitted and fully drained each shard context
        // one at a time, which serialized the rings when a batch touched multiple
        // shards.  The intended async shape is to make all shard rings visible to
        // the kernel before waiting for any one shard to finish.
        std::vector<IoContext *> activeContexts;
        activeContexts.reserve(ioContextsByShardIndex.size());
        for (auto &[shardIndex, context] : ioContextsByShardIndex) {
            (void)shardIndex;
            if (context.submittedNotCompleted == 0) {
                continue;
            }
            activeContexts.push_back(&context);
            stats.drainContextVisits += 1;
            stats.drainMaxInflightReads = std::max<uint64_t>(stats.drainMaxInflightReads, context.submittedNotCompleted);
        }

        for (IoContext *context : activeContexts) {
            THOR_THROW_IF_FALSE(context != nullptr);
            const SteadyClock::time_point submitStart = diagnosticNow();
            context->io.submit();
            stats.drainSubmitCalls += 1;
            stats.drainSubmitNanoseconds += diagnosticElapsedNanoseconds(submitStart);
        }

        const SteadyClock::time_point waitLoopStart = diagnosticNow();
        for (IoContext *context : activeContexts) {
            THOR_THROW_IF_FALSE(context != nullptr);
            while (context->submittedNotCompleted != 0) {
                drainOne(*context);
            }
        }
        stats.drainWaitLoopNanoseconds += diagnosticElapsedNanoseconds(waitLoopStart);
        stats.drainNanoseconds += diagnosticElapsedNanoseconds(drainStart);
    }
};

IndexedDatasetReader::IndexedDatasetReader(std::unique_ptr<Impl> impl) : impl(std::move(impl)) {
    THOR_THROW_IF_FALSE(this->impl != nullptr);
}

IndexedDatasetReader::~IndexedDatasetReader() = default;

std::shared_ptr<IndexedDatasetReader> IndexedDatasetReader::openDataset(
    const std::filesystem::path &datasetPath) {
    return std::shared_ptr<IndexedDatasetReader>(new IndexedDatasetReader(Impl::openDataset(datasetPath, nullptr)));
}

std::shared_ptr<IndexedDatasetReader> IndexedDatasetReader::openDataset(
    const std::filesystem::path &datasetPath, const DatasetLayout &requestedLayout) {
    return std::shared_ptr<IndexedDatasetReader>(
        new IndexedDatasetReader(Impl::openDataset(datasetPath, &requestedLayout)));
}

std::unique_ptr<IndexedDatasetReader::Session> IndexedDatasetReader::createSession(uint64_t queueDepth) {
    return std::unique_ptr<Session>(new Session(shared_from_this(), queueDepth));
}

const DatasetLayout &IndexedDatasetReader::getLayout() const { return impl->layout; }

uint64_t IndexedDatasetReader::getNumExamples() const { return impl->numExamples; }

uint64_t IndexedDatasetReader::getRecordSizeBytes() const { return impl->layout.recordSizeBytes(); }

uint64_t IndexedDatasetReader::getTensorCount() const { return static_cast<uint64_t>(impl->directTensorSpecs.size()); }

uint64_t IndexedDatasetReader::getWindowedTensorCount() const {
    return static_cast<uint64_t>(impl->windowedReadSpecs.size());
}

uint64_t IndexedDatasetReader::getLayoutTensorOrdinal(std::string_view tensorName) const {
    const auto it = impl->tensorOrdinalByName.find(std::string(tensorName));
    if (it == impl->tensorOrdinalByName.end()) {
        throw std::runtime_error("IndexedDatasetReader tensor not found in layout: " + std::string(tensorName));
    }
    return it->second;
}

uint64_t IndexedDatasetReader::getLayoutWindowedTensorOrdinal(std::string_view tensorName) const {
    const auto it = impl->windowedTensorOrdinalByName.find(std::string(tensorName));
    if (it == impl->windowedTensorOrdinalByName.end()) {
        throw std::runtime_error("IndexedDatasetReader windowed tensor not found in layout: " + std::string(tensorName));
    }
    return it->second;
}

void IndexedDatasetReader::validateGlobalIndex(uint64_t index, const char *context) const {
    if (index >= impl->numExamples) {
        throw std::runtime_error(std::string("IndexedDatasetReader ") + (context == nullptr ? "" : context) +
                                 " index is outside dataset row count.");
    }
}

IndexedDatasetReader::Session::Session(std::shared_ptr<IndexedDatasetReader> reader, uint64_t queueDepth)
    : impl(std::make_unique<Impl>(std::move(reader), queueDepth)) {}

IndexedDatasetReader::Session::~Session() = default;

void IndexedDatasetReader::Session::loadExampleInto(uint64_t globalExampleIndex,
                                                              uint64_t batchSlot,
                                                              const std::vector<uint8_t *> &tensorBasePointers) {
    THOR_THROW_IF_FALSE(impl != nullptr);
    THOR_THROW_IF_FALSE(impl->owner != nullptr);
    const IndexedDatasetReader::Impl &reader = *impl->owner->impl;
    if (!reader.windowedReadSpecs.empty()) {
        throw std::runtime_error(
            "IndexedDatasetReader::Session windowed layouts require windowed tensor destination pointers.");
    }
    loadExampleInto(globalExampleIndex, batchSlot, tensorBasePointers, {}, {});
}

void IndexedDatasetReader::Session::loadDirectExampleInto(uint64_t globalExampleIndex,
                                                                    uint64_t batchSlot,
                                                                    const std::vector<uint8_t *> &tensorBasePointers) {
    THOR_THROW_IF_FALSE(impl != nullptr);
    THOR_THROW_IF_FALSE(impl->owner != nullptr);
    const IndexedDatasetReader::Impl &reader = *impl->owner->impl;
    if (tensorBasePointers.size() != reader.directTensorSpecs.size()) {
        throw std::runtime_error("IndexedDatasetReader::Session direct destination tensor count does not match layout tensor count.");
    }

    const SteadyClock::time_point loadExampleStart = diagnosticNow();
    impl->stats.loadExampleCalls += 1;
    impl->owner->validateGlobalIndex(globalExampleIndex, "direct load");
    if (reader.directTensorSpecs.empty()) {
        impl->stats.loadExampleNanoseconds += diagnosticElapsedNanoseconds(loadExampleStart);
        return;
    }

    uint64_t localExampleIndex = 0;
    const SteadyClock::time_point resolveStart = diagnosticNow();
    const IndexedDatasetReader::Impl::ShardInfo &shardInfo = reader.resolveShard(globalExampleIndex, localExampleIndex);
    impl->stats.resolveShardNanoseconds += diagnosticElapsedNanoseconds(resolveStart);

    const uint64_t shardIndex = static_cast<uint64_t>(&shardInfo - reader.shards.data());
    const SteadyClock::time_point contextStart = diagnosticNow();
    IndexedDatasetReader::Session::Impl::IoContext &context = impl->contextFor(shardIndex, shardInfo);
    impl->stats.shardContextLookupNanoseconds += diagnosticElapsedNanoseconds(contextStart);

    const SteadyClock::time_point requestStart = diagnosticNow();
    DatasetShardReadRequest request = shardInfo.shard->getExampleReadRequest(ExampleType::TRAIN, localExampleIndex);
    impl->stats.shardReadRequestNanoseconds += diagnosticElapsedNanoseconds(requestStart);
    if (request.numBytes != reader.layout.recordSizeBytes()) {
        throw std::runtime_error("IndexedDatasetReader shard read request size does not match layout record size.");
    }
    impl->readRecord(context, request.fileOffsetBytes, globalExampleIndex, batchSlot, tensorBasePointers, {}, {}, false);
    impl->stats.loadExampleNanoseconds += diagnosticElapsedNanoseconds(loadExampleStart);
}

void IndexedDatasetReader::Session::loadExampleInto(uint64_t globalExampleIndex,
                                                              uint64_t batchSlot,
                                                              const std::vector<uint8_t *> &tensorBasePointers,
                                                              const std::vector<uint8_t *> &windowedTensorBasePointers,
                                                              const std::vector<uint8_t *> &windowedMaskBasePointers) {
    THOR_THROW_IF_FALSE(impl != nullptr);
    THOR_THROW_IF_FALSE(impl->owner != nullptr);
    const IndexedDatasetReader::Impl &reader = *impl->owner->impl;
    if (tensorBasePointers.size() != reader.directTensorSpecs.size()) {
        throw std::runtime_error("IndexedDatasetReader::Session destination tensor count does not match layout tensor count.");
    }
    if (windowedTensorBasePointers.size() != reader.windowedReadSpecs.size()) {
        throw std::runtime_error(
            "IndexedDatasetReader::Session windowed destination tensor count does not match layout windowed tensor count.");
    }
    if (windowedMaskBasePointers.size() != reader.windowedReadSpecs.size()) {
        throw std::runtime_error(
            "IndexedDatasetReader::Session windowed mask destination tensor count does not match layout windowed tensor count.");
    }

    const SteadyClock::time_point loadExampleStart = diagnosticNow();
    impl->stats.loadExampleCalls += 1;
    impl->owner->validateGlobalIndex(globalExampleIndex, "load");

    if (reader.recordReadSpans.empty()) {
        impl->materializeWindowedTensorsForRow(globalExampleIndex,
                                                batchSlot,
                                                windowedTensorBasePointers,
                                                windowedMaskBasePointers,
                                                nullptr);
        impl->stats.loadExampleNanoseconds += diagnosticElapsedNanoseconds(loadExampleStart);
        return;
    }

    uint64_t localExampleIndex = 0;
    const SteadyClock::time_point resolveStart = diagnosticNow();
    const IndexedDatasetReader::Impl::ShardInfo &shardInfo = reader.resolveShard(globalExampleIndex, localExampleIndex);
    impl->stats.resolveShardNanoseconds += diagnosticElapsedNanoseconds(resolveStart);

    const uint64_t shardIndex = static_cast<uint64_t>(&shardInfo - reader.shards.data());
    const SteadyClock::time_point contextStart = diagnosticNow();
    IndexedDatasetReader::Session::Impl::IoContext &context = impl->contextFor(shardIndex, shardInfo);
    impl->stats.shardContextLookupNanoseconds += diagnosticElapsedNanoseconds(contextStart);

    const SteadyClock::time_point requestStart = diagnosticNow();
    DatasetShardReadRequest request = shardInfo.shard->getExampleReadRequest(ExampleType::TRAIN, localExampleIndex);
    impl->stats.shardReadRequestNanoseconds += diagnosticElapsedNanoseconds(requestStart);
    if (request.numBytes != reader.layout.recordSizeBytes()) {
        throw std::runtime_error("IndexedDatasetReader shard read request size does not match layout record size.");
    }
    impl->readRecord(context,
                     request.fileOffsetBytes,
                     globalExampleIndex,
                     batchSlot,
                     tensorBasePointers,
                     windowedTensorBasePointers,
                     windowedMaskBasePointers,
                     true);
    impl->stats.loadExampleNanoseconds += diagnosticElapsedNanoseconds(loadExampleStart);
}

void IndexedDatasetReader::Session::drain() {
    THOR_THROW_IF_FALSE(impl != nullptr);
    impl->drain();
}

IndexedDatasetReaderSessionStats IndexedDatasetReader::Session::takeStats() {
    THOR_THROW_IF_FALSE(impl != nullptr);
    impl->drain();
    IndexedDatasetReaderSessionStats out = impl->stats;
    out.resolvedIoBackends.assign(impl->resolvedIoBackends.begin(), impl->resolvedIoBackends.end());
    impl->stats = IndexedDatasetReaderSessionStats();
    impl->resolvedIoBackends.clear();
    return out;
}
