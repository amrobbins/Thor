#include "Utilities/Loaders/IndexedLocalNamedExampleReader.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Loaders/LocalNamedExampleDatasetWriter.h"
#include "Utilities/Loaders/Shard.h"
#include "Utilities/TarFile/UringDirect.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <deque>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
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

uint64_t runtimeIovMax() {
    const long value = ::sysconf(_SC_IOV_MAX);
    if (value <= 0) {
        return 1024;
    }
    return static_cast<uint64_t>(value);
}

unsigned checkedQueueDepthForUring(uint64_t queueDepth) {
    if (queueDepth == 0 || queueDepth > static_cast<uint64_t>(std::numeric_limits<unsigned>::max())) {
        throw std::runtime_error("IndexedLocalNamedExampleReader queue depth is outside unsigned range.");
    }
    return static_cast<unsigned>(queueDepth);
}

uint32_t checkedUint32(uint64_t value, const char *context) {
    if (value == 0 || value > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::runtime_error(std::string(context) + " is outside uint32_t range.");
    }
    return static_cast<uint32_t>(value);
}


}  // namespace

class IndexedLocalNamedExampleReader::Impl {
   public:
    struct ShardInfo {
        std::filesystem::path path;
        std::string filename;
        uint64_t globalStart = 0;
        uint64_t numExamples = 0;
        std::shared_ptr<Shard> shard;
    };

    struct DirectReadSpec {
        uint64_t sourceOffsetBytes = 0;
        uint64_t numBytes = 0;
    };

    std::filesystem::path datasetPath;
    LocalNamedExampleLayout layout;
    std::vector<ShardInfo> shards;
    std::vector<DirectReadSpec> directReadSpecs;
    std::map<std::string, uint64_t> tensorOrdinalByName;
    uint64_t numExamples = 0;

    static std::unique_ptr<Impl> openDataset(const std::filesystem::path &datasetPath,
                                             const LocalNamedExampleLayout &requestedLayout) {
        const std::filesystem::path manifestPath = datasetPath / LocalNamedExampleDatasetWriter::MANIFEST_FILENAME;
        std::ifstream in(manifestPath, std::ios::binary);
        if (!in.is_open()) {
            throw std::runtime_error("IndexedLocalNamedExampleReader failed to open manifest: " + manifestPath.string());
        }

        json manifest;
        in >> manifest;
        if (!in.good() && !in.eof()) {
            throw std::runtime_error("IndexedLocalNamedExampleReader failed while reading manifest: " + manifestPath.string());
        }

        if (!manifest.contains("storage_mode") ||
            LocalNamedExampleDatasetWriter::storageModeFromString(manifest.at("storage_mode").get<std::string>()) !=
                LocalNamedExampleDatasetWriter::StorageMode::INDEXED) {
            throw std::runtime_error("IndexedLocalNamedExampleReader requires an indexed local named example dataset.");
        }

        auto out = std::unique_ptr<Impl>(new Impl());
        out->datasetPath = datasetPath;
        out->layout = LocalNamedExampleLayout::fromJson(manifest);
        out->layout.validateRequestedLayoutExact(requestedLayout);
        out->numExamples = manifest.at("num_examples").get<uint64_t>();

        const json &shardsJson = manifest.at("shards");
        if (!shardsJson.is_array()) {
            throw std::runtime_error("IndexedLocalNamedExampleReader manifest shards field must be an array.");
        }

        out->shards.reserve(shardsJson.size());
        for (const json &shardJson : shardsJson) {
            ShardInfo info;
            info.filename = shardJson.at("file").get<std::string>();
            info.path = datasetPath / info.filename;
            info.globalStart = shardJson.at("global_start").get<uint64_t>();
            info.numExamples = shardJson.at("num_examples").get<uint64_t>();
            info.shard = std::make_shared<Shard>();
            info.shard->openShard(info.path.string());
            if (info.shard->getExampleSizeInBytes() != out->layout.recordSizeBytes()) {
                throw std::runtime_error("IndexedLocalNamedExampleReader shard record size does not match manifest layout for: " +
                                         info.path.string());
            }
            if (info.shard->getNumExamples(ExampleType::TRAIN) != info.numExamples) {
                throw std::runtime_error("IndexedLocalNamedExampleReader shard train example count does not match manifest for: " +
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
                throw std::runtime_error("IndexedLocalNamedExampleReader indexed shard global_start values are not contiguous.");
            }
            expectedGlobalStart = checkedAdd(expectedGlobalStart, info.numExamples, "IndexedLocalNamedExampleReader shard coverage");
        }
        if (expectedGlobalStart != out->numExamples) {
            throw std::runtime_error("IndexedLocalNamedExampleReader shard coverage does not match manifest num_examples.");
        }

        if (out->layout.tensors().size() > runtimeIovMax()) {
            throw std::runtime_error("IndexedLocalNamedExampleReader layout tensor count exceeds system IOV_MAX for readv.");
        }

        out->directReadSpecs.reserve(out->layout.tensors().size());
        uint64_t expectedTensorOffsetBytes = 0;
        uint64_t ordinal = 0;
        for (const LocalNamedExampleLayout::TensorSpec &spec : out->layout.tensors()) {
            if (spec.offsetBytes != expectedTensorOffsetBytes) {
                throw std::runtime_error("IndexedLocalNamedExampleReader readv direct-read layout requires dense contiguous tensor "
                                         "records in source-offset order.");
            }
            const auto [insertIt, inserted] = out->tensorOrdinalByName.emplace(spec.name, ordinal);
            (void)insertIt;
            THOR_THROW_IF_FALSE(inserted);
            out->directReadSpecs.push_back(DirectReadSpec{.sourceOffsetBytes = spec.offsetBytes, .numBytes = spec.numBytes});
            expectedTensorOffsetBytes = checkedAdd(expectedTensorOffsetBytes,
                                                   spec.numBytes,
                                                   "IndexedLocalNamedExampleReader direct-read tensor layout");
            ordinal += 1;
        }
        if (expectedTensorOffsetBytes != out->layout.recordSizeBytes()) {
            throw std::runtime_error("IndexedLocalNamedExampleReader readv direct-read tensor specs do not cover record_size_bytes.");
        }

        return out;
    }

    const ShardInfo &resolveShard(uint64_t globalExampleIndex, uint64_t &localExampleIndex) const {
        if (globalExampleIndex >= numExamples) {
            throw std::runtime_error("IndexedLocalNamedExampleReader global index outside dataset row count.");
        }
        const auto it = std::upper_bound(shards.begin(),
                                         shards.end(),
                                         globalExampleIndex,
                                         [](uint64_t index, const ShardInfo &info) { return index < info.globalStart; });
        if (it == shards.begin()) {
            throw std::runtime_error("IndexedLocalNamedExampleReader failed to resolve global index to a shard.");
        }
        const ShardInfo &info = *(it - 1);
        THOR_THROW_IF_FALSE(globalExampleIndex >= info.globalStart);
        localExampleIndex = globalExampleIndex - info.globalStart;
        THOR_THROW_IF_FALSE(localExampleIndex < info.numExamples);
        return info;
    }
};

class IndexedLocalNamedExampleReader::Session::Impl {
   public:
    struct IoContext {
        std::string filename;
        UringDirect io;
        std::vector<std::vector<iovec>> iovecSlots;
        std::deque<uint64_t> freeIovecSlots;
        std::deque<uint64_t> submittedIovecSlotsInOrder;
        uint64_t submittedNotCompleted = 0;

        IoContext(uint64_t queueDepth, uint64_t tensorCount)
            : io(checkedQueueDepthForUring(queueDepth)) {
            THOR_THROW_IF_FALSE(queueDepth > 0);
            THOR_THROW_IF_FALSE(tensorCount > 0);
            iovecSlots.resize(static_cast<size_t>(queueDepth));
            for (uint64_t slot = 0; slot < queueDepth; ++slot) {
                iovecSlots.at(static_cast<size_t>(slot)).resize(static_cast<size_t>(tensorCount));
                freeIovecSlots.push_back(slot);
            }
        }

        IoContext(const IoContext &) = delete;
        IoContext &operator=(const IoContext &) = delete;
        IoContext(IoContext &&) noexcept = default;
        IoContext &operator=(IoContext &&) noexcept = default;
    };

    std::shared_ptr<IndexedLocalNamedExampleReader> owner;
    std::map<uint64_t, IoContext> ioContextsByShardIndex;
    IndexedLocalNamedExampleReaderSessionStats stats;
    std::set<std::string> resolvedIoBackends;
    const uint64_t queueDepth;

    Impl(std::shared_ptr<IndexedLocalNamedExampleReader> owner, uint64_t queueDepth)
        : owner(std::move(owner)), queueDepth(std::max<uint64_t>(queueDepth, 1)) {
        THOR_THROW_IF_FALSE(this->owner != nullptr);
    }

    IoContext &contextFor(uint64_t shardIndex, const IndexedLocalNamedExampleReader::Impl::ShardInfo &shardInfo) {
        auto it = ioContextsByShardIndex.find(shardIndex);
        if (it != ioContextsByShardIndex.end()) {
            return it->second;
        }

        const uint64_t tensorCount = static_cast<uint64_t>(owner->impl->directReadSpecs.size());
        IoContext context(queueDepth, tensorCount);
        context.filename = shardInfo.path.string();
        context.io.registerCachedLoadFile(context.filename);
        resolvedIoBackends.insert(std::string(context.io.activeBackendName()) + "_readv");

        auto [insertIt, inserted] = ioContextsByShardIndex.emplace(shardIndex, std::move(context));
        THOR_THROW_IF_FALSE(inserted);
        return insertIt->second;
    }

    void drainOne(IoContext &context) {
        if (context.submittedNotCompleted == 0) {
            return;
        }

        UringDirect::Completion completion = context.io.waitCompletionInOrder();
        if (completion.responseCode < 0) {
            throw std::runtime_error("IndexedLocalNamedExampleReader async readv failed for shard '" + context.filename +
                                     "': " + std::strerror(-completion.responseCode));
        }

        if (context.submittedIovecSlotsInOrder.empty()) {
            throw std::runtime_error("IndexedLocalNamedExampleReader async readv completion without an iovec slot.");
        }

        const uint64_t iovecSlot = context.submittedIovecSlotsInOrder.front();
        context.submittedIovecSlotsInOrder.pop_front();
        context.freeIovecSlots.push_back(iovecSlot);
        --context.submittedNotCompleted;

        stats.readCallsCompleted += 1;
        stats.readBytesCompleted += static_cast<uint64_t>(completion.responseCode);
    }

    void readRecord(IoContext &context,
                    uint64_t fileOffsetBytes,
                    uint64_t batchSlot,
                    const std::vector<uint8_t *> &tensorBasePointers) {
        THOR_THROW_IF_FALSE(owner != nullptr);
        const IndexedLocalNamedExampleReader::Impl &reader = *owner->impl;
        const uint64_t tensorCount = static_cast<uint64_t>(reader.directReadSpecs.size());
        THOR_THROW_IF_FALSE(tensorBasePointers.size() == tensorCount);

        if (context.freeIovecSlots.empty()) {
            context.io.submit();
            drainOne(context);
        }
        if (context.freeIovecSlots.empty()) {
            throw std::runtime_error("IndexedLocalNamedExampleReader async readv iovec slot pool is empty after draining.");
        }

        const uint64_t iovecSlot = context.freeIovecSlots.front();
        context.freeIovecSlots.pop_front();

        std::vector<iovec> &iovecs = context.iovecSlots.at(static_cast<size_t>(iovecSlot));
        THOR_THROW_IF_FALSE(iovecs.size() == tensorCount);

        const IndexedLocalNamedExampleReader::Impl::DirectReadSpec *const directReadSpecs = reader.directReadSpecs.data();
        uint8_t *const *const basePointers = tensorBasePointers.data();
        iovec *const directReadIovecs = iovecs.data();

        for (uint64_t ordinal = 0; ordinal < tensorCount; ++ordinal) {
            const IndexedLocalNamedExampleReader::Impl::DirectReadSpec &spec = directReadSpecs[ordinal];
            uint8_t *const basePointer = basePointers[ordinal];
            if (basePointer == nullptr) {
                throw std::runtime_error("IndexedLocalNamedExampleReader::Session received a null destination tensor pointer.");
            }
            const uint64_t destinationOffset = checkedMul(batchSlot,
                                                          spec.numBytes,
                                                          "IndexedLocalNamedExampleReader destination tensor slot");
            directReadIovecs[ordinal].iov_base = basePointer + destinationOffset;
            directReadIovecs[ordinal].iov_len = static_cast<size_t>(spec.numBytes);
        }

        const uint32_t recordSizeBytes = checkedUint32(reader.layout.recordSizeBytes(),
                                                       "IndexedLocalNamedExampleReader record size");
        while (!context.io.submitReadvCached(iovecs.data(), static_cast<unsigned>(iovecs.size()), fileOffsetBytes, recordSizeBytes)) {
            context.io.submit();
            if (context.submittedNotCompleted == 0) {
                context.freeIovecSlots.push_front(iovecSlot);
                throw std::runtime_error("IndexedLocalNamedExampleReader async readv submission failed with no in-flight reads to drain.");
            }
            drainOne(context);
        }

        context.submittedIovecSlotsInOrder.push_back(iovecSlot);
        ++context.submittedNotCompleted;
        stats.readCallsSubmitted += 1;
        stats.readBytesSubmitted += reader.layout.recordSizeBytes();
    }

    void drain() {
        for (auto &[shardIndex, context] : ioContextsByShardIndex) {
            (void)shardIndex;
            context.io.submit();
            while (context.submittedNotCompleted != 0) {
                drainOne(context);
            }
        }
    }
};

IndexedLocalNamedExampleReader::IndexedLocalNamedExampleReader(std::unique_ptr<Impl> impl) : impl(std::move(impl)) {
    THOR_THROW_IF_FALSE(this->impl != nullptr);
}

IndexedLocalNamedExampleReader::~IndexedLocalNamedExampleReader() = default;

std::shared_ptr<IndexedLocalNamedExampleReader> IndexedLocalNamedExampleReader::openDataset(
    const std::filesystem::path &datasetPath, const LocalNamedExampleLayout &requestedLayout) {
    return std::shared_ptr<IndexedLocalNamedExampleReader>(
        new IndexedLocalNamedExampleReader(Impl::openDataset(datasetPath, requestedLayout)));
}

std::unique_ptr<IndexedLocalNamedExampleReader::Session> IndexedLocalNamedExampleReader::createSession(uint64_t queueDepth) {
    return std::unique_ptr<Session>(new Session(shared_from_this(), queueDepth));
}

const LocalNamedExampleLayout &IndexedLocalNamedExampleReader::getLayout() const { return impl->layout; }

uint64_t IndexedLocalNamedExampleReader::getNumExamples() const { return impl->numExamples; }

uint64_t IndexedLocalNamedExampleReader::getRecordSizeBytes() const { return impl->layout.recordSizeBytes(); }

uint64_t IndexedLocalNamedExampleReader::getTensorCount() const { return static_cast<uint64_t>(impl->layout.tensors().size()); }

uint64_t IndexedLocalNamedExampleReader::getLayoutTensorOrdinal(std::string_view tensorName) const {
    const auto it = impl->tensorOrdinalByName.find(std::string(tensorName));
    if (it == impl->tensorOrdinalByName.end()) {
        throw std::runtime_error("IndexedLocalNamedExampleReader tensor not found in layout: " + std::string(tensorName));
    }
    return it->second;
}

void IndexedLocalNamedExampleReader::validateGlobalIndex(uint64_t index, const char *context) const {
    if (index >= impl->numExamples) {
        throw std::runtime_error(std::string("IndexedLocalNamedExampleReader ") + (context == nullptr ? "" : context) +
                                 " index is outside dataset row count.");
    }
}

IndexedLocalNamedExampleReader::Session::Session(std::shared_ptr<IndexedLocalNamedExampleReader> reader, uint64_t queueDepth)
    : impl(std::make_unique<Impl>(std::move(reader), queueDepth)) {}

IndexedLocalNamedExampleReader::Session::~Session() = default;

void IndexedLocalNamedExampleReader::Session::loadExampleInto(uint64_t globalExampleIndex,
                                                              uint64_t batchSlot,
                                                              const std::vector<uint8_t *> &tensorBasePointers) {
    THOR_THROW_IF_FALSE(impl != nullptr);
    THOR_THROW_IF_FALSE(impl->owner != nullptr);
    const IndexedLocalNamedExampleReader::Impl &reader = *impl->owner->impl;
    if (tensorBasePointers.size() != reader.directReadSpecs.size()) {
        throw std::runtime_error("IndexedLocalNamedExampleReader::Session destination tensor count does not match layout tensor count.");
    }

    uint64_t localExampleIndex = 0;
    const IndexedLocalNamedExampleReader::Impl::ShardInfo &shardInfo = reader.resolveShard(globalExampleIndex, localExampleIndex);
    const uint64_t shardIndex = static_cast<uint64_t>(&shardInfo - reader.shards.data());
    IndexedLocalNamedExampleReader::Session::Impl::IoContext &context = impl->contextFor(shardIndex, shardInfo);

    ShardExampleReadRequest request = shardInfo.shard->getExampleReadRequest(ExampleType::TRAIN, localExampleIndex);
    if (request.numBytes != reader.layout.recordSizeBytes()) {
        throw std::runtime_error("IndexedLocalNamedExampleReader shard read request size does not match layout record size.");
    }
    impl->readRecord(context, request.fileOffsetBytes, batchSlot, tensorBasePointers);
}

void IndexedLocalNamedExampleReader::Session::drain() {
    THOR_THROW_IF_FALSE(impl != nullptr);
    impl->drain();
}

IndexedLocalNamedExampleReaderSessionStats IndexedLocalNamedExampleReader::Session::takeStats() {
    THOR_THROW_IF_FALSE(impl != nullptr);
    impl->drain();
    IndexedLocalNamedExampleReaderSessionStats out = impl->stats;
    out.resolvedIoBackends.assign(impl->resolvedIoBackends.begin(), impl->resolvedIoBackends.end());
    impl->stats = IndexedLocalNamedExampleReaderSessionStats();
    impl->resolvedIoBackends.clear();
    return out;
}
