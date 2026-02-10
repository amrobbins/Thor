#pragma once

#include <archive.h>
#include <archive_entry.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

namespace thor_file {

static constexpr char kFooterMagic[8] = {'T', 'H', 'O', 'R', 'I', 'D', 'X', '1'};
static constexpr size_t kFooterSize = 24;

struct EntryInfo {
    uint32_t archiveShard = 0;
    uint64_t fileDataOffset = 0;
    uint64_t tensorDataOffset = 0;
    uint64_t size = 0;
    uint32_t crc = 0;
};

struct ArchivePlanEntry {
    ThorImplementation::Tensor tensor;

    uint64_t tensorOffsetBytes = 0;
    uint64_t fileOffsetBytes = 0;
    uint64_t numBytes = 0;
    std::string pathInTar;
    uint32_t expectedCrc;
};

struct ArchiveShardPlan {
    ArchiveShardPlan() = default;
    explicit ArchiveShardPlan(const std::string& path) : archiveShardPath(path) {}

    std::string archiveShardPath;
    uint32_t shardNumber;
    uint64_t totalBytes = 0;
    std::vector<ArchivePlanEntry> entries;
};

struct ArchiveWorkerJobContext {
    ArchiveWorkerJobContext(std::unordered_map<std::string, std::vector<EntryInfo>>& archiveIndex,
                            std::mutex& archiveIndexMutex,
                            std::filesystem::path archiveDirectory)
        : archiveIndex(archiveIndex), archiveIndexMutex(archiveIndexMutex), archiveDirectory(archiveDirectory) {}

    std::unordered_map<std::string, std::vector<EntryInfo>>& archiveIndex;
    std::mutex& archiveIndexMutex;
    std::filesystem::path archiveDirectory;
};

struct ArchiveReaderContext {
    ArchiveReaderContext(std::filesystem::path archiveDirectory, std::string& errorMessage, std::mutex& mtx)
        : archiveDirectory(archiveDirectory), errorMessage(errorMessage), mtx(mtx) {}
    std::filesystem::path archiveDirectory;
    std::string& errorMessage;
    std::mutex& mtx;
};

inline void throwArchive(archive* a, const char* what) {
    const char* err = archive_error_string(a);
    throw std::runtime_error(std::string(what) + ": " + (err ? err : "(unknown libarchive error)"));
}

inline std::string cleanTarPath(const std::string& pathInTar) {
    namespace fs = std::filesystem;
    fs::path p(pathInTar);
    p = p.lexically_normal();

    // Reject absolute paths / rooted paths.
    if (p.is_absolute() || p.has_root_path() || p.has_root_name() || p.has_root_directory()) {
        throw std::runtime_error("tar path must be relative");
    }

    // Reject .. segments.
    for (const auto& part : p) {
        if (part == "..")
            throw std::runtime_error("tar path must not contain '..'");
    }

    if (p.empty() || p == ".")
        throw std::runtime_error("tar path is empty");

    return p.generic_string();  // forward slashes
}

inline std::string shard_filename(const std::string& prefix, uint32_t shard_idx) {
    // prefix: "path/to/MyModel" -> "path/to/MyModel.000000.thor.tar"
    char buf[64];
    std::snprintf(buf, sizeof(buf), ".%06u.thor.tar", shard_idx);
    return prefix + std::string(buf);
}

}  // namespace thor_file

static constexpr uint32_t fiveHundredMB = (uint32_t(1) << 29);
static constexpr uint32_t thirtyTwoK = (uint32_t(1) << 15);
static constexpr uint32_t fourKBMask = (uint32_t(1) << 12) - 1;
static constexpr uint32_t fourKBAligned = ~fourKBMask;
// 500MB for payload-ish, plus slack for prefix alignment and safety.
static constexpr uint32_t fiveHundredMBPlusTail = fiveHundredMB + thirtyTwoK;
