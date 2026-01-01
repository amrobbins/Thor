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
static constexpr size_t kFooterSize = 16;

struct EntryInfo {
    uint32_t shard = 0;
    uint64_t data_offset = 0;
    uint64_t size = 0;
};

struct ShardState {
    std::string path;
    std::ofstream out;
    archive* a = nullptr;
    uint64_t pos = 0;
    bool closed = false;

    std::vector<char> io_buf;
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
    // prefix: "path/to/MyModel" -> "path/to/MyModel.000000.thor"
    char buf[64];
    std::snprintf(buf, sizeof(buf), ".%06u.thor", shard_idx);
    return prefix + std::string(buf);
}

}  // namespace thor_file
