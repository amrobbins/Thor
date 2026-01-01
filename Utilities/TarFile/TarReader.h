#pragma once

#include "Utilities/TarFile/TarArchive.h"

namespace thor_file {

// -------------------- TarReader (scan-on-open, index offsets) --------------------
//
// Uses libarchive to parse PAX correctly (x/g headers, long names, large sizes).
// Assumes uncompressed tar file on disk so offsets are meaningful and seekable.

struct ShardFd {
    std::string path;
    int fd = -1;
    uint64_t size = 0;
};

struct FileSliceFd {
    int fd = -1;  // owned by TarReader; valid while TarReader lives
    std::string shard_path;
    uint32_t shard_index = 0;
    uint64_t offset = 0;
    uint64_t size = 0;
};

inline uint64_t read_u64_le(const uint8_t b[8]) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i) {
        v = (v << 8) | static_cast<uint64_t>(b[i]);
    }
    return v;
}

inline std::string read_footer_json(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("TarReader::scan: failed to open shard: " + path);

    in.seekg(0, std::ios::end);
    const std::streamoff file_size_off = in.tellg();
    if (file_size_off < 0)
        throw std::runtime_error("TarReader::scan: tellg failed: " + path);
    const uint64_t file_size = static_cast<uint64_t>(file_size_off);

    if (file_size < kFooterSize) {
        throw std::runtime_error("TarReader::scan: file too small to contain footer: " + path);
    }

    // Read footer
    in.seekg(static_cast<std::streamoff>(file_size - kFooterSize), std::ios::beg);

    char magic[8];
    uint8_t len_bytes[8];
    in.read(magic, 8);
    in.read(reinterpret_cast<char*>(len_bytes), 8);
    if (!in)
        throw std::runtime_error("TarReader::scan: failed to read footer: " + path);

    if (std::memcmp(magic, kFooterMagic, 8) != 0) {
        throw std::runtime_error("TarReader::scan: footer magic mismatch (no index?): " + path);
    }

    const uint64_t json_len = read_u64_le(len_bytes);
    if (json_len == 0) {
        throw std::runtime_error("TarReader::scan: json_len is 0: " + path);
    }
    if (json_len > file_size - kFooterSize) {
        throw std::runtime_error("TarReader::scan: json_len exceeds file size: " + path);
    }

    const uint64_t json_start = file_size - kFooterSize - json_len;

    in.seekg(static_cast<std::streamoff>(json_start), std::ios::beg);
    std::string json(json_len, '\0');
    in.read(json.data(), static_cast<std::streamsize>(json_len));
    if (!in)
        throw std::runtime_error("TarReader::scan: failed to read json blob: " + path);

    return json;
}

inline void require(bool cond, const std::string& msg) {
    if (!cond)
        throw std::runtime_error("TarReader::scan: " + msg);
}

inline bool is_sha32(std::string_view s) {
    if (s.size() != 32)
        return false;
    for (char c : s) {
        const bool ok = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
        if (!ok)
            return false;
    }
    return true;
}

inline std::string canonicalize_index_json_for_compare(nlohmann::json j) {
    // Remove or normalize shard-specific fields so indexes compare equal across shards.
    if (j.is_object()) {
        j.erase("shard_index");
    }
    // dump() produces a stable representation for a given key order; nlohmann preserves insertion
    // order by default, but since you're generating these, that's fine. If you want stricter,
    // consider setting ordered_json in your project.
    return j.dump();
}

class TarReader {
   public:
    explicit TarReader(std::string tarPath);
    virtual ~TarReader();
    const std::unordered_map<std::string, EntryInfo>& entries() const;
    bool contains(std::string pathInTar) const;
    const EntryInfo& info(std::string pathInTar) const;
    uint64_t dataOffset(std::string pathInTar) const;
    // Random-access read of the whole file (works because archive is uncompressed and contiguous).
    void readFile(std::string pathInTar, void* mem, uint64_t fileSize) const;
    FileSliceFd getFileSliceFd(std::string pathInTar) const;

   private:
    void scan();

    // std::string tarPath;
    // std::unordered_map<std::string, EntryInfo> entries_;

    std::string prefix_;      // e.g. "path/to/MyModel"
    std::string archive_id_;  // 32-char hex-ish
    uint32_t num_shards_;
    std::unordered_map<std::string, EntryInfo> index_;
    std::vector<std::string> shard_paths_;
    std::vector<ShardFd> shard_fds_;
};

}  // namespace thor_file
