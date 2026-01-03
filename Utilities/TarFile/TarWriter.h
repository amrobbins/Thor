#pragma once

#include "Utilities/TarFile/TarArchive.h"

namespace thor_file {

void throwArchive(archive* a, const char* what);

// -------------------- TarWriter (PAX, uncompressed) --------------------

class TarWriter {
   public:
    explicit TarWriter(std::string tarPath, bool overwriteIfExists, uint64_t shard_payload_limit_bytes = 5e9);
    TarWriter(const TarWriter&) = delete;
    TarWriter& operator=(const TarWriter&) = delete;
    virtual ~TarWriter();

    void add_bytes(std::string path_in_tar, const void* data, size_t size, int permissions = 0644, std::time_t mtime = std::time(nullptr));

    void finishArchive();  // appends JSON index + footer to each shard

   protected:
    std::vector<uint8_t> generateId();

   private:
    void openShard_(uint32_t shard_index);
    void closeShard_(uint32_t shard_index);

    static la_ssize_t write_cb(archive*, void* client_data, const void* buffer, size_t length);
    static int close_cb(archive*, void* client_data);

    // e.g. "path/to/MyModel"
    std::string prefix_;
    uint64_t shard_payload_limit_ = 0;
    std::vector<ShardState> shards_;
    uint32_t cur_ = 0;

    // 32-char id like a "sha" (hex). Set this in your ctor (CSPRNG, etc.).
    std::string archive_id_;

    // global index (same across shards except shard_index in the JSON)
    std::unordered_map<std::string, EntryInfo> index_;

    bool finished_ = false;
};

static void write_u64_le(std::ostream& out, uint64_t v) {
    uint8_t b[8];
    for (int i = 0; i < 8; ++i) {
        b[i] = static_cast<uint8_t>((v >> (8 * i)) & 0xFF);
    }
    out.write(reinterpret_cast<const char*>(b), 8);
}

static void write_u32_le(std::ostream& out, uint32_t v) {
    uint8_t b[4];
    b[0] = static_cast<uint8_t>((v >> 0) & 0xFF);
    b[1] = static_cast<uint8_t>((v >> 8) & 0xFF);
    b[2] = static_cast<uint8_t>((v >> 16) & 0xFF);
    b[3] = static_cast<uint8_t>((v >> 24) & 0xFF);
    out.write(reinterpret_cast<const char*>(b), 4);
}

}  // namespace thor_file
