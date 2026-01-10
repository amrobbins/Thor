#include <gtest/gtest.h>

#include <nlohmann/json.hpp>
#include "Utilities/TarFile/Crc32c.h"
#include "Utilities/TarFile/TarWriter.h"  // adjust include path as needed

#include <unistd.h>
#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

namespace {

static constexpr std::array<char, 8> kMagic = {'T', 'H', 'O', 'R', 'I', 'D', 'X', '1'};
static constexpr uint64_t kFooterSize = 24;  // 8 magic + 8 len + 4 crc + 4 reserved

static uint64_t read_u64_le(const uint8_t b[8]) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i) {
        v = (v << 8) | static_cast<uint64_t>(b[i]);
    }
    return v;
}

static bool is_sha32_hex(const std::string& s) {
    if (s.size() != 32)
        return false;
    for (char c : s) {
        const bool ok = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
        if (!ok)
            return false;
    }
    return true;
}

static std::vector<uint8_t> read_payload_at(const fs::path& shard_path, uint64_t offset, uint64_t size) {
    std::ifstream in(shard_path, std::ios::binary);
    if (!in)
        throw std::runtime_error("failed to open shard for payload read: " + shard_path.string());

    in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!in)
        throw std::runtime_error("seek failed: " + shard_path.string());

    std::vector<uint8_t> buf(static_cast<size_t>(size));
    in.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
    if (!in)
        throw std::runtime_error("payload read failed: " + shard_path.string());

    return buf;
}

static std::string make_tmp_prefix(const std::string& stem) {
    // /tmp/<stem>_<pid>_<counter>
    static int counter = 0;
    ++counter;

    int pid = 0;
#if defined(__unix__) || defined(__APPLE__)
    pid = static_cast<int>(::getpid());
#endif
    char buf[256];
    std::snprintf(buf, sizeof(buf), "/tmp/%s_%d_%d", stem.c_str(), pid, counter);
    return std::string(buf);
}

static void cleanup_prefix_files(const std::string& prefix) {
    // remove prefix.thor.tar and prefix.000000.thor.tar ... prefix.000010.thor.tar (best effort)
    std::error_code ec;
    fs::remove(prefix + ".thor.tar", ec);

    for (int i = 0; i < 32; ++i) {
        char suffix[64];
        std::snprintf(suffix, sizeof(suffix), ".%06d.thor.tar", i);
        fs::remove(prefix + std::string(suffix), ec);
    }
}

static nlohmann::json canonicalize_for_compare(nlohmann::json j) {
    if (j.is_object())
        j.erase("shard_index");
    return j;
}

}  // namespace

// -------------------- Tests --------------------

static uint32_t read_u32_le(const uint8_t b[4]) {
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}

static nlohmann::json load_footer_index_json(const fs::path& shard_path) {
    std::ifstream in(shard_path, std::ios::binary);
    if (!in)
        throw std::runtime_error("failed to open shard: " + shard_path.string());

    in.seekg(0, std::ios::end);
    const std::streamoff end = in.tellg();
    if (end < static_cast<std::streamoff>(kFooterSize)) {
        throw std::runtime_error("shard too small to contain footer: " + shard_path.string());
    }
    const uint64_t file_size = static_cast<uint64_t>(end);

    in.seekg(static_cast<std::streamoff>(file_size - kFooterSize), std::ios::beg);

    char magic[8];
    uint8_t len_bytes[8];
    uint8_t crc_bytes[4];
    uint8_t reserved_bytes[4];

    in.read(magic, 8);
    in.read(reinterpret_cast<char*>(len_bytes), 8);
    in.read(reinterpret_cast<char*>(crc_bytes), 4);
    in.read(reinterpret_cast<char*>(reserved_bytes), 4);
    if (!in)
        throw std::runtime_error("failed reading footer: " + shard_path.string());

    if (std::memcmp(magic, kMagic.data(), 8) != 0) {
        throw std::runtime_error("footer magic mismatch (no index?): " + shard_path.string());
    }

    const uint64_t json_len = read_u64_le(len_bytes);
    if (json_len == 0 || json_len > file_size - kFooterSize) {
        throw std::runtime_error("invalid json_len in footer: " + shard_path.string());
    }

    const uint32_t index_crc_expected = read_u32_le(crc_bytes);

    const uint64_t json_start = file_size - kFooterSize - json_len;
    in.seekg(static_cast<std::streamoff>(json_start), std::ios::beg);

    std::string json_str(json_len, '\0');
    in.read(json_str.data(), static_cast<std::streamsize>(json_len));
    if (!in)
        throw std::runtime_error("failed reading json blob: " + shard_path.string());

    // Validate index CRC (IEEE)
    const uint32_t index_crc_got = thor_file::Crc32c::compute((uint8_t*)json_str.data(), json_str.size());
    if (index_crc_got != index_crc_expected) {
        throw std::runtime_error("index CRC mismatch in " + shard_path.string() + " expected=" + std::to_string(index_crc_expected) +
                                 " got=" + std::to_string(index_crc_got));
    }

    return nlohmann::json::parse(json_str);
}

TEST(TarWriter, SingleShard_WritesFooterIndexAndOffsetsWork) {
    const std::string prefix = make_tmp_prefix("TarWriterSingle");
    cleanup_prefix_files(prefix);

    const std::string tar_path_prefix = prefix;
    const std::filesystem::path archiveDir = std::filesystem::path(tar_path_prefix).remove_filename();
    const std::string archiveName = std::filesystem::path(tar_path_prefix).filename().string();
    const uint64_t shard_limit = 0;  // no rollover
    const bool overwrite = true;

    // Create tiny files
    const std::string hello = "hello from gtest\n";
    std::vector<uint8_t> blob(64 * 1024, 0xAB);

    // Precompute expected CRCs (IEEE) for validation
    const uint32_t hello_crc = thor_file::Crc32c::compute((uint8_t*)hello.data(), hello.size());
    const uint32_t blob_crc = thor_file::Crc32c::compute((uint8_t*)blob.data(), blob.size());

    {
        thor_file::TarWriter w(archiveName, archiveDir, overwrite, shard_limit);
        w.addArchiveFile("docs/hello.txt", hello.data(), hello.size(), 0644, time(nullptr));
        w.addArchiveFile("data/blob.bin", blob.data(), blob.size(), 0644, time(nullptr));
        w.finishArchive();
    }

    const fs::path shard0 = prefix + ".thor.tar";
    ASSERT_TRUE(fs::exists(shard0)) << shard0;

    // Should be single shard
    EXPECT_FALSE(fs::exists(prefix + ".000000.thor.tar"));

    // Load footer JSON (and validate its CRC)
    nlohmann::json j0;
    ASSERT_NO_THROW(j0 = load_footer_index_json(shard0));

    ASSERT_TRUE(j0.is_object());
    ASSERT_TRUE(j0.contains("archive_id"));
    ASSERT_TRUE(j0.contains("num_shards"));
    ASSERT_TRUE(j0.contains("shard_index"));
    ASSERT_TRUE(j0.contains("files"));

    EXPECT_TRUE(is_sha32_hex(j0["archive_id"].get<std::string>()));
    EXPECT_EQ(j0["num_shards"].get<uint32_t>(), 1u);
    EXPECT_EQ(j0["shard_index"].get<uint32_t>(), 0u);

    const auto& files = j0["files"];
    ASSERT_TRUE(files.is_object());
    ASSERT_TRUE(files.contains("docs/hello.txt"));
    ASSERT_TRUE(files.contains("data/blob.bin"));

    // Read back hello via offsets + validate per-entry CRC
    {
        const auto& e = files["docs/hello.txt"];
        ASSERT_EQ(e["shard"].get<uint32_t>(), 0u);
        const uint64_t off = e["data_offset"].get<uint64_t>();
        const uint64_t sz = e["size"].get<uint64_t>();
        ASSERT_TRUE(e.contains("crc"));
        EXPECT_EQ(e["crc"].get<uint32_t>(), hello_crc);

        auto payload = read_payload_at(shard0, off, sz);
        EXPECT_EQ(payload.size(), hello.size());
        std::string got(payload.begin(), payload.end());
        EXPECT_EQ(got, hello);

        // Verify CRC over the payload bytes too
        EXPECT_EQ(thor_file::Crc32c::compute((uint8_t*)payload.data(), payload.size()), hello_crc);
    }

    // Read back blob via offsets + validate per-entry CRC
    {
        const auto& e = files["data/blob.bin"];
        ASSERT_EQ(e["shard"].get<uint32_t>(), 0u);
        const uint64_t off = e["data_offset"].get<uint64_t>();
        const uint64_t sz = e["size"].get<uint64_t>();
        ASSERT_TRUE(e.contains("crc"));
        EXPECT_EQ(e["crc"].get<uint32_t>(), blob_crc);

        auto payload = read_payload_at(shard0, off, sz);
        ASSERT_EQ(payload.size(), blob.size());
        EXPECT_EQ(std::memcmp(payload.data(), blob.data(), blob.size()), 0);

        // Verify CRC over the payload bytes too
        EXPECT_EQ(thor_file::Crc32c::compute((uint8_t*)payload.data(), payload.size()), blob_crc);
    }

    cleanup_prefix_files(prefix);
}

TEST(TarWriter, MultiShard_RenamesAndIndexesMatchAcrossShards) {
    const std::string prefix = make_tmp_prefix("TarWriterMulti");
    cleanup_prefix_files(prefix);

    const std::string tar_path_prefix = prefix;
    const std::filesystem::path archiveDir = std::filesystem::path(tar_path_prefix).remove_filename();
    const std::string archiveName = std::filesystem::path(tar_path_prefix).filename().string();
    const bool overwrite = true;

    // Use a very small limit to *force* rollover on the second add_bytes()
    const uint64_t shard_limit = 1;

    // Two small files; second should trigger shard 1 creation
    const std::string a = "AAA";
    const std::string b = "BBB";

    const uint32_t a_crc = thor_file::Crc32c::compute((uint8_t*)a.data(), a.size());
    const uint32_t b_crc = thor_file::Crc32c::compute((uint8_t*)b.data(), b.size());

    {
        thor_file::TarWriter w(archiveName, archiveDir, overwrite, shard_limit);
        w.addArchiveFile("a.txt", a.data(), a.size(), 0644, time(nullptr));
        w.addArchiveFile("b.txt", b.data(), b.size(), 0644, time(nullptr));
        w.finishArchive();
    }

    const fs::path shard0 = prefix + ".000000.thor.tar";
    const fs::path shard1 = prefix + ".000001.thor.tar";

    // Your rule: shard 0 is initially prefix.thor.tar, but once shard 1 is created it renames to .000000
    ASSERT_TRUE(fs::exists(shard0)) << shard0;
    ASSERT_TRUE(fs::exists(shard1)) << shard1;
    EXPECT_FALSE(fs::exists(prefix + ".thor.tar")) << "expected prefix.thor.tar to be renamed away";

    nlohmann::json j0, j1;
    ASSERT_NO_THROW(j0 = load_footer_index_json(shard0));  // validates footer CRC
    ASSERT_NO_THROW(j1 = load_footer_index_json(shard1));  // validates footer CRC

    // Basic fields
    ASSERT_TRUE(j0.is_object());
    ASSERT_TRUE(j1.is_object());

    EXPECT_EQ(j0["num_shards"].get<uint32_t>(), 2u);
    EXPECT_EQ(j1["num_shards"].get<uint32_t>(), 2u);

    EXPECT_EQ(j0["shard_index"].get<uint32_t>(), 0u);
    EXPECT_EQ(j1["shard_index"].get<uint32_t>(), 1u);

    EXPECT_TRUE(is_sha32_hex(j0["archive_id"].get<std::string>()));
    EXPECT_EQ(j0["archive_id"].get<std::string>(), j1["archive_id"].get<std::string>());

    // Indexes should match except shard_index
    EXPECT_EQ(canonicalize_for_compare(j0), canonicalize_for_compare(j1));

    // Verify both files listed
    const auto& files0 = j0["files"];
    const auto& files1 = j1["files"];
    ASSERT_TRUE(files0.contains("a.txt"));
    ASSERT_TRUE(files0.contains("b.txt"));
    ASSERT_TRUE(files1.contains("a.txt"));
    ASSERT_TRUE(files1.contains("b.txt"));

    // Random-access read using the shard specified by the index + validate per-entry crc32c
    auto check_file = [&](const nlohmann::json& files, const std::string& path, const std::string& expected, uint32_t expected_crc) {
        const auto& e = files[path];
        const uint32_t shard = e["shard"].get<uint32_t>();
        const uint64_t off = e["data_offset"].get<uint64_t>();
        const uint64_t sz = e["size"].get<uint64_t>();

        ASSERT_TRUE(e.contains("crc")) << "missing crc for " << path;
        EXPECT_EQ(e["crc"].get<uint32_t>(), expected_crc) << "crc mismatch in index for " << path;

        fs::path shard_path = (shard == 0) ? shard0 : shard1;
        auto payload = read_payload_at(shard_path, off, sz);
        std::string got(payload.begin(), payload.end());
        EXPECT_EQ(got, expected);

        // Validate payload CRC too
        EXPECT_EQ(thor_file::Crc32c::compute((uint8_t*)payload.data(), payload.size()), expected_crc)
            << "payload crc mismatch for " << path;
    };

    // Check using shard0's index (they're identical except shard_index anyway)
    check_file(files0, "a.txt", a, a_crc);
    check_file(files0, "b.txt", b, b_crc);

    cleanup_prefix_files(prefix);
}
