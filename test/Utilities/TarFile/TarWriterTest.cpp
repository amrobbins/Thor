#include <gtest/gtest.h>

#include <nlohmann/json.hpp>
#include "Utilities/TarFile/TarWriter.h"  // adjust include path as needed

#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace {

static constexpr std::array<char, 8> kMagic = {'T', 'H', 'O', 'R', 'I', 'D', 'X', '1'};
static constexpr uint64_t kFooterSize = 16;  // 8 magic + 8 json_len

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
    in.read(magic, 8);
    in.read(reinterpret_cast<char*>(len_bytes), 8);
    if (!in)
        throw std::runtime_error("failed reading footer: " + shard_path.string());

    if (std::memcmp(magic, kMagic.data(), 8) != 0) {
        throw std::runtime_error("footer magic mismatch (no index?): " + shard_path.string());
    }

    const uint64_t json_len = read_u64_le(len_bytes);
    if (json_len == 0 || json_len > file_size - kFooterSize) {
        throw std::runtime_error("invalid json_len in footer: " + shard_path.string());
    }

    const uint64_t json_start = file_size - kFooterSize - json_len;
    in.seekg(static_cast<std::streamoff>(json_start), std::ios::beg);

    std::string json_str(json_len, '\0');
    in.read(json_str.data(), static_cast<std::streamsize>(json_len));
    if (!in)
        throw std::runtime_error("failed reading json blob: " + shard_path.string());

    return nlohmann::json::parse(json_str);
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
    // remove prefix.thor and prefix.000000.thor ... prefix.000010.thor (best effort)
    std::error_code ec;
    fs::remove(prefix + ".thor", ec);

    for (int i = 0; i < 32; ++i) {
        char suffix[64];
        std::snprintf(suffix, sizeof(suffix), ".%06d.thor", i);
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

TEST(TarWriter, SingleShard_WritesFooterIndexAndOffsetsWork) {
    const std::string prefix = make_tmp_prefix("TarWriterSingle");
    cleanup_prefix_files(prefix);

    const std::string tar_path_prefix = prefix;  // your TarWriter ctor takes the prefix
    const uint64_t shard_limit = 0;              // 0 => no rollover
    const bool overwrite = true;

    // Create tiny files
    const std::string hello = "hello from gtest\n";
    std::vector<uint8_t> blob(64 * 1024, 0xAB);

    {
        thor_file::TarWriter w(tar_path_prefix, shard_limit, overwrite);
        w.add_bytes("docs/hello.txt", hello.data(), hello.size(), 0644, time(nullptr));
        w.add_bytes("data/blob.bin", blob.data(), blob.size(), 0644, time(nullptr));
        w.finishArchive();
    }

    const fs::path shard0 = prefix + ".thor";
    ASSERT_TRUE(fs::exists(shard0)) << shard0;

    // Should be single shard, so numbered shard may not exist
    EXPECT_FALSE(fs::exists(prefix + ".000000.thor"));

    // Load footer JSON
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

    // Read back hello via offsets
    {
        const auto& e = files["docs/hello.txt"];
        ASSERT_EQ(e["shard"].get<uint32_t>(), 0u);
        const uint64_t off = e["data_offset"].get<uint64_t>();
        const uint64_t sz = e["size"].get<uint64_t>();

        auto payload = read_payload_at(shard0, off, sz);
        std::string got(payload.begin(), payload.end());
        EXPECT_EQ(got, hello);
    }

    // Read back blob via offsets
    {
        const auto& e = files["data/blob.bin"];
        ASSERT_EQ(e["shard"].get<uint32_t>(), 0u);
        const uint64_t off = e["data_offset"].get<uint64_t>();
        const uint64_t sz = e["size"].get<uint64_t>();

        auto payload = read_payload_at(shard0, off, sz);
        ASSERT_EQ(payload.size(), blob.size());
        EXPECT_EQ(std::memcmp(payload.data(), blob.data(), blob.size()), 0);
    }

    cleanup_prefix_files(prefix);
}

TEST(TarWriter, MultiShard_RenamesAndIndexesMatchAcrossShards) {
    const std::string prefix = make_tmp_prefix("TarWriterMulti");
    cleanup_prefix_files(prefix);

    const std::string tar_path_prefix = prefix;
    const bool overwrite = true;

    // Use a very small limit to *force* rollover on the second add_bytes()
    const uint64_t shard_limit = 1;

    // Two small files; second should trigger shard 1 creation
    const std::string a = "AAA";
    const std::string b = "BBB";

    {
        thor_file::TarWriter w(tar_path_prefix, shard_limit, overwrite);
        w.add_bytes("a.txt", a.data(), a.size(), 0644, time(nullptr));
        w.add_bytes("b.txt", b.data(), b.size(), 0644, time(nullptr));
        w.finishArchive();
    }

    const fs::path shard0 = prefix + ".000000.thor";
    const fs::path shard1 = prefix + ".000001.thor";

    // Your rule: shard 0 is initially prefix.thor, but once shard 1 is created it renames to .000000
    ASSERT_TRUE(fs::exists(shard0)) << shard0;
    ASSERT_TRUE(fs::exists(shard1)) << shard1;
    EXPECT_FALSE(fs::exists(prefix + ".thor")) << "expected prefix.thor to be renamed away";

    nlohmann::json j0, j1;
    ASSERT_NO_THROW(j0 = load_footer_index_json(shard0));
    ASSERT_NO_THROW(j1 = load_footer_index_json(shard1));

    // Basic fields
    EXPECT_EQ(j0["num_shards"].get<uint32_t>(), 2u);
    EXPECT_EQ(j1["num_shards"].get<uint32_t>(), 2u);

    EXPECT_EQ(j0["shard_index"].get<uint32_t>(), 0u);
    EXPECT_EQ(j1["shard_index"].get<uint32_t>(), 1u);

    EXPECT_TRUE(is_sha32_hex(j0["archive_id"].get<std::string>()));
    EXPECT_EQ(j0["archive_id"].get<std::string>(), j1["archive_id"].get<std::string>());

    // Indexes should match except shard_index
    EXPECT_EQ(canonicalize_for_compare(j0), canonicalize_for_compare(j1));

    // Verify both files listed
    const auto& files = j0["files"];
    ASSERT_TRUE(files.contains("a.txt"));
    ASSERT_TRUE(files.contains("b.txt"));

    // Random-access read using the shard specified by the index
    auto check_file = [&](const std::string& path, const std::string& expected) {
        const auto& e = files[path];
        const uint32_t shard = e["shard"].get<uint32_t>();
        const uint64_t off = e["data_offset"].get<uint64_t>();
        const uint64_t sz = e["size"].get<uint64_t>();

        fs::path shard_path = (shard == 0) ? shard0 : shard1;
        auto payload = read_payload_at(shard_path, off, sz);
        std::string got(payload.begin(), payload.end());
        EXPECT_EQ(got, expected);
    };

    check_file("a.txt", a);
    check_file("b.txt", b);

    cleanup_prefix_files(prefix);
}
