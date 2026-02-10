#include <gtest/gtest.h>

#include "Utilities/TarFile/Crc32c.h"
#include "Utilities/TarFile/TarReader.h"
#include "Utilities/TarFile/TarWriter.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include <unistd.h>

namespace fs = std::filesystem;
using namespace thor_file;

static std::string makeTmpPrefix(const std::string& stem) {
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

void cleanup_prefix_files(const std::string& prefix) {
    std::error_code ec;
    fs::remove(prefix + ".thor.tar", ec);
    for (int i = 0; i < 64; ++i) {
        char suf[64];
        std::snprintf(suf, sizeof(suf), ".%06d.thor.tar", i);
        fs::remove(prefix + std::string(suf), ec);
    }
}

struct CleanupGuard {
    std::string prefix;
    ~CleanupGuard() { cleanup_prefix_files(prefix); }
};

std::vector<uint8_t> make_pattern_bytes(size_t n, uint32_t seed) {
    std::vector<uint8_t> v(n);
    // Simple deterministic pattern (fast, repeatable)
    uint32_t x = seed * 2654435761u + 12345u;
    for (size_t i = 0; i < n; ++i) {
        x ^= (x << 13);
        x ^= (x >> 17);
        x ^= (x << 5);
        v[i] = static_cast<uint8_t>(x & 0xFF);
    }
    return v;
}

void read_and_expect(TarReader& reader, const std::string& path, const std::vector<uint8_t>& expected) {
    ASSERT_TRUE(reader.containsFile(path)) << "missing: " << path;

    std::vector<EntryInfo> fileEntries = reader.getFileShards(path);
    // const auto& info = r.getFileShards(path);
    ASSERT_EQ(fileEntries.size(), 1UL);
    EntryInfo info = fileEntries[0];
    ASSERT_EQ(info.size, static_cast<uint64_t>(expected.size())) << "size mismatch for " << path;

    // std::vector<uint8_t> got(expected.size());
    // reader.readFile(path, got.data(), static_cast<uint64_t>(got.size()));

    ThorImplementation::TensorPlacement cpuPlacement(ThorImplementation::TensorPlacement::MemDevices::CPU);
    ThorImplementation::TensorDescriptor descriptor(ThorImplementation::TensorDescriptor::DataType::UINT8, {expected.size()});
    ThorImplementation::Tensor got(cpuPlacement, descriptor);
    reader.registerReadRequest(path, got);
    reader.executeReadRequests();

    ASSERT_EQ(got.getArraySizeInBytes(), expected.size());
    ASSERT_EQ(std::memcmp(got.getMemPtr<uint8_t>(), expected.data(), expected.size()), 0) << "payload mismatch for " << path;

    const uint32_t crc_expected = crc32_ieee(0xFFFFFFFF, (uint8_t*)expected.data(), expected.size());
    EXPECT_EQ(info.crc, crc_expected) << "index crc32c mismatch for " << path;
}

// -----------------------------------------------------------------------------
// Round trip: single shard
// -----------------------------------------------------------------------------
TEST(TarRoundTrip, SingleShard_CreateThenRead_VerifyBytes) {
    const std::string prefix = makeTmpPrefix("thor_tar_single");
    CleanupGuard guard{prefix};

    ThorImplementation::TensorPlacement cpuPlacement(ThorImplementation::TensorPlacement::MemDevices::CPU);

    // Build some files
    const std::string hello_str = "hello from gtest\n";
    ThorImplementation::TensorDescriptor helloDescriptor(ThorImplementation::TensorDescriptor::DataType::UINT8, {hello_str.size()});
    ThorImplementation::Tensor helloTensor(cpuPlacement, helloDescriptor);
    memcpy(helloTensor.getMemPtr<void>(), hello_str.data(), hello_str.size());

    uint32_t blobBytes = 256 * 1024;
    const std::vector<uint8_t> blob = make_pattern_bytes(blobBytes, 42);
    ThorImplementation::TensorDescriptor blobDescriptor(ThorImplementation::TensorDescriptor::DataType::UINT8, {blobBytes});
    ThorImplementation::Tensor blobTensor(cpuPlacement, blobDescriptor);
    memcpy(blobTensor.getMemPtr<void>(), blob.data(), blob.size());

    const std::filesystem::path archiveDir = std::filesystem::path(prefix).remove_filename();
    const std::string archiveName = std::filesystem::path(prefix).filename().string();

    // Write single shard (no rollover)
    {
        const uint64_t shard_limit = 1'000'000;
        const bool overwrite = true;

        TarWriter w(archiveName, shard_limit);
        w.addArchiveFile("docs/hello.txt", helloTensor);
        w.addArchiveFile("data/blob.bin", blobTensor);
        w.createArchive(archiveDir, overwrite);
    }

    // Expect the single-file naming convention
    ASSERT_TRUE(fs::exists(prefix + ".thor.tar"));
    EXPECT_FALSE(fs::exists(prefix + ".000000.thor.tar"));

    // Read back and validate content
    TarReader r(archiveName, archiveDir);
    const std::vector<uint8_t> hello(hello_str.begin(), hello_str.end());
    read_and_expect(r, "docs/hello.txt", hello);
    read_and_expect(r, "data/blob.bin", blob);

    // A couple sanity checks about indexing behavior
    ASSERT_TRUE(r.containsFile("docs/hello.txt"));
    ASSERT_TRUE(r.containsFile("data/blob.bin"));
    ASSERT_FALSE(r.containsFile("nope.txt"));
}

// -----------------------------------------------------------------------------
// Round trip: multi shard + rename shard0 (.thor.tar -> .000000.thor.tar)
// -----------------------------------------------------------------------------
TEST(TarRoundTrip, MultiShard_CreateThenRead_VerifyBytesAcrossShards) {
    const std::string prefix = makeTmpPrefix("thor_tar_multi");
    CleanupGuard guard{prefix};

    const std::filesystem::path archiveDir = std::filesystem::path(prefix).remove_filename();
    const std::string archiveName = std::filesystem::path(prefix).filename().string();

    ThorImplementation::TensorPlacement cpuPlacement(ThorImplementation::TensorPlacement::MemDevices::CPU);

    // Make files large enough to require rollover under a small-ish shard limit.
    // Note: your TarWriter uses PAX + conservative slack, so pick a limit that definitely rolls.
    uint32_t fileSize = 512 * 1024;
    const std::vector<uint8_t> a = make_pattern_bytes(fileSize, 1);  // 512 KiB
    const std::vector<uint8_t> b = make_pattern_bytes(fileSize, 2);  // 512 KiB
    const std::vector<uint8_t> c = make_pattern_bytes(fileSize, 3);  // 512 KiB

    ThorImplementation::TensorDescriptor descriptor(ThorImplementation::TensorDescriptor::DataType::UINT8, {fileSize});

    ThorImplementation::Tensor aTensor(cpuPlacement, descriptor);
    memcpy(aTensor.getMemPtr<void>(), a.data(), fileSize);
    ThorImplementation::Tensor bTensor(cpuPlacement, descriptor);
    memcpy(bTensor.getMemPtr<void>(), b.data(), fileSize);
    ThorImplementation::Tensor cTensor(cpuPlacement, descriptor);
    memcpy(cTensor.getMemPtr<void>(), c.data(), fileSize);

    {
        const bool overwrite = true;

        // Force rollover: limit smaller than (header + payload + pax slack).
        // 256 KiB is almost guaranteed to roll with your 4 KiB pax slack + tar padding.
        const uint64_t shard_limit = 256 * 1024;

        TarWriter w(archiveName, shard_limit);
        w.addArchiveFile("a.bin", aTensor);
        w.addArchiveFile("b.bin", bTensor);
        w.addArchiveFile("c.bin", cTensor);
        w.createArchive(archiveDir, overwrite);
    }

    // Once a 2nd shard exists, shard0 should have been renamed to numbered form.
    ASSERT_TRUE(fs::exists(prefix + ".000000.thor.tar"));
    ASSERT_TRUE(fs::exists(prefix + ".000001.thor.tar"));
    EXPECT_FALSE(fs::exists(prefix + ".thor.tar"));

    // Read back using TarReader's scan+index+pread
    TarReader r(archiveName, archiveDir);

    read_and_expect(r, "a.bin", a);
    read_and_expect(r, "b.bin", b);
    read_and_expect(r, "c.bin", c);
}

static constexpr char testKFooterMagic[8] = {'T', 'H', 'O', 'R', 'I', 'D', 'X', '1'};
static constexpr uint64_t testKFooterSize = 24;  // 8 magic + 8 json_len + 4 crc + 4 reserved

static uint64_t read_u64_le(const uint8_t b[8]) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i)
        v = (v << 8) | static_cast<uint64_t>(b[i]);
    return v;
}

static void write_u32_le(std::fstream& io, uint64_t off, uint32_t v) {
    uint8_t b[4];
    b[0] = (uint8_t)(v & 0xFF);
    b[1] = (uint8_t)((v >> 8) & 0xFF);
    b[2] = (uint8_t)((v >> 16) & 0xFF);
    b[3] = (uint8_t)((v >> 24) & 0xFF);
    io.seekp((std::streamoff)off, std::ios::beg);
    if (!io)
        throw std::runtime_error("seekp failed while patching");
    io.write(reinterpret_cast<const char*>(b), 4);
    if (!io)
        throw std::runtime_error("write failed while patching u32");
}

struct FooterInfo24 {
    uint64_t file_size = 0;
    uint64_t json_len = 0;
    uint64_t json_start = 0;
    uint64_t footer_start = 0;
    uint64_t index_crc_off = 0;  // absolute file offset of index_crc field in footer
    std::string json;
};

static FooterInfo24 read_footer_info24(const fs::path& shard_path) {
    std::ifstream in(shard_path, std::ios::binary);
    if (!in)
        throw std::runtime_error("failed to open shard: " + shard_path.string());

    in.seekg(0, std::ios::end);
    const std::streamoff end = in.tellg();
    if (end < (std::streamoff)testKFooterSize)
        throw std::runtime_error("shard too small for footer: " + shard_path.string());

    FooterInfo24 fi;
    fi.file_size = (uint64_t)end;
    fi.footer_start = fi.file_size - testKFooterSize;

    in.seekg((std::streamoff)fi.footer_start, std::ios::beg);

    char magic[8];
    uint8_t len_bytes[8];
    uint8_t crc_bytes[4];
    uint8_t reserved_bytes[4];

    in.read(magic, 8);
    in.read(reinterpret_cast<char*>(len_bytes), 8);
    in.read(reinterpret_cast<char*>(crc_bytes), 4);
    in.read(reinterpret_cast<char*>(reserved_bytes), 4);
    if (!in)
        throw std::runtime_error("failed to read footer: " + shard_path.string());

    if (std::memcmp(magic, testKFooterMagic, 8) != 0)
        throw std::runtime_error("footer magic mismatch: " + shard_path.string());

    fi.json_len = read_u64_le(len_bytes);
    if (fi.json_len == 0 || fi.json_len > fi.file_size - testKFooterSize)
        throw std::runtime_error("invalid json_len in footer: " + shard_path.string());

    fi.json_start = fi.file_size - testKFooterSize - fi.json_len;

    // footer layout: magic(8) + len(8) + index_crc(4) + reserved(4)
    fi.index_crc_off = fi.footer_start + 8 + 8;

    in.seekg((std::streamoff)fi.json_start, std::ios::beg);
    fi.json.resize((size_t)fi.json_len);
    in.read(fi.json.data(), (std::streamsize)fi.json.size());
    if (!in)
        throw std::runtime_error("failed to read json blob: " + shard_path.string());

    return fi;
}

static void corrupt_archive_id_in_shard_and_fix_index_crc(const fs::path& shard_path) {
    FooterInfo24 fi = read_footer_info24(shard_path);

    const std::string key = "\"archive_id\":\"";
    const size_t key_pos = fi.json.find(key);
    if (key_pos == std::string::npos)
        throw std::runtime_error("could not find archive_id key in json: " + shard_path.string());

    const size_t id_pos = key_pos + key.size();
    if (id_pos + 32 > fi.json.size())
        throw std::runtime_error("archive_id field too short in json: " + shard_path.string());

    std::string old_id = fi.json.substr(id_pos, 32);

    // Make a different-but-valid 32-hex-ish id by flipping first char between '0' and '1'
    std::string new_id = old_id;
    new_id[0] = (new_id[0] == '0') ? '1' : '0';
    if (new_id == old_id)
        throw std::runtime_error("failed to produce different archive_id");

    // Patch JSON on disk (only 32 bytes) at its absolute file offset
    const uint64_t id_file_off = fi.json_start + (uint64_t)id_pos;

    {
        std::fstream io(shard_path, std::ios::binary | std::ios::in | std::ios::out);
        if (!io)
            throw std::runtime_error("failed to open shard for patching: " + shard_path.string());

        io.seekp((std::streamoff)id_file_off, std::ios::beg);
        if (!io)
            throw std::runtime_error("seekp failed patching archive_id: " + shard_path.string());

        io.write(new_id.data(), (std::streamsize)new_id.size());
        if (!io)
            throw std::runtime_error("write failed patching archive_id: " + shard_path.string());

        // Re-read JSON bytes (or update the in-memory copy and CRC that)
        // We'll update in-memory then recompute.
        fi.json.replace(id_pos, 32, new_id);

        const uint32_t new_index_crc = crc32_ieee(0xFFFFFFFF, (uint8_t*)fi.json.data(), fi.json.size());

        // Patch footer index_crc field
        write_u32_le(io, fi.index_crc_off, new_index_crc);

        io.flush();
        if (!io)
            throw std::runtime_error("flush failed patching: " + shard_path.string());
    }
}

static FooterInfo24 read_footer_info(const fs::path& shard_path) {
    std::ifstream in(shard_path, std::ios::binary);
    if (!in)
        throw std::runtime_error("failed to open shard: " + shard_path.string());

    in.seekg(0, std::ios::end);
    const std::streamoff end = in.tellg();
    if (end < static_cast<std::streamoff>(testKFooterSize)) {
        throw std::runtime_error("shard too small for footer: " + shard_path.string());
    }
    FooterInfo24 fi;
    fi.file_size = static_cast<uint64_t>(end);

    // read footer
    in.seekg(static_cast<std::streamoff>(fi.file_size - testKFooterSize), std::ios::beg);

    char magic[8];
    uint8_t len_bytes[8];
    in.read(magic, 8);
    in.read(reinterpret_cast<char*>(len_bytes), 8);
    if (!in)
        throw std::runtime_error("failed to read footer: " + shard_path.string());

    if (std::memcmp(magic, testKFooterMagic, 8) != 0) {
        throw std::runtime_error("footer magic mismatch: " + shard_path.string());
    }

    fi.json_len = read_u64_le(len_bytes);
    if (fi.json_len == 0 || fi.json_len > fi.file_size - testKFooterSize) {
        throw std::runtime_error("invalid json_len in footer: " + shard_path.string());
    }

    fi.json_start = fi.file_size - testKFooterSize - fi.json_len;

    in.seekg(static_cast<std::streamoff>(fi.json_start), std::ios::beg);
    fi.json.resize(static_cast<size_t>(fi.json_len));
    in.read(fi.json.data(), static_cast<std::streamsize>(fi.json.size()));
    if (!in)
        throw std::runtime_error("failed to read json blob: " + shard_path.string());

    return fi;
}

static void overwrite_bytes_at(const fs::path& p, uint64_t offset, const void* data, size_t len) {
    std::fstream io(p, std::ios::binary | std::ios::in | std::ios::out);
    if (!io)
        throw std::runtime_error("failed to open for patching: " + p.string());

    io.seekp(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!io)
        throw std::runtime_error("seekp failed patching: " + p.string());

    io.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(len));
    if (!io)
        throw std::runtime_error("write failed patching: " + p.string());

    io.flush();
    if (!io)
        throw std::runtime_error("flush failed patching: " + p.string());
}

static void corrupt_archive_id_in_shard(const fs::path& shard_path) {
    FooterInfo24 fi = read_footer_info(shard_path);

    // Find `"archive_id":"<32 chars>"`
    const std::string key = "\"archive_id\":\"";
    const size_t key_pos = fi.json.find(key);
    if (key_pos == std::string::npos) {
        throw std::runtime_error("could not find archive_id key in json: " + shard_path.string());
    }

    const size_t id_pos = key_pos + key.size();
    if (id_pos + 32 > fi.json.size()) {
        throw std::runtime_error("archive_id field too short in json: " + shard_path.string());
    }

    std::string old_id = fi.json.substr(id_pos, 32);

    // Make a different-but-valid 32-hex id by flipping the first nibble.
    std::string new_id = old_id;
    if (new_id[0] != '0')
        new_id[0] = '0';
    else
        new_id[0] = '1';

    if (new_id == old_id) {
        throw std::runtime_error("failed to produce a different archive_id for corruption");
    }

    // Patch only the 32 id chars in-place so json_len/footer remain valid.
    const uint64_t file_offset = fi.json_start + static_cast<uint64_t>(id_pos);
    overwrite_bytes_at(shard_path, file_offset, new_id.data(), new_id.size());
}

static std::vector<uint8_t> make_bytes(size_t n, uint8_t value) { return std::vector<uint8_t>(n, value); }

TEST(TarRoundTrip, RejectsArchiveIdMismatchAcrossThreeShards) {
    const std::string prefix = makeTmpPrefix("thor_tar_corrupt_id");
    CleanupGuard guard{prefix};

    const std::filesystem::path archiveDir = std::filesystem::path(prefix).remove_filename();
    const std::string archiveName = std::filesystem::path(prefix).filename().string();

    const uint64_t shard_limit = 256 * 1024;  // 256 KiB
    const bool overwrite = true;

    uint32_t fileSize = 512 * 1024;
    const auto a = make_bytes(fileSize, 0xA1);
    const auto b = make_bytes(fileSize, 0xB2);
    const auto c = make_bytes(fileSize, 0xC3);

    ThorImplementation::TensorPlacement cpuPlacement(ThorImplementation::TensorPlacement::MemDevices::CPU);
    ThorImplementation::TensorDescriptor descriptor(ThorImplementation::TensorDescriptor::DataType::UINT8, {fileSize});

    ThorImplementation::Tensor aTensor(cpuPlacement, descriptor);
    memcpy(aTensor.getMemPtr<void>(), a.data(), fileSize);
    ThorImplementation::Tensor bTensor(cpuPlacement, descriptor);
    memcpy(bTensor.getMemPtr<void>(), b.data(), fileSize);
    ThorImplementation::Tensor cTensor(cpuPlacement, descriptor);
    memcpy(cTensor.getMemPtr<void>(), c.data(), fileSize);

    {
        TarWriter w(archiveName, shard_limit);
        w.addArchiveFile("a.bin", aTensor);
        w.addArchiveFile("b.bin", bTensor);
        w.addArchiveFile("c.bin", cTensor);
        w.createArchive(archiveDir, overwrite);
    }

    const fs::path s0 = prefix + ".000000.thor.tar";
    const fs::path s1 = prefix + ".000001.thor.tar";
    const fs::path s2 = prefix + ".000002.thor.tar";

    ASSERT_TRUE(fs::exists(s0));
    ASSERT_TRUE(fs::exists(s1));
    ASSERT_TRUE(fs::exists(s2));

    // Corrupt shard 1's archive id (make it disagree with shards 0 and 2),
    // but keep footer CRC consistent so we test the mismatch logic (not CRC failure).
    ASSERT_NO_THROW(corrupt_archive_id_in_shard_and_fix_index_crc(s1));

    // Now TarReader should refuse to open due to archive_id mismatch.
    try {
        TarReader r(archiveName, archiveDir);
        FAIL() << "Expected TarReader to throw due to archive_id mismatch, but it constructed successfully.";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_TRUE(msg.find("Archive ids mismatch") != std::string::npos || msg.find("mismatch") != std::string::npos)
            << "Unexpected error message: " << msg;
    }
}

TEST(TarRoundTrip, RejectsWrongFooterMagicNumber) {
    const std::string prefix = makeTmpPrefix("thor_tar_bad_magic");
    CleanupGuard guard{prefix};

    const std::filesystem::path archiveDir = std::filesystem::path(prefix).remove_filename();
    const std::string archiveName = std::filesystem::path(prefix).filename().string();

    const uint64_t shard_limit = 0;  // single shard
    const bool overwrite = true;

    uint32_t fileSize = 64 * 1024;
    const auto payload = make_bytes(fileSize, 0x5A);

    ThorImplementation::TensorPlacement cpuPlacement(ThorImplementation::TensorPlacement::MemDevices::CPU);
    ThorImplementation::TensorDescriptor descriptor(ThorImplementation::TensorDescriptor::DataType::UINT8, {fileSize});
    ThorImplementation::Tensor payloadTensor(cpuPlacement, descriptor);
    memcpy(payloadTensor.getMemPtr<void>(), payload.data(), fileSize);

    {
        TarWriter w(archiveName, shard_limit);
        w.addArchiveFile("x.bin", payloadTensor);
        w.createArchive(archiveDir, overwrite);
    }

    const fs::path shard0 = prefix + ".thor.tar";
    ASSERT_TRUE(fs::exists(shard0));

    // Patch the footer magic (last 16 bytes = 8 magic + 8 len)
    // Keep length unchanged; only corrupt magic bytes.
    std::array<char, 8> bad_magic = {'B', 'A', 'D', 'M', 'A', 'G', 'I', 'C'};
    const uint64_t file_size = static_cast<uint64_t>(fs::file_size(shard0));
    ASSERT_GT(file_size, testKFooterSize);

    const uint64_t magic_off = file_size - testKFooterSize;  // start of footer
    ASSERT_NO_THROW(overwrite_bytes_at(shard0, magic_off, bad_magic.data(), bad_magic.size()));

    // Reader should reject due to wrong magic.
    try {
        TarReader r(archiveName, archiveDir);
        FAIL() << "Expected TarReader to throw due to wrong footer magic, but it constructed successfully.";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_TRUE(msg.find("magic") != std::string::npos || msg.find("footer") != std::string::npos ||
                    msg.find("index") != std::string::npos)
            << "Unexpected error message: " << msg;
    }
}

TEST(TarRoundTrip, ManyFiles_ManyShards_1MiBLimit_100FilesTotal10MiB) {
    const std::string prefix = makeTmpPrefix("thor_tar_100files");
    CleanupGuard guard{prefix};

    const std::filesystem::path archiveDir = std::filesystem::path(prefix).remove_filename();
    const std::string archiveName = std::filesystem::path(prefix).filename().string();

    const uint64_t shard_limit = 1ull * 1024 * 1024;  // 1 MiB
    const bool overwrite = true;

    // Generate 100 files with random-ish sizes that sum to exactly 10 MiB.
    // Sizes are chosen deterministically (no RNG dependency) but "random-looking".
    static constexpr uint32_t kNumFiles = 100;
    static constexpr uint64_t kTotalBytes = 10ull * 1024 * 1024;

    std::vector<uint32_t> sizes(kNumFiles, 0);

    // Make 99 variable sizes between 1 KiB and 256 KiB, then set last as remainder.
    // Clamp to keep it sane and ensure remainder >= 1.
    uint64_t used = 0;
    for (uint32_t i = 0; i < kNumFiles - 1; ++i) {
        // simple LCG-ish mix
        uint32_t x = (i + 1) * 1103515245u + 12345u;
        uint32_t sz = 1024u + (x % (256u * 1024u));  // [1KiB, 256KiB]
        sizes[i] = sz;
        used += sz;
    }
    // Ensure remainder is valid; if not, renormalize by scaling down.
    if (used >= kTotalBytes) {
        // Scale down first 99 sizes proportionally so remainder is positive.
        const double scale = double(kTotalBytes - 1024) / double(used);  // leave at least 1KiB for last
        used = 0;
        for (uint32_t i = 0; i < kNumFiles - 1; ++i) {
            uint32_t sz = static_cast<uint32_t>(double(sizes[i]) * scale);
            sz = std::max<uint32_t>(1024u, sz);
            sizes[i] = sz;
            used += sz;
        }
    }
    sizes[kNumFiles - 1] = static_cast<uint32_t>(kTotalBytes - used);
    ASSERT_GE(sizes[kNumFiles - 1], 1u);

    // Build file contents (patterned for verifiable round-trip) and write.
    std::vector<std::string> paths;
    paths.reserve(kNumFiles);
    std::vector<std::vector<uint8_t>> contents;
    contents.reserve(kNumFiles);

    ThorImplementation::TensorPlacement cpuPlacement(ThorImplementation::TensorPlacement::MemDevices::CPU);

    {
        TarWriter w(archiveName, shard_limit);

        uint64_t total_written = 0;
        for (uint32_t i = 0; i < kNumFiles; ++i) {
            char name[64];
            std::snprintf(name, sizeof(name), "dir/file_%03u.bin", i);
            paths.emplace_back(name);

            auto data = make_pattern_bytes(static_cast<size_t>(sizes[i]), 1000u + i);
            total_written += data.size();

            ThorImplementation::TensorDescriptor descriptor(ThorImplementation::TensorDescriptor::DataType::UINT8, {data.size()});
            ThorImplementation::Tensor dataTensor(cpuPlacement, descriptor);
            memcpy(dataTensor.getMemPtr<void>(), data.data(), data.size());

            contents.push_back(std::move(data));

            w.addArchiveFile(paths.back(), dataTensor);
        }
        ASSERT_EQ(total_written, kTotalBytes);

        w.createArchive(archiveDir, overwrite);
    }

    // The archive should exist (either single or multi; with this size it should be multi)
    // Your TarWriter renames shard0 to numbered form once it creates shard1.
    ASSERT_TRUE(fs::exists(prefix + ".000000.thor.tar")) << "expected multi-shard output";

    // Read back and verify all files via TarReader (pread path).
    TarReader r(archiveName, archiveDir);

    std::vector<ThorImplementation::Tensor> destTensors;
    std::vector<uint32_t> actualCrcs;

    for (uint32_t i = 0; i < kNumFiles; ++i) {
        ASSERT_TRUE(r.containsFile(paths[i])) << "missing: " << paths[i];

        std::vector<EntryInfo> fileEntries = r.getFileShards(paths[i]);
        // const auto& info = r.getFileShards(path);
        ASSERT_EQ(fileEntries.size(), 1UL);
        EntryInfo info = fileEntries[0];
        ASSERT_EQ(info.size, static_cast<uint64_t>(contents[i].size())) << "size mismatch for " << paths[i];
        actualCrcs.push_back(info.crc);

        ThorImplementation::TensorDescriptor descriptor(ThorImplementation::TensorDescriptor::DataType::UINT8, {contents[i].size()});
        ThorImplementation::Tensor destTensor(cpuPlacement, descriptor);
        destTensors.push_back(destTensor);

        r.registerReadRequest(paths[i], destTensor);
    }

    r.executeReadRequests();

    for (uint32_t i = 0; i < kNumFiles; ++i) {
        ASSERT_EQ(destTensors[i].getArraySizeInBytes(), contents[i].size());
        ASSERT_EQ(std::memcmp(destTensors[i].getMemPtr<uint8_t>(), contents[i].data(), contents[i].size()), 0)
            << "payload mismatch for " << paths[i];

        const uint32_t crc_expected = crc32_ieee(0xFFFFFFFF, (uint8_t*)contents[i].data(), contents[i].size());
        EXPECT_EQ(actualCrcs[i], crc_expected) << "index crc32c mismatch for " << paths[i];
    }

    // Optional: ensure shard count is "reasonable" (at least 10 shards for 10MiB @ 1MiB limit)
    // We don't assume exact count due to tar headers/index trailer overhead.
    uint32_t shard_count = 0;
    for (int i = 0; i < 512; ++i) {
        char suf[64];
        std::snprintf(suf, sizeof(suf), ".%06d.thor.tar", i);
        if (fs::exists(prefix + std::string(suf)))
            shard_count++;
        else
            break;
    }
    EXPECT_GE(shard_count, 8u);  // loose lower bound
}

// Corrupt just the footer CRC (index_crc) on a shard, without touching JSON bytes.
// Then TarReader should throw due to index CRC mismatch.
TEST(TarRoundTrip, RejectsBadIndexCrcInFooter) {
    const std::string prefix = makeTmpPrefix("thor_tar_bad_index_crc");
    CleanupGuard guard{prefix};

    const std::filesystem::path archiveDir = std::filesystem::path(prefix).remove_filename();
    const std::string archiveName = std::filesystem::path(prefix).filename().string();

    const uint64_t shard_limit = 1'000'000;  // single shard
    const bool overwrite = true;

    ThorImplementation::TensorPlacement cpuPlacement(ThorImplementation::TensorPlacement::MemDevices::CPU);

    // Build some files
    const std::string hello_str = "hello from gtest\n";
    ThorImplementation::TensorDescriptor helloDescriptor(ThorImplementation::TensorDescriptor::DataType::UINT8, {hello_str.size()});
    ThorImplementation::Tensor helloTensor(cpuPlacement, helloDescriptor);
    memcpy(helloTensor.getMemPtr<void>(), hello_str.data(), hello_str.size());

    uint32_t blobBytes = 256 * 1024;
    const std::vector<uint8_t> blob = make_pattern_bytes(blobBytes, 778);
    ThorImplementation::TensorDescriptor blobDescriptor(ThorImplementation::TensorDescriptor::DataType::UINT8, {blobBytes});
    ThorImplementation::Tensor blobTensor(cpuPlacement, blobDescriptor);
    memcpy(blobTensor.getMemPtr<void>(), blob.data(), blob.size());

    {
        TarWriter w(archiveName, shard_limit);
        w.addArchiveFile("docs/hello.txt", helloTensor);
        w.addArchiveFile("data/blob.bin", blobTensor);
        w.createArchive(archiveDir, overwrite);
    }

    const fs::path shard0 = prefix + ".thor.tar";
    ASSERT_TRUE(fs::exists(shard0)) << shard0;

    // Footer layout (24 bytes): magic(8) + json_len(8) + index_crc(4) + reserved(4)
    static constexpr uint64_t testKFooterSize = 24;

    const uint64_t file_size = static_cast<uint64_t>(fs::file_size(shard0));
    ASSERT_GT(file_size, testKFooterSize);

    const uint64_t footer_start = file_size - testKFooterSize;
    const uint64_t index_crc_off = footer_start + 8 + 8;  // after magic+len

    // Read existing index_crc so we can flip it.
    uint32_t old_crc = 0;
    {
        std::ifstream in(shard0, std::ios::binary);
        ASSERT_TRUE(in.good());
        in.seekg(static_cast<std::streamoff>(index_crc_off), std::ios::beg);
        ASSERT_TRUE(in.good());
        uint8_t b[4];
        in.read(reinterpret_cast<char*>(b), 4);
        ASSERT_TRUE(in.good());
        old_crc = (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    }

    // Flip bits to guarantee mismatch.
    const uint32_t bad_crc = old_crc ^ 0xA5A5A5A5u;

    // Patch the footer index_crc field.
    {
        std::fstream io(shard0, std::ios::binary | std::ios::in | std::ios::out);
        ASSERT_TRUE(io.good());
        io.seekp(static_cast<std::streamoff>(index_crc_off), std::ios::beg);
        ASSERT_TRUE(io.good());

        uint8_t b[4];
        b[0] = static_cast<uint8_t>(bad_crc & 0xFF);
        b[1] = static_cast<uint8_t>((bad_crc >> 8) & 0xFF);
        b[2] = static_cast<uint8_t>((bad_crc >> 16) & 0xFF);
        b[3] = static_cast<uint8_t>((bad_crc >> 24) & 0xFF);
        io.write(reinterpret_cast<const char*>(b), 4);
        ASSERT_TRUE(io.good());
        io.flush();
        ASSERT_TRUE(io.good());
    }

    // Now TarReader should reject due to index CRC mismatch.
    EXPECT_THROW({ TarReader r(archiveName, archiveDir); }, std::runtime_error);
}

inline uint32_t read_u32_le(const uint8_t b[4]) {
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}

static nlohmann::json load_footer_index_json(const fs::path& shard_path) {
    std::ifstream in(shard_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("load_footer_index_json: failed to open: " + shard_path.string());
    }

    in.seekg(0, std::ios::end);
    const std::streamoff end = in.tellg();
    if (end < (std::streamoff)testKFooterSize) {
        throw std::runtime_error("load_footer_index_json: file too small for footer: " + shard_path.string());
    }

    const uint64_t file_size = (uint64_t)end;
    const uint64_t footer_start = file_size - testKFooterSize;

    in.seekg((std::streamoff)footer_start, std::ios::beg);

    char magic[8];
    uint8_t len_bytes[8];
    uint8_t crc_bytes[4];
    uint8_t reserved_bytes[4];

    in.read(magic, 8);
    in.read(reinterpret_cast<char*>(len_bytes), 8);
    in.read(reinterpret_cast<char*>(crc_bytes), 4);
    in.read(reinterpret_cast<char*>(reserved_bytes), 4);
    if (!in) {
        throw std::runtime_error("load_footer_index_json: failed reading footer: " + shard_path.string());
    }

    if (std::memcmp(magic, testKFooterMagic, 8) != 0) {
        throw std::runtime_error("load_footer_index_json: footer magic mismatch: " + shard_path.string());
    }

    const uint64_t json_len = read_u64_le(len_bytes);
    if (json_len == 0 || json_len > file_size - testKFooterSize) {
        throw std::runtime_error("load_footer_index_json: invalid json_len in footer: " + shard_path.string());
    }

    const uint32_t index_crc_expected = read_u32_le(crc_bytes);
    (void)reserved_bytes;  // reserved for future

    const uint64_t json_start = file_size - testKFooterSize - json_len;

    in.seekg((std::streamoff)json_start, std::ios::beg);
    std::string json_str(json_len, '\0');
    in.read(json_str.data(), (std::streamsize)json_str.size());
    if (!in) {
        throw std::runtime_error("load_footer_index_json: failed reading json blob: " + shard_path.string());
    }

    // Validate index CRC (IEEE CRC-32)
    const uint32_t index_crc_got = crc32_ieee(0xFFFFFFFF, (uint8_t*)json_str.data(), json_str.size());
    if (index_crc_got != index_crc_expected) {
        throw std::runtime_error("load_footer_index_json: index CRC mismatch in " + shard_path.string() +
                                 " expected=" + std::to_string(index_crc_expected) + " got=" + std::to_string(index_crc_got));
    }

    try {
        return nlohmann::json::parse(json_str);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("load_footer_index_json: JSON parse failed: ") + e.what());
    }
}

// Corrupt one bit of a file's payload and ensure TarReader throws when validate=true.
TEST(TarRoundTrip, CorruptSingleBitInPayload_ThrowsOnValidatedRead) {
    const std::string prefix = makeTmpPrefix("thor_tar_corrupt_payload_bit");
    CleanupGuard guard{prefix};

    const std::filesystem::path archiveDir = std::filesystem::path(prefix).remove_filename();
    const std::string archiveName = std::filesystem::path(prefix).filename().string();
    const uint64_t shard_limit = 1'000'000;  // single shard
    const bool overwrite = true;

    ThorImplementation::TensorPlacement cpuPlacement(ThorImplementation::TensorPlacement::MemDevices::CPU);

    // Build some files
    const std::string hello_str = "hello from gtest\n";
    ThorImplementation::TensorDescriptor helloDescriptor(ThorImplementation::TensorDescriptor::DataType::UINT8, {hello_str.size()});
    ThorImplementation::Tensor helloTensor(cpuPlacement, helloDescriptor);
    memcpy(helloTensor.getMemPtr<void>(), hello_str.data(), hello_str.size());

    uint32_t blobBytes = 75 * 1024;
    const std::vector<uint8_t> blob = make_pattern_bytes(blobBytes, 999);
    ThorImplementation::TensorDescriptor blobDescriptor(ThorImplementation::TensorDescriptor::DataType::UINT8, {blobBytes});
    ThorImplementation::Tensor blobTensor(cpuPlacement, blobDescriptor);
    memcpy(blobTensor.getMemPtr<void>(), blob.data(), blob.size());

    {
        TarWriter w(archiveName, shard_limit);
        w.addArchiveFile("docs/hello.txt", helloTensor);
        w.addArchiveFile("data/blob.bin", blobTensor);
        w.createArchive(archiveDir, overwrite);
    }

    const fs::path shard0 = prefix + ".thor.tar";
    ASSERT_TRUE(fs::exists(shard0)) << shard0;

    // Load footer index JSON (validates index CRC) and get offsets for blob
    nlohmann::json j0;
    ASSERT_NO_THROW(j0 = load_footer_index_json(shard0));
    ASSERT_TRUE(j0.contains("files"));
    const auto& files = j0["files"];
    ASSERT_TRUE(files.contains("data/blob.bin"));
    ASSERT_TRUE(j0.contains("checksum_alg"));
    const auto& checksum_alg = j0.at("checksum_alg").get<std::string>();
    ASSERT_TRUE(checksum_alg == "crc32_ieee");

    const auto& e = files["data/blob.bin"][0];
    const uint64_t off = e["file_data_offset"].get<uint64_t>();
    const uint64_t sz = e["size"].get<uint64_t>();
    ASSERT_GT(sz, 16u) << "blob too small for corruption test";
    ASSERT_TRUE(e.contains("crc"));

    // Corrupt a single bit somewhere inside the payload (not in the header / footer).
    // We'll flip the lowest bit of one byte near the middle.
    const uint64_t corrupt_pos = off + (sz / 2);

    {
        std::fstream io(shard0, std::ios::binary | std::ios::in | std::ios::out);
        ASSERT_TRUE(io.good()) << "failed to open shard for patching";

        io.seekg(static_cast<std::streamoff>(corrupt_pos), std::ios::beg);
        ASSERT_TRUE(io.good());

        char byte = 0;
        io.read(&byte, 1);
        ASSERT_TRUE(io.good());

        byte ^= 0x01;  // flip one bit

        io.seekp(static_cast<std::streamoff>(corrupt_pos), std::ios::beg);
        ASSERT_TRUE(io.good());

        io.write(&byte, 1);
        ASSERT_TRUE(io.good());
        io.flush();
        ASSERT_TRUE(io.good());
    }

    // Now reading with validate=true should throw due to CRC mismatch.
    TarReader r(archiveName, archiveDir);

    ThorImplementation::TensorDescriptor descriptor(ThorImplementation::TensorDescriptor::DataType::UINT8, {sz});
    ThorImplementation::Tensor out(cpuPlacement, descriptor);

    r.registerReadRequest("data/blob.bin", out);

    EXPECT_THROW({ r.executeReadRequests(); }, std::runtime_error);
}

// FIXME: Test multiple shards for a single file
