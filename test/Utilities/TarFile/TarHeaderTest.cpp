// GTests for the TAR header helper:
// - Simple ustar header (name/prefix split) without PAX
// - Overlong filenames requiring PAX "path="
// - Also validates checksum fields and basic 512-block alignment properties

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "Utilities/TarFile/TarHeaderHelper.h"

static uint64_t parse_octal_field(const char* p, size_t n) {
    // Parse NUL/space-terminated octal field.
    uint64_t v = 0;
    size_t i = 0;
    // skip leading spaces/NUL
    while (i < n && (p[i] == ' ' || p[i] == '\0'))
        ++i;
    for (; i < n; ++i) {
        char c = p[i];
        if (c == '\0' || c == ' ')
            break;
        if (c < '0' || c > '7')
            break;
        v = (v << 3) + uint64_t(c - '0');
    }
    return v;
}

static uint32_t checksum_with_spaces(const TarHeader& h) {
    TarHeader tmp = h;
    std::memset(tmp.chksum, ' ', sizeof(tmp.chksum));
    const unsigned char* p = reinterpret_cast<const unsigned char*>(&tmp);
    uint32_t sum = 0;
    for (size_t i = 0; i < 512; ++i)
        sum += p[i];
    return sum;
}

static std::string tar_name_full(const TarHeader& h) {
    std::string name(h.name, h.name + 100);
    name.resize(std::strlen(name.c_str()));  // trims at first NUL

    std::string prefix(h.prefix, h.prefix + 155);
    prefix.resize(std::strlen(prefix.c_str()));

    if (!prefix.empty())
        return prefix + "/" + name;
    return name;
}

static void expect_ustar_magic(const TarHeader& h) {
    EXPECT_EQ(std::string(h.magic, h.magic + 5), "ustar");
    EXPECT_TRUE(h.version[0] == '0' && h.version[1] == '0');
}

static void expect_checksum_valid(const TarHeader& h) {
    // stored checksum field is octal
    uint64_t stored = parse_octal_field(h.chksum, sizeof(h.chksum));
    uint32_t computed = checksum_with_spaces(h);
    EXPECT_EQ(stored, computed);
}

static void expect_header_size_field(const TarHeader& h, uint64_t expectedSize) {
    uint64_t sz = parse_octal_field(h.size, sizeof(h.size));
    EXPECT_EQ(sz, expectedSize);
}

static bool contains_substr(const uint8_t* p, size_t n, const std::string& s) {
    if (s.empty() || n < s.size())
        return false;
    for (size_t i = 0; i + s.size() <= n; ++i) {
        if (std::memcmp(p + i, s.data(), s.size()) == 0)
            return true;
    }
    return false;
}

// ------------------------ Tests ------------------------

TEST(TarHeaderHelper, SimpleUstar_NoPax) {
    // prevFileSize has padding added first
    const uint64_t prevFileSize = 1000;  // pad = 24 to 512
    const uint64_t nextFileSize = 1234;

    const std::string nextPath = "dir/subdir/file.bin";  // fits in name

    alignas(4096) uint8_t scratch[8192];
    std::memset(scratch, 0xCD, sizeof(scratch));

    TarSeparatorBuildResult dbg{};
    uint32_t outBytes = createTarSeparator(nextPath,
                                           nextFileSize,
                                           prevFileSize,
                                           /*tailBytesIn=*/0,
                                           scratch,
                                           sizeof(scratch),
                                           &dbg);

    // Should include:
    // - prev payload pad to 512
    // - NO pax
    // - one file header (512)
    EXPECT_EQ(dbg.paxBytes, 0u);
    EXPECT_EQ(dbg.fileHeaderBytes, 512u);
    EXPECT_EQ(dbg.payloadPadBytes, (uint32_t)((512 - (prevFileSize % 512)) % 512));
    EXPECT_EQ(outBytes, dbg.totalBytesAppended);

    // Layout: [payloadPad] [fileHeader]
    ASSERT_GE(outBytes, dbg.payloadPadBytes + 512u);
    const uint8_t* hdrPtr = scratch + dbg.payloadPadBytes;

    TarHeader h{};
    std::memcpy(&h, hdrPtr, sizeof(h));

    expect_ustar_magic(h);
    EXPECT_EQ(h.typeflag, '0');
    expect_checksum_valid(h);
    expect_header_size_field(h, nextFileSize);

    // Path should round-trip from prefix+name
    EXPECT_EQ(tar_name_full(h), nextPath);
}

TEST(TarHeaderHelper, UstarPrefixSplit_NoPax) {
    // Create a path > 100 but splittable into prefix<=155 and name<=100.
    // prefix ~ 120, name ~ 50.
    std::string prefix(120, 'a');
    std::string name(50, 'b');
    std::string nextPath = prefix + "/" + name;

    const uint64_t prevFileSize = 0;
    const uint64_t nextFileSize = 4096;

    alignas(4096) uint8_t scratch[8192];
    std::memset(scratch, 0, sizeof(scratch));

    TarSeparatorBuildResult dbg{};
    uint32_t outBytes = createTarSeparator(nextPath, nextFileSize, prevFileSize, 0, scratch, sizeof(scratch), &dbg);

    EXPECT_EQ(dbg.paxBytes, 0u);
    EXPECT_EQ(dbg.fileHeaderBytes, 512u);

    TarHeader h{};
    std::memcpy(&h, scratch + dbg.payloadPadBytes, sizeof(h));
    expect_ustar_magic(h);
    expect_checksum_valid(h);
    EXPECT_EQ(tar_name_full(h), nextPath);
}

TEST(TarHeaderHelper, OverlongName_UsesPaxPath) {
    // Construct a very long path that cannot be represented by ustar name/prefix split.
    // Easiest: no '/' at all and >100 bytes => cannot split.
    std::string longName(300, 'x');  // >100 and no slash => requires pax
    const std::string nextPath = longName;

    const uint64_t prevFileSize = 7;      // pad = 505
    const uint64_t nextFileSize = 12345;  // fits octal, but path forces pax

    std::vector<uint8_t> scratch;
    scratch.resize(64 * 1024);
    std::memset(scratch.data(), 0xCD, scratch.size());

    TarSeparatorBuildResult dbg{};
    uint32_t outBytes = createTarSeparator(nextPath, nextFileSize, prevFileSize, 0, scratch.data(), (uint32_t)scratch.size(), &dbg);

    EXPECT_GT(dbg.paxBytes, 0u);
    EXPECT_EQ(dbg.fileHeaderBytes, 512u);
    EXPECT_EQ(dbg.payloadPadBytes, (uint32_t)((512 - (prevFileSize % 512)) % 512));
    EXPECT_EQ(outBytes, dbg.totalBytesAppended);

    // Layout: [prev pad] [pax header 512] [pax payload + pad] [file header 512]
    const uint32_t pad = dbg.payloadPadBytes;
    ASSERT_GE(outBytes, pad + 512u + 512u);

    TarHeader paxHdr{};
    std::memcpy(&paxHdr, scratch.data() + pad, sizeof(paxHdr));
    expect_ustar_magic(paxHdr);
    EXPECT_EQ(paxHdr.typeflag, 'x');
    expect_checksum_valid(paxHdr);

    uint64_t paxPayloadBytes = parse_octal_field(paxHdr.size, sizeof(paxHdr.size));
    ASSERT_GT(paxPayloadBytes, 0u);

    const uint8_t* paxPayloadPtr = scratch.data() + pad + 512u;
    ASSERT_LE((size_t)paxPayloadBytes, scratch.size() - (pad + 512u));

    // PAX payload must contain "path=<nextPath>\n" somewhere
    std::string expectLine = "path=" + nextPath + "\n";
    EXPECT_TRUE(contains_substr(paxPayloadPtr, (size_t)paxPayloadBytes, expectLine));

    // File header is after pax payload + its 512 padding
    uint32_t paxPad = (uint32_t)((512 - (paxPayloadBytes % 512)) % 512);
    const uint8_t* fileHdrPtr = paxPayloadPtr + paxPayloadBytes + paxPad;

    TarHeader fileHdr{};
    std::memcpy(&fileHdr, fileHdrPtr, sizeof(fileHdr));
    expect_ustar_magic(fileHdr);
    EXPECT_EQ(fileHdr.typeflag, '0');
    expect_checksum_valid(fileHdr);
    expect_header_size_field(fileHdr, nextFileSize);

    // Name might be truncated fallback in the ustar header, but that's okay—PAX supplies real path.
}

TEST(TarHeaderHelper, TailBytesIn_AppendsToExistingTail) {
    const uint64_t prevFileSize = 512;
    const uint64_t nextFileSize = 1;
    const std::string nextPath = "file.txt";

    alignas(4096) uint8_t scratch[8192];
    std::memset(scratch, 0, sizeof(scratch));

    // Pre-fill some tail bytes (simulate carry)
    const uint32_t tailIn = 123;
    for (uint32_t i = 0; i < tailIn; ++i)
        scratch[i] = 0xAB;

    TarSeparatorBuildResult dbg{};
    uint32_t outBytes = createTarSeparator(nextPath, nextFileSize, prevFileSize, tailIn, scratch, sizeof(scratch), &dbg);

    // Ensure the initial tail is preserved
    for (uint32_t i = 0; i < tailIn; ++i) {
        ASSERT_EQ(scratch[i], 0xAB);
    }

    ASSERT_EQ(outBytes, tailIn + dbg.totalBytesAppended);
}

TEST(TarHeaderHelper, PrevPayloadPadCorrectness) {
    // prev size 0 => pad 0
    // prev size 1 => pad 511
    // prev size 511 => pad 1
    // prev size 512 => pad 0
    struct Case {
        uint64_t prev;
        uint32_t pad;
    };
    Case cases[] = {
        {0, 0},
        {1, 511},
        {511, 1},
        {512, 0},
        {1000, (uint32_t)((512 - (1000 % 512)) % 512)},
    };

    for (auto c : cases) {
        alignas(4096) uint8_t scratch[8192];
        std::memset(scratch, 0xCD, sizeof(scratch));
        TarSeparatorBuildResult dbg{};
        uint32_t outBytes = createTarSeparator("x", 0, c.prev, 0, scratch, sizeof(scratch), &dbg);
        EXPECT_EQ(dbg.payloadPadBytes, c.pad);
        EXPECT_EQ(outBytes, dbg.totalBytesAppended);
        // padding bytes must be zero
        for (uint32_t i = 0; i < c.pad; ++i) {
            ASSERT_EQ(scratch[i], 0) << "pad not zero at i=" << i << " prev=" << c.prev;
        }
    }
}

class ScopedUnlink {
   public:
    explicit ScopedUnlink(std::string path) : path_(std::move(path)) {}
    ~ScopedUnlink() {
        if (path_.empty())
            return;
        if (::unlink(path_.c_str()) != 0) {
            std::fprintf(stderr, "ScopedUnlink: unlink('%s') failed: %s\n", path_.c_str(), std::strerror(errno));
        }
    }
    void release() { path_.clear(); }
    const std::string& path() const { return path_; }

   private:
    std::string path_;
};

static bool env_bool(const char* name, bool def = false) {
    const char* v = std::getenv(name);
    if (!v || !*v)
        return def;
    return (std::strcmp(v, "1") == 0) || (std::strcmp(v, "true") == 0) || (std::strcmp(v, "TRUE") == 0);
}

static std::string shellEscapeDoubleQuoted(const std::string& s) {
    // For our use: paths are expected to be sane. Still escape backslash and double quote.
    std::string out;
    out.reserve(s.size() + 2);
    out.push_back('"');
    for (char c : s) {
        if (c == '\\' || c == '"')
            out.push_back('\\');
        out.push_back(c);
    }
    out.push_back('"');
    return out;
}

static std::string runCommandCaptureStdout(const std::string& cmd, int* exitCodeOut = nullptr) {
    std::array<char, 4096> buf{};
    std::string out;

    FILE* pipe = ::popen(cmd.c_str(), "r");
    if (!pipe) {
        throw std::runtime_error("popen failed for: " + cmd);
    }

    while (!std::feof(pipe)) {
        size_t n = std::fread(buf.data(), 1, buf.size(), pipe);
        if (n)
            out.append(buf.data(), n);
    }

    int rc = ::pclose(pipe);
    if (exitCodeOut)
        *exitCodeOut = rc;
    return out;
}

static void writeZeros(std::ofstream& out, uint64_t n) {
    static constexpr size_t kChunk = 4096;
    static uint8_t zeros[kChunk] = {0};
    while (n) {
        size_t w = static_cast<size_t>(std::min<uint64_t>(n, kChunk));
        out.write(reinterpret_cast<const char*>(zeros), static_cast<std::streamsize>(w));
        if (!out)
            throw std::runtime_error("ofstream writeZeros failed");
        n -= w;
    }
}

static uint64_t tarPad512(uint64_t n) { return (TAR_BLOCK - (n % TAR_BLOCK)) % TAR_BLOCK; }

// ----------------------------------------------
// The actual interop test
// ----------------------------------------------
TEST(TarInterop, TarListsAndExtractsSimpleAndPaxPaths) {
    // Verify `tar` exists
    {
        int rc = 0;
        (void)runCommandCaptureStdout("tar --version >/dev/null 2>&1", &rc);
        if (rc != 0) {
            GTEST_SKIP() << "`tar` not available on PATH";
        }
    }

    // Build two test entries
    const std::string path1 = "dir/file1.bin";
    const std::string payload1 = "hello123";  // 8 bytes
    const uint64_t size1 = payload1.size();

    const std::string path2 = std::string(300, 'x');  // forces PAX (no '/' and >100)
    const std::string payload2 = "world456";          // 8 bytes
    const uint64_t size2 = payload2.size();

    // Temp tar path (use your existing temp helper if you have one)
    // std::string tarPath = makeTmpPrefix("tar_interop") + ".tar";
    std::string tarPath = "/tmp/tar_interop_test_" + std::to_string(::getpid()) + ".tar";
    ScopedUnlink cleanup(tarPath);

    std::ofstream out(tarPath, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(out) << "failed to open " << tarPath;

    // Scratch for header building
    std::vector<uint8_t> scratch;
    scratch.resize(64 * 1024);
    std::memset(scratch.data(), 0, scratch.size());

    uint32_t tail = 0;

    // 1) Write headers for file1 (prevFileSize=0, so no pad emitted)
    TarSeparatorBuildResult dbg1{};
    tail = createTarSeparator(path1, size1, /*prevFileSize=*/0, tail, scratch.data(), (uint32_t)scratch.size(), &dbg1);
    ASSERT_GT(tail, 0u);
    out.write(reinterpret_cast<const char*>(scratch.data()), tail);
    ASSERT_TRUE(out);

    // 2) Write payload1
    out.write(payload1.data(), static_cast<std::streamsize>(payload1.size()));
    ASSERT_TRUE(out);

    // 3) Write separator for file2 (includes pad after payload1 + (PAX?) + file2 header)
    std::memset(scratch.data(), 0, scratch.size());
    tail = 0;
    TarSeparatorBuildResult dbg2{};
    tail = createTarSeparator(path2, size2, /*prevFileSize=*/size1, tail, scratch.data(), (uint32_t)scratch.size(), &dbg2);
    ASSERT_GT(tail, 0u);
    out.write(reinterpret_cast<const char*>(scratch.data()), tail);
    ASSERT_TRUE(out);

    // 4) Write payload2
    out.write(payload2.data(), static_cast<std::streamsize>(payload2.size()));
    ASSERT_TRUE(out);

    // 5) Pad after payload2 to 512 boundary
    writeZeros(out, tarPad512(size2));

    // 6) End-of-archive markers: two 512-byte zero blocks
    writeZeros(out, 2 * TAR_BLOCK);

    out.flush();
    out.close();

    // --------- Interop: list entries ----------
    {
        // tar -tf <archive>
        std::string cmd = "tar -tf " + shellEscapeDoubleQuoted(tarPath);
        int rc = 0;
        std::string listing = runCommandCaptureStdout(cmd, &rc);
        ASSERT_EQ(rc, 0) << "tar -tf failed\ncmd: " << cmd << "\noutput:\n" << listing;

        // listing is newline-delimited
        auto hasLine = [&](const std::string& needle) {
            std::istringstream iss(listing);
            std::string line;
            while (std::getline(iss, line)) {
                if (line == needle)
                    return true;
            }
            return false;
        };

        EXPECT_TRUE(hasLine(path1)) << "tar listing missing " << path1 << "\nlisting:\n" << listing;
        EXPECT_TRUE(hasLine(path2)) << "tar listing missing long pax path\nlisting:\n" << listing;
    }

    // --------- Interop: extract payload1 ----------
    {
        // tar -xOf <archive> <path>
        std::string cmd = "tar -xOf " + shellEscapeDoubleQuoted(tarPath) + " " + shellEscapeDoubleQuoted(path1);
        int rc = 0;
        std::string data = runCommandCaptureStdout(cmd, &rc);
        ASSERT_EQ(rc, 0) << "tar -xOf failed for " << path1 << "\ncmd: " << cmd << "\noutput:\n" << data;
        EXPECT_EQ(data, payload1);
    }

    // --------- Interop: extract payload2 (long pax path) ----------
    {
        std::string cmd = "tar -xOf " + shellEscapeDoubleQuoted(tarPath) + " " + shellEscapeDoubleQuoted(path2);
        int rc = 0;
        std::string data = runCommandCaptureStdout(cmd, &rc);
        ASSERT_EQ(rc, 0) << "tar -xOf failed for long pax path\ncmd: " << cmd << "\noutput:\n" << data;
        EXPECT_EQ(data, payload2);
    }
}

// It tests appendTarEndOfArchive(lastFileSize, ...) guarantees:
//  - appends >= (pad512(lastFileSize) + 1024)
//  - appended bytes are all zero
//  - final tailBytesOut is 4KB-aligned
//  - the first (pad512 + 1024) bytes are present (not required to be exactly that many)
TEST(TarHeaderHelper, AppendTarEndOfArchive_PadsTo512AndAddsEOA_AndAligns4k) {
    auto pad512 = [](uint64_t n) -> uint32_t { return static_cast<uint32_t>((512 - (n % 512)) % 512); };

    // Try a few combinations that exercise different remainders.
    struct Case {
        uint64_t lastFileSize;
        uint32_t tailIn;
    };
    Case cases[] = {
        {0, 0},
        {1, 0},
        {511, 0},
        {512, 0},
        {1000, 0},
        {12345, 7},
        {12345, 4095},  // near alignment boundary
        {12345, 4096},  // already aligned
        {777, 123},     // arbitrary
    };

    for (const auto& c : cases) {
        // Big enough scratch. (If you use smaller scratch in your code, tune accordingly.)
        std::vector<uint8_t> scratch(64 * 1024, 0xCD);

        // Pretend existing tail bytes are non-zero (to ensure we don't overwrite them).
        for (uint32_t i = 0; i < c.tailIn; ++i)
            scratch[i] = 0xAB;

        uint32_t out = appendTarEndOfArchive(c.lastFileSize, c.tailIn, scratch.data(), static_cast<uint32_t>(scratch.size()));

        // tail preserved
        for (uint32_t i = 0; i < c.tailIn; ++i) {
            ASSERT_EQ(scratch[i], 0xAB) << "tail overwritten at i=" << i;
        }

        ASSERT_GE(out, c.tailIn) << "tail decreased";
        uint32_t appended = out - c.tailIn;

        const uint32_t minAppend = pad512(c.lastFileSize) + 1024u;
        ASSERT_GE(appended, minAppend) << "appended too small: appended=" << appended << " min=" << minAppend
                                       << " lastFileSize=" << c.lastFileSize << " tailIn=" << c.tailIn;

        // must end on 4KB boundary
        ASSERT_EQ(out & (4096u - 1u), 0u) << "not 4KB aligned: out=" << out << " lastFileSize=" << c.lastFileSize << " tailIn=" << c.tailIn;

        // appended region must be all zero
        for (uint32_t i = 0; i < appended; ++i) {
            ASSERT_EQ(scratch[c.tailIn + i], 0u) << "non-zero appended byte at offset=" << i << " (abs=" << (c.tailIn + i) << ")"
                                                 << " lastFileSize=" << c.lastFileSize << " tailIn=" << c.tailIn;
        }
    }
}
