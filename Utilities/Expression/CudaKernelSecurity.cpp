#include "Utilities/Expression/CudaKernelSecurity.h"
#include "Utilities/Expression/CudaKernelExpression.h"

#include <openssl/crypto.h>
#include <openssl/evp.h>
#include <openssl/hmac.h>
#include <openssl/rand.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using json = nlohmann::json;

namespace ThorImplementation {
namespace {

using EvpPkeyPtr = std::unique_ptr<EVP_PKEY, decltype(&EVP_PKEY_free)>;
using EvpMdCtxPtr = std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)>;
using EvpCipherCtxPtr = std::unique_ptr<EVP_CIPHER_CTX, decltype(&EVP_CIPHER_CTX_free)>;

std::string opensslErrorPrefix(const std::string& what) { return what + " failed while handling CudaKernelExpression source protection."; }

std::string hexEncode(const unsigned char* data, size_t len) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out;
    out.resize(len * 2);
    for (size_t i = 0; i < len; ++i) {
        out[2 * i] = kHex[(data[i] >> 4) & 0xF];
        out[2 * i + 1] = kHex[data[i] & 0xF];
    }
    return out;
}

std::vector<unsigned char> hexDecodeRaw(std::string value) {
    auto trim_prefix = [](std::string& s, const std::string& prefix) {
        if (s.rfind(prefix, 0) == 0) {
            s.erase(0, prefix.size());
        }
    };
    trim_prefix(value, "ed25519:");
    trim_prefix(value, "sha256:");
    trim_prefix(value, "aes256-gcm:");
    trim_prefix(value, "aes-256-gcm:");

    value.erase(std::remove_if(value.begin(), value.end(), [](unsigned char c) { return std::isspace(c) != 0 || c == ':' || c == '-'; }),
                value.end());

    if (value.size() % 2 != 0) {
        throw std::runtime_error("Hex value has an odd number of digits.");
    }

    auto nibble = [](char c) -> unsigned char {
        if (c >= '0' && c <= '9')
            return static_cast<unsigned char>(c - '0');
        if (c >= 'a' && c <= 'f')
            return static_cast<unsigned char>(10 + c - 'a');
        if (c >= 'A' && c <= 'F')
            return static_cast<unsigned char>(10 + c - 'A');
        throw std::runtime_error("Hex value contains a non-hex character.");
    };

    std::vector<unsigned char> out(value.size() / 2);
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = static_cast<unsigned char>((nibble(value[2 * i]) << 4) | nibble(value[2 * i + 1]));
    }
    return out;
}

std::string sha256Hex(const unsigned char* data, size_t len) {
    EvpMdCtxPtr ctx(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    if (!ctx || EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) <= 0 || EVP_DigestUpdate(ctx.get(), data, len) <= 0) {
        throw std::runtime_error(opensslErrorPrefix("EVP SHA-256 digest"));
    }
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int digest_len = 0;
    if (EVP_DigestFinal_ex(ctx.get(), digest, &digest_len) <= 0) {
        throw std::runtime_error(opensslErrorPrefix("EVP_DigestFinal_ex"));
    }
    return "sha256:" + hexEncode(digest, digest_len);
}

std::string sha256Hex(const std::string& bytes) { return sha256Hex(reinterpret_cast<const unsigned char*>(bytes.data()), bytes.size()); }

std::string sha256Hex(const std::vector<unsigned char>& bytes) { return sha256Hex(bytes.data(), bytes.size()); }

const std::vector<unsigned char>& processEphemeralSigningSecret() {
    static const std::vector<unsigned char> secret = [] {
        if (RAND_status() != 1) {
            throw std::runtime_error("OpenSSL CSPRNG is not seeded; refusing to initialize CudaKernelExpression signing secret.");
        }
        std::vector<unsigned char> value(32);
        if (RAND_bytes(value.data(), static_cast<int>(value.size())) != 1) {
            throw std::runtime_error(opensslErrorPrefix("RAND_bytes"));
        }
        return value;
    }();
    return secret;
}

std::vector<unsigned char> hmacSha256(const std::vector<unsigned char>& key, const std::string& message) {
    std::vector<unsigned char> digest(EVP_MAX_MD_SIZE);
    unsigned int digest_len = 0;
    unsigned char* result = HMAC(EVP_sha256(),
                                 key.data(),
                                 static_cast<int>(key.size()),
                                 reinterpret_cast<const unsigned char*>(message.data()),
                                 message.size(),
                                 digest.data(),
                                 &digest_len);
    if (result == nullptr || digest_len != 32) {
        throw std::runtime_error(opensslErrorPrefix("HMAC-SHA256"));
    }
    digest.resize(digest_len);
    return digest;
}

std::string normalizeEd25519PublicKey(const std::string& public_key) {
    std::vector<unsigned char> raw = hexDecodeRaw(public_key);
    if (raw.size() != 32) {
        throw std::runtime_error(
            "CudaKernelExpression trusted Ed25519 public key must be 32 raw bytes encoded as hex, optionally prefixed with 'ed25519:'.");
    }
    return "ed25519:" + hexEncode(raw.data(), raw.size());
}

std::string normalizeAes256GcmSourceDecryptionKey(const std::string& source_decryption_key) {
    std::vector<unsigned char> raw = hexDecodeRaw(source_decryption_key);
    if (raw.size() != 32) {
        throw std::runtime_error(
            "CudaKernelExpression source decryption key must be 32 raw bytes encoded as hex, optionally prefixed with 'aes256-gcm:'.");
    }
    return "aes256-gcm:" + hexEncode(raw.data(), raw.size());
}

std::mutex& signingPublicKeyRegistryMutex() {
    static std::mutex mutex;
    return mutex;
}

std::unordered_map<std::string, std::string>& signingPublicKeyRegistry() {
    static std::unordered_map<std::string, std::string> registry;
    return registry;
}

std::mutex& sourceDecryptionKeyRegistryMutex() {
    static std::mutex mutex;
    return mutex;
}

std::unordered_map<std::string, std::string>& sourceDecryptionKeyRegistry() {
    static std::unordered_map<std::string, std::string> registry;
    return registry;
}

std::string publicKeyFingerprintFromRaw(const std::vector<unsigned char>& public_key) { return sha256Hex(public_key); }

std::string sourceDecryptionKeyFingerprintFromRaw(const std::vector<unsigned char>& source_decryption_key) {
    return sha256Hex(source_decryption_key);
}

bool fingerprintContainsRawKeyMaterial(const std::string& fingerprint, const std::vector<unsigned char>& key) {
    try {
        return hexDecodeRaw(fingerprint) == key;
    } catch (const std::exception&) {
        return false;
    }
}

bool publicKeyFingerprintContainsPublicKeyMaterial(const std::string& public_key_fingerprint,
                                                   const std::vector<unsigned char>& public_key) {
    return fingerprintContainsRawKeyMaterial(public_key_fingerprint, public_key);
}

void requireFingerprintDoesNotContainKeyMaterial(const std::string& fingerprint,
                                                 const std::vector<unsigned char>& key,
                                                 const char* error_message) {
    if (fingerprintContainsRawKeyMaterial(fingerprint, key)) {
        throw std::logic_error(error_message);
    }
}

std::string publicKeyFingerprintFromText(const std::string& public_key) {
    return publicKeyFingerprintFromRaw(hexDecodeRaw(normalizeEd25519PublicKey(public_key)));
}

std::string sourceDecryptionKeyFingerprintFromText(const std::string& source_decryption_key) {
    return sourceDecryptionKeyFingerprintFromRaw(hexDecodeRaw(normalizeAes256GcmSourceDecryptionKey(source_decryption_key)));
}

void rememberEphemeralSigningPublicKey(const std::string& public_key_text) {
    const std::string fingerprint = publicKeyFingerprintFromText(public_key_text);
    std::lock_guard<std::mutex> lock(signingPublicKeyRegistryMutex());
    signingPublicKeyRegistry()[fingerprint] = public_key_text;
}

void rememberEphemeralSourceDecryptionKey(const std::string& source_decryption_key_text) {
    const std::string fingerprint = sourceDecryptionKeyFingerprintFromText(source_decryption_key_text);
    std::lock_guard<std::mutex> lock(sourceDecryptionKeyRegistryMutex());
    sourceDecryptionKeyRegistry()[fingerprint] = source_decryption_key_text;
}

struct CudaKernelManifestSigningKey {
    EvpPkeyPtr key;
    std::vector<unsigned char> public_key;
    std::string public_key_text;
    std::string public_key_fingerprint;

    CudaKernelManifestSigningKey(EvpPkeyPtr&& key_in,
                                 std::vector<unsigned char>&& public_key_in,
                                 std::string public_key_text_in,
                                 std::string public_key_fingerprint_in)
        : key(std::move(key_in)),
          public_key(std::move(public_key_in)),
          public_key_text(std::move(public_key_text_in)),
          public_key_fingerprint(std::move(public_key_fingerprint_in)) {}
};

struct CudaKernelSourceDecryptionKey {
    std::vector<unsigned char> key;
    std::string key_text;
    std::string key_fingerprint;
};

CudaKernelManifestSigningKey cudaKernelSigningKeyFromSeed(const std::vector<unsigned char>& seed) {
    if (seed.size() != 32) {
        throw std::logic_error("Internal CudaKernelExpression signing error: Ed25519 seed must be 32 bytes.");
    }

    EvpPkeyPtr key(EVP_PKEY_new_raw_private_key(EVP_PKEY_ED25519, nullptr, seed.data(), seed.size()), EVP_PKEY_free);
    if (!key) {
        throw std::runtime_error(opensslErrorPrefix("EVP_PKEY_new_raw_private_key"));
    }

    std::vector<unsigned char> public_key(32);
    size_t public_key_len = public_key.size();
    if (EVP_PKEY_get_raw_public_key(key.get(), public_key.data(), &public_key_len) <= 0 || public_key_len != public_key.size()) {
        throw std::runtime_error(opensslErrorPrefix("EVP_PKEY_get_raw_public_key"));
    }

    const std::string public_key_text = "ed25519:" + hexEncode(public_key.data(), public_key.size());
    const std::string public_key_fingerprint = publicKeyFingerprintFromRaw(public_key);
    requireFingerprintDoesNotContainKeyMaterial(
        public_key_fingerprint,
        public_key,
        "Internal CudaKernelExpression signing error: public_key_fingerprint contains public key material instead of a SHA-256 digest.");
    rememberEphemeralSigningPublicKey(public_key_text);

    return CudaKernelManifestSigningKey(std::move(key), std::move(public_key), public_key_text, public_key_fingerprint);
}

CudaKernelManifestSigningKey cudaKernelDeriveManifestSigningKey(const std::string& signing_context) {
    // The private signing key is derived from a process-local CSPRNG secret and
    // the encrypted, unsigned serialization payload. This keeps the signing key
    // ephemeral and not user-provided while making repeated signatures for an
    // identical protected payload deterministic within one process. The source
    // decryption key is generated separately and never stored in the manifest.
    constexpr const char* kDomain = "thor.cuda_kernel_expression.manifest_signing_key.v2\n";
    const std::vector<unsigned char> seed = hmacSha256(processEphemeralSigningSecret(), std::string(kDomain) + signing_context);
    return cudaKernelSigningKeyFromSeed(seed);
}

CudaKernelSourceDecryptionKey cudaKernelDeriveSourceDecryptionKey(const std::string& encryption_context) {
    // The source decryption key is process-ephemeral and out-of-band, just like
    // the signing public key. Deriving it from the unsigned plaintext payload
    // keeps repeated in-process serialization/key queries stable without ever
    // writing the key into the model manifest.
    constexpr const char* kDomain = "thor.cuda_kernel_expression.source_decryption_key.v1\n";
    CudaKernelSourceDecryptionKey out;
    out.key = hmacSha256(processEphemeralSigningSecret(), std::string(kDomain) + encryption_context);
    out.key_text = "aes256-gcm:" + hexEncode(out.key.data(), out.key.size());
    out.key_fingerprint = sourceDecryptionKeyFingerprintFromRaw(out.key);
    requireFingerprintDoesNotContainKeyMaterial(
        out.key_fingerprint,
        out.key,
        "Internal CudaKernelExpression encryption error: source_decryption_key_fingerprint contains key material instead of a SHA-256 digest.");
    rememberEphemeralSourceDecryptionKey(out.key_text);
    return out;
}

CudaKernelSourceDecryptionKey sourceDecryptionKeyFromText(const std::string& key_text) {
    CudaKernelSourceDecryptionKey out;
    out.key_text = normalizeAes256GcmSourceDecryptionKey(key_text);
    out.key = hexDecodeRaw(out.key_text);
    out.key_fingerprint = sourceDecryptionKeyFingerprintFromRaw(out.key);
    requireFingerprintDoesNotContainKeyMaterial(
        out.key_fingerprint,
        out.key,
        "Internal CudaKernelExpression encryption error: source_decryption_key_fingerprint contains key material instead of a SHA-256 digest.");
    return out;
}

std::string lookupEphemeralSigningPublicKeyByFingerprint(const std::string& fingerprint) {
    std::lock_guard<std::mutex> lock(signingPublicKeyRegistryMutex());
    auto it = signingPublicKeyRegistry().find(fingerprint);
    if (it == signingPublicKeyRegistry().end()) {
        return {};
    }
    return it->second;
}

std::string lookupEphemeralSourceDecryptionKeyByFingerprint(const std::string& fingerprint) {
    std::lock_guard<std::mutex> lock(sourceDecryptionKeyRegistryMutex());
    auto it = sourceDecryptionKeyRegistry().find(fingerprint);
    if (it == sourceDecryptionKeyRegistry().end()) {
        return {};
    }
    return it->second;
}

void requireSizeFitsOpenSslInt(size_t size, const char* what) {
    if (size > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string("CudaKernelExpression ") + what + " is too large for OpenSSL EVP one-shot processing.");
    }
}

struct Aes256GcmEncryptedBytes {
    std::vector<unsigned char> nonce;
    std::vector<unsigned char> ciphertext;
    std::vector<unsigned char> tag;
};

Aes256GcmEncryptedBytes aes256GcmEncrypt(const std::vector<unsigned char>& key,
                                           const std::vector<unsigned char>& nonce,
                                           const std::string& plaintext) {
    if (key.size() != 32) {
        throw std::logic_error("AES-256-GCM source encryption requires a 32-byte key.");
    }
    if (nonce.size() != 12) {
        throw std::logic_error("AES-256-GCM source encryption requires a 12-byte nonce.");
    }
    requireSizeFitsOpenSslInt(plaintext.size(), "source plaintext");

    Aes256GcmEncryptedBytes out;
    out.nonce = nonce;
    out.tag.resize(16);

    EvpCipherCtxPtr ctx(EVP_CIPHER_CTX_new(), EVP_CIPHER_CTX_free);
    if (!ctx || EVP_EncryptInit_ex(ctx.get(), EVP_aes_256_gcm(), nullptr, nullptr, nullptr) != 1 ||
        EVP_CIPHER_CTX_ctrl(ctx.get(), EVP_CTRL_GCM_SET_IVLEN, static_cast<int>(out.nonce.size()), nullptr) != 1 ||
        EVP_EncryptInit_ex(ctx.get(), nullptr, nullptr, key.data(), out.nonce.data()) != 1) {
        throw std::runtime_error(opensslErrorPrefix("EVP AES-256-GCM encrypt init"));
    }

    out.ciphertext.resize(plaintext.size() + 16);
    int written = 0;
    if (!plaintext.empty() && EVP_EncryptUpdate(ctx.get(),
                                                out.ciphertext.data(),
                                                &written,
                                                reinterpret_cast<const unsigned char*>(plaintext.data()),
                                                static_cast<int>(plaintext.size())) != 1) {
        throw std::runtime_error(opensslErrorPrefix("EVP_EncryptUpdate"));
    }
    int final_written = 0;
    if (EVP_EncryptFinal_ex(ctx.get(), out.ciphertext.data() + written, &final_written) != 1) {
        throw std::runtime_error(opensslErrorPrefix("EVP_EncryptFinal_ex"));
    }
    out.ciphertext.resize(static_cast<size_t>(written + final_written));

    if (EVP_CIPHER_CTX_ctrl(ctx.get(), EVP_CTRL_GCM_GET_TAG, static_cast<int>(out.tag.size()), out.tag.data()) != 1) {
        throw std::runtime_error(opensslErrorPrefix("EVP_CTRL_GCM_GET_TAG"));
    }
    return out;
}

std::string aes256GcmDecrypt(const std::vector<unsigned char>& key,
                             const std::vector<unsigned char>& nonce,
                             const std::vector<unsigned char>& ciphertext,
                             const std::vector<unsigned char>& tag) {
    if (key.size() != 32) {
        throw std::logic_error("AES-256-GCM source decryption requires a 32-byte key.");
    }
    if (nonce.size() != 12) {
        throw std::runtime_error("CudaKernelExpression AES-256-GCM nonce must be 12 bytes.");
    }
    if (tag.size() != 16) {
        throw std::runtime_error("CudaKernelExpression AES-256-GCM tag must be 16 bytes.");
    }
    requireSizeFitsOpenSslInt(ciphertext.size(), "source ciphertext");

    EvpCipherCtxPtr ctx(EVP_CIPHER_CTX_new(), EVP_CIPHER_CTX_free);
    if (!ctx || EVP_DecryptInit_ex(ctx.get(), EVP_aes_256_gcm(), nullptr, nullptr, nullptr) != 1 ||
        EVP_CIPHER_CTX_ctrl(ctx.get(), EVP_CTRL_GCM_SET_IVLEN, static_cast<int>(nonce.size()), nullptr) != 1 ||
        EVP_DecryptInit_ex(ctx.get(), nullptr, nullptr, key.data(), nonce.data()) != 1) {
        throw std::runtime_error(opensslErrorPrefix("EVP AES-256-GCM decrypt init"));
    }

    std::vector<unsigned char> plaintext(ciphertext.size() + 16);
    int written = 0;
    if (!ciphertext.empty() &&
        EVP_DecryptUpdate(ctx.get(), plaintext.data(), &written, ciphertext.data(), static_cast<int>(ciphertext.size())) != 1) {
        throw std::runtime_error(opensslErrorPrefix("EVP_DecryptUpdate"));
    }
    if (EVP_CIPHER_CTX_ctrl(ctx.get(), EVP_CTRL_GCM_SET_TAG, static_cast<int>(tag.size()), const_cast<unsigned char*>(tag.data())) != 1) {
        throw std::runtime_error(opensslErrorPrefix("EVP_CTRL_GCM_SET_TAG"));
    }
    int final_written = 0;
    const int ok = EVP_DecryptFinal_ex(ctx.get(), plaintext.data() + written, &final_written);
    if (ok != 1) {
        throw std::runtime_error("CudaKernelExpression encrypted CUDA source authentication failed; wrong decryption key or corrupted ciphertext.");
    }
    plaintext.resize(static_cast<size_t>(written + final_written));
    return std::string(reinterpret_cast<const char*>(plaintext.data()), plaintext.size());
}

json cudaKernelJsonWithoutManifestSignatures(json value) {
    if (value.is_object()) {
        value.erase("cuda_kernel_manifest_signature");
        for (auto iter = value.begin(); iter != value.end(); ++iter) {
            iter.value() = cudaKernelJsonWithoutManifestSignatures(iter.value());
        }
    } else if (value.is_array()) {
        for (json& item : value) {
            item = cudaKernelJsonWithoutManifestSignatures(item);
        }
    }
    return value;
}

std::string cudaKernelSigningContextForRootJson(const json& root_json) { return cudaKernelJsonWithoutManifestSignatures(root_json).dump(); }

bool isExpressionWithCudaKernels(const json& value) {
    return value.is_object() && value.value("type", std::string{}) == "thor.expression" && value.contains("cuda_kernels") &&
           value.at("cuda_kernels").is_array() && !value.at("cuda_kernels").empty();
}

bool jsonContainsCudaKernelExpressions(const json& value) {
    if (isExpressionWithCudaKernels(value)) {
        return true;
    }
    if (value.is_object()) {
        for (auto iter = value.begin(); iter != value.end(); ++iter) {
            if (jsonContainsCudaKernelExpressions(iter.value())) {
                return true;
            }
        }
    } else if (value.is_array()) {
        for (const json& item : value) {
            if (jsonContainsCudaKernelExpressions(item)) {
                return true;
            }
        }
    }
    return false;
}

bool kernelJsonContainsEncryptedSource(const json& kernel_json) {
    return kernel_json.contains("source_encryption") || kernel_json.contains("encrypted_source");
}

std::vector<unsigned char> deriveSourceEncryptionNonce(const CudaKernelSourceDecryptionKey& key, size_t kernel_ordinal, const json& kernel_json) {
    constexpr const char* kDomain = "thor.cuda_kernel_expression.source_nonce.v1\n";
    const std::string nonce_context = std::string(kDomain) + std::to_string(kernel_ordinal) + "\n" + kernel_json.dump();
    std::vector<unsigned char> digest = hmacSha256(key.key, nonce_context);
    digest.resize(12);
    return digest;
}

void encryptCudaKernelSourcesInExpression(json& expression_json, const CudaKernelSourceDecryptionKey& key, size_t& kernel_ordinal) {
    if (!isExpressionWithCudaKernels(expression_json)) {
        return;
    }
    for (json& kernel_json : expression_json.at("cuda_kernels")) {
        if (kernelJsonContainsEncryptedSource(kernel_json)) {
            continue;
        }
        if (!kernel_json.value("loaded_source_compilation_allowed", false)) {
            throw std::runtime_error(
                "Refusing to encrypt/sign CudaKernelExpression CUDA source that was loaded from a serialized model without trusted-key "
                "compilation enabled. Inspecting or saving an untrusted loaded expression must not mint a new protected artifact.");
        }
        const std::string source = kernel_json.at("source").get<std::string>();
        const std::vector<unsigned char> nonce = deriveSourceEncryptionNonce(key, kernel_ordinal++, kernel_json);
        Aes256GcmEncryptedBytes encrypted = aes256GcmEncrypt(key.key, nonce, source);
        kernel_json["encrypted_source"] = hexEncode(encrypted.ciphertext.data(), encrypted.ciphertext.size());
        kernel_json["source_encryption"] = json{{"schema_version", 1},
                                                {"algorithm", "aes-256-gcm"},
                                                {"source_decryption_key_fingerprint", key.key_fingerprint},
                                                {"nonce", hexEncode(encrypted.nonce.data(), encrypted.nonce.size())},
                                                {"tag", hexEncode(encrypted.tag.data(), encrypted.tag.size())}};
        kernel_json.erase("source");
    }
}

bool encryptCudaKernelSourcesRecursive(json& value, const CudaKernelSourceDecryptionKey& key, size_t& kernel_ordinal) {
    bool protected_any = false;
    if (value.is_object()) {
        if (isExpressionWithCudaKernels(value)) {
            encryptCudaKernelSourcesInExpression(value, key, kernel_ordinal);
            protected_any = true;
        }
        for (auto iter = value.begin(); iter != value.end(); ++iter) {
            protected_any = encryptCudaKernelSourcesRecursive(iter.value(), key, kernel_ordinal) || protected_any;
        }
    } else if (value.is_array()) {
        for (json& item : value) {
            protected_any = encryptCudaKernelSourcesRecursive(item, key, kernel_ordinal) || protected_any;
        }
    }
    return protected_any;
}

json canonicalKernelJson(const json& kernel_json) {
    json kernel;
    kernel["schema_version"] = kernel_json.at("schema_version");
    kernel["name"] = kernel_json.at("name");
    if (kernelJsonContainsEncryptedSource(kernel_json)) {
        kernel["encrypted_source"] = kernel_json.at("encrypted_source");
        kernel["source_encryption"] = kernel_json.at("source_encryption");
    } else {
        kernel["source"] = kernel_json.at("source");
    }
    kernel["entry"] = kernel_json.at("entry");
    kernel["use_fast_math"] = kernel_json.value("use_fast_math", false);
    kernel["compiled_source_hash"] = kernel_json.value("compiled_source_hash", std::string{});
    kernel["inputs"] = kernel_json.at("inputs");
    kernel["outputs"] = kernel_json.at("outputs");
    kernel["scalars"] = kernel_json.at("scalars");
    kernel["launch"] = kernel_json.at("launch");
    return kernel;
}

json signaturePayloadForExpression(const json& expression_json) {
    json payload;
    payload["schema_version"] = 1;
    payload["type"] = "thor.cuda_kernel_expression_manifest";
    payload["expression_type"] = expression_json.at("type");
    payload["expression_schema_version"] = expression_json.at("schema_version");
    payload["expression_canonical_hash"] = expression_json.value("canonical_hash", std::string{});
    payload["inputs"] = expression_json.at("inputs");
    payload["nodes"] = expression_json.at("nodes");
    payload["outputs"] = expression_json.at("outputs");
    payload["expected_input_names"] = expression_json.value("expected_input_names", std::vector<std::string>{});
    payload["expected_output_names"] = expression_json.value("expected_output_names", std::vector<std::string>{});

    payload["cuda_kernels"] = json::array();
    for (const json& kernel_json : expression_json.at("cuda_kernels")) {
        payload["cuda_kernels"].push_back(canonicalKernelJson(kernel_json));
    }
    return payload;
}

std::string cudaKernelAttachManifestSignatureWithKey(json& expression_json, const CudaKernelManifestSigningKey& signing_key) {
    if (!isExpressionWithCudaKernels(expression_json)) {
        return {};
    }

    const json manifest = cudaKernelManifestFromExpressionJson(expression_json);
    const std::string manifest_bytes = cudaKernelManifestCanonicalBytes(manifest);

    EvpMdCtxPtr md_ctx(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    if (!md_ctx || EVP_DigestSignInit(md_ctx.get(), nullptr, nullptr, nullptr, signing_key.key.get()) <= 0) {
        throw std::runtime_error(opensslErrorPrefix("EVP_DigestSignInit"));
    }

    std::vector<unsigned char> signature(64);
    size_t signature_len = signature.size();
    if (EVP_DigestSign(md_ctx.get(),
                       signature.data(),
                       &signature_len,
                       reinterpret_cast<const unsigned char*>(manifest_bytes.data()),
                       manifest_bytes.size()) <= 0) {
        throw std::runtime_error(opensslErrorPrefix("EVP_DigestSign"));
    }
    signature.resize(signature_len);

    expression_json["cuda_kernel_manifest_signature"] = json{{"schema_version", 1},
                                                             {"algorithm", "ed25519"},
                                                             {"public_key_fingerprint", signing_key.public_key_fingerprint},
                                                             {"manifest_sha256", sha256Hex(manifest_bytes)},
                                                             {"signature", "ed25519:" + hexEncode(signature.data(), signature.size())},
                                                             {"disclaimer", cudaKernelLoadedModelSafetyDisclaimer()}};
    return signing_key.public_key_text;
}

void signCudaKernelExpressionsRecursive(json& value, const CudaKernelManifestSigningKey& signing_key, bool& signed_any) {
    if (value.is_object()) {
        if (isExpressionWithCudaKernels(value)) {
            (void)cudaKernelAttachManifestSignatureWithKey(value, signing_key);
            signed_any = true;
        }
        for (auto iter = value.begin(); iter != value.end(); ++iter) {
            signCudaKernelExpressionsRecursive(iter.value(), signing_key, signed_any);
        }
    } else if (value.is_array()) {
        for (json& item : value) {
            signCudaKernelExpressionsRecursive(item, signing_key, signed_any);
        }
    }
}

void collectCudaKernelSigningPublicKeysRecursive(const json& j, std::set<std::string>& keys) {
    if (j.is_object()) {
        auto it = j.find("cuda_kernel_manifest_signature");
        if (it != j.end() && it->is_object()) {
            const std::string fingerprint = it->value("public_key_fingerprint", std::string{});
            if (!fingerprint.empty()) {
                const std::string public_key = lookupEphemeralSigningPublicKeyByFingerprint(fingerprint);
                if (!public_key.empty()) {
                    keys.insert(public_key);
                }
            }
        }
        for (auto iter = j.begin(); iter != j.end(); ++iter) {
            collectCudaKernelSigningPublicKeysRecursive(iter.value(), keys);
        }
    } else if (j.is_array()) {
        for (const json& item : j) {
            collectCudaKernelSigningPublicKeysRecursive(item, keys);
        }
    }
}

void collectCudaKernelOutOfBandKeysRecursive(const json& j, std::set<std::pair<std::string, std::string>>& keys) {
    if (j.is_object()) {
        if (isExpressionWithCudaKernels(j)) {
            std::string signing_public_key;
            auto sig_it = j.find("cuda_kernel_manifest_signature");
            if (sig_it != j.end() && sig_it->is_object()) {
                const std::string fingerprint = sig_it->value("public_key_fingerprint", std::string{});
                if (!fingerprint.empty()) {
                    signing_public_key = lookupEphemeralSigningPublicKeyByFingerprint(fingerprint);
                }
            }
            for (const json& kernel : j.at("cuda_kernels")) {
                std::string source_decryption_key;
                auto enc_it = kernel.find("source_encryption");
                if (enc_it != kernel.end() && enc_it->is_object()) {
                    const std::string fingerprint = enc_it->value("source_decryption_key_fingerprint", std::string{});
                    if (!fingerprint.empty()) {
                        source_decryption_key = lookupEphemeralSourceDecryptionKeyByFingerprint(fingerprint);
                    }
                }
                if (!signing_public_key.empty() || !source_decryption_key.empty()) {
                    keys.emplace(signing_public_key, source_decryption_key);
                }
            }
        }
        for (auto iter = j.begin(); iter != j.end(); ++iter) {
            collectCudaKernelOutOfBandKeysRecursive(iter.value(), keys);
        }
    } else if (j.is_array()) {
        for (const json& item : j) {
            collectCudaKernelOutOfBandKeysRecursive(item, keys);
        }
    }
}

void collectCudaKernelSourceInfoRecursive(const json& j, std::vector<CudaKernelSourceInspection>& out) {
    if (j.is_object()) {
        if (j.contains("cuda_kernels") && j.at("cuda_kernels").is_array()) {
            const json* signature = nullptr;
            auto sig_it = j.find("cuda_kernel_manifest_signature");
            if (sig_it != j.end() && sig_it->is_object()) {
                signature = &(*sig_it);
            }
            for (const json& kernel : j.at("cuda_kernels")) {
                CudaKernelSourceInspection info;
                info.name = kernel.value("name", std::string{});
                info.entrypoint = kernel.value("entry", std::string{});
                info.compiled_source_hash = kernel.value("compiled_source_hash", std::string{});
                info.source_encrypted = kernelJsonContainsEncryptedSource(kernel);
                if (info.source_encrypted) {
                    const json& encryption = kernel.at("source_encryption");
                    info.source_encryption_algorithm = encryption.value("algorithm", std::string{});
                    info.source_decryption_key_fingerprint = encryption.value("source_decryption_key_fingerprint", std::string{});
                } else {
                    info.source = kernel.value("source", std::string{});
                    info.compiled_source = cudaKernelExpressionCompiledSourceForInspection(info.source);
                }
                // Raw JSON inspection is conservative: a serialized field may reflect that the model was saved from a local,
                // compilable expression, but loaded CUDA source is not actually compilable until the caller opts in with trusted keys.
                info.loaded_source_compilation_allowed = false;
                if (signature != nullptr) {
                    info.signature_algorithm = signature->value("algorithm", std::string{});
                    info.signing_public_key_fingerprint = signature->value("public_key_fingerprint", std::string{});
                    info.signature = signature->value("signature", std::string{});
                }
                out.push_back(std::move(info));
            }
        }
        for (auto iter = j.begin(); iter != j.end(); ++iter) {
            collectCudaKernelSourceInfoRecursive(iter.value(), out);
        }
    } else if (j.is_array()) {
        for (const json& item : j) {
            collectCudaKernelSourceInfoRecursive(item, out);
        }
    }
}

}  // namespace

json cudaKernelSourceInspectionToJson(const CudaKernelSourceInspection& info) {
    json entry{{"name", info.name},
               {"entrypoint", info.entrypoint},
               {"source", info.source},
               {"compiled_source", info.compiled_source},
               {"compiled_source_hash", info.compiled_source_hash},
               {"loaded_source_compilation_allowed", info.loaded_source_compilation_allowed},
               {"source_encrypted", info.source_encrypted}};
    if (!info.source_encryption_algorithm.empty()) {
        entry["source_encryption_algorithm"] = info.source_encryption_algorithm;
    }
    if (!info.source_decryption_key_fingerprint.empty()) {
        entry["source_decryption_key_fingerprint"] = info.source_decryption_key_fingerprint;
    }
    if (!info.signature_algorithm.empty()) {
        entry["signature_algorithm"] = info.signature_algorithm;
    }
    if (!info.signing_public_key_fingerprint.empty()) {
        entry["signing_public_key_fingerprint"] = info.signing_public_key_fingerprint;
    }
    if (!info.signature.empty()) {
        entry["signature"] = info.signature;
    }
    return entry;
}

json cudaKernelSourceInspectionListToJson(const std::vector<CudaKernelSourceInspection>& infos) {
    json result = json::array();
    for (const CudaKernelSourceInspection& info : infos) {
        result.push_back(cudaKernelSourceInspectionToJson(info));
    }
    return result;
}

json cudaKernelManifestFromExpressionJson(const json& expression_json) { return signaturePayloadForExpression(expression_json); }

std::string cudaKernelManifestCanonicalBytes(const json& manifest) { return manifest.dump(); }

CudaKernelOutOfBandKeys cudaKernelGenerateAndAttachManifestSignature(json& expression_json) {
    std::vector<CudaKernelOutOfBandKeys> keys = cudaKernelGenerateAndAttachManifestSignatures(expression_json);
    if (keys.empty()) {
        return {};
    }
    return keys.front();
}

std::vector<CudaKernelOutOfBandKeys> cudaKernelGenerateAndAttachManifestSignatures(json& root_json) {
    if (!jsonContainsCudaKernelExpressions(root_json)) {
        return {};
    }

    const std::string encryption_context = cudaKernelSigningContextForRootJson(root_json);
    CudaKernelSourceDecryptionKey decryption_key = cudaKernelDeriveSourceDecryptionKey(encryption_context);
    size_t kernel_ordinal = 0;
    const bool protected_any = encryptCudaKernelSourcesRecursive(root_json, decryption_key, kernel_ordinal);
    if (!protected_any) {
        return {};
    }

    const std::string signing_context = cudaKernelSigningContextForRootJson(root_json);
    const CudaKernelManifestSigningKey signing_key = cudaKernelDeriveManifestSigningKey(signing_context);

    bool signed_any = false;
    signCudaKernelExpressionsRecursive(root_json, signing_key, signed_any);
    if (!signed_any) {
        return {};
    }

    return {CudaKernelOutOfBandKeys{.signing_public_key = signing_key.public_key_text,
                                    .source_decryption_key = decryption_key.key_text}};
}

std::vector<CudaKernelOutOfBandKeys> collectCudaKernelOutOfBandKeys(const json& j) {
    std::set<std::pair<std::string, std::string>> pairs;
    collectCudaKernelOutOfBandKeysRecursive(j, pairs);
    std::vector<CudaKernelOutOfBandKeys> out;
    out.reserve(pairs.size());
    for (const auto& [signing_public_key, source_decryption_key] : pairs) {
        out.push_back(CudaKernelOutOfBandKeys{.signing_public_key = signing_public_key, .source_decryption_key = source_decryption_key});
    }
    return out;
}

bool cudaKernelExpressionJsonContainsEncryptedSources(const json& expression_json) {
    if (!expression_json.contains("cuda_kernels") || !expression_json.at("cuda_kernels").is_array()) {
        return false;
    }
    for (const json& kernel : expression_json.at("cuda_kernels")) {
        if (kernelJsonContainsEncryptedSource(kernel)) {
            return true;
        }
    }
    return false;
}

bool cudaKernelExpressionJsonContainsPlaintextSources(const json& expression_json) {
    if (!expression_json.contains("cuda_kernels") || !expression_json.at("cuda_kernels").is_array()) {
        return false;
    }
    for (const json& kernel : expression_json.at("cuda_kernels")) {
        if (kernel.contains("source")) {
            return true;
        }
    }
    return false;
}

json cudaKernelDecryptSerializedCudaSources(const json& expression_json, const std::string& trusted_source_decryption_key) {
    if (!cudaKernelExpressionJsonContainsEncryptedSources(expression_json)) {
        return expression_json;
    }
    if (trusted_source_decryption_key.empty()) {
        throw std::runtime_error(
            "A CudaKernelExpression AES-256-GCM source decryption key is required to load encrypted CUDA source. The key is printed "
            "out-of-band when the model is saved and is never stored in the model manifest.");
    }

    CudaKernelSourceDecryptionKey key = sourceDecryptionKeyFromText(trusted_source_decryption_key);
    json decrypted = expression_json;
    for (json& kernel : decrypted.at("cuda_kernels")) {
        if (!kernelJsonContainsEncryptedSource(kernel)) {
            continue;
        }
        const json& encryption = kernel.at("source_encryption");
        if (encryption.value("schema_version", 0) != 1 || encryption.value("algorithm", std::string{}) != "aes-256-gcm") {
            throw std::runtime_error("Unsupported CudaKernelExpression source encryption metadata.");
        }
        const std::string expected_key_fingerprint = encryption.value("source_decryption_key_fingerprint", std::string{});
        if (expected_key_fingerprint.empty()) {
            throw std::runtime_error("Serialized CudaKernelExpression source encryption metadata does not contain a source_decryption_key_fingerprint.");
        }
        if (fingerprintContainsRawKeyMaterial(expected_key_fingerprint, key.key)) {
            throw std::runtime_error(
                "Serialized CudaKernelExpression source_decryption_key_fingerprint contains key material instead of a SHA-256 digest; "
                "refusing to decrypt loaded CUDA source.");
        }
        if (expected_key_fingerprint != key.key_fingerprint) {
            throw std::runtime_error(
                "The CudaKernelExpression source decryption key provided by the caller does not match the key fingerprint recorded with "
                "the saved model's encrypted CUDA source.");
        }

        const std::vector<unsigned char> nonce = hexDecodeRaw(encryption.at("nonce").get<std::string>());
        const std::vector<unsigned char> tag = hexDecodeRaw(encryption.at("tag").get<std::string>());
        const std::vector<unsigned char> ciphertext = hexDecodeRaw(kernel.at("encrypted_source").get<std::string>());
        const std::string source = aes256GcmDecrypt(key.key, nonce, ciphertext, tag);
        // CudaKernelExpression::deserialize validates compiled_source_hash after
        // the plaintext source is restored. Keep that check in the expression
        // implementation so it uses the same stable hash function as saving.
        kernel["source"] = source;
        kernel.erase("encrypted_source");
        kernel.erase("source_encryption");
    }
    return decrypted;
}

CudaKernelSignatureVerificationResult cudaKernelVerifyManifestSignature(const json& expression_json,
                                                                        const std::string& trusted_public_key) {
    if (!expression_json.contains("cuda_kernels") || expression_json.at("cuda_kernels").empty()) {
        return {true, "Expression contains no CudaKernelExpression CUDA source."};
    }
    if (trusted_public_key.empty()) {
        return {false,
                "A trusted Ed25519 public key is required to load or compile CudaKernelExpression CUDA source from a saved model. "
                "Serialized CudaKernelExpression CUDA source must be encrypted and requires the out-of-band signing public key and "
                "source decryption key printed when the model was saved."};
    }
    if (!expression_json.contains("cuda_kernel_manifest_signature")) {
        return {false, "Serialized expression contains CudaKernelExpression CUDA source but no cuda_kernel_manifest_signature."};
    }

    const json& sig_json = expression_json.at("cuda_kernel_manifest_signature");
    if (sig_json.value("schema_version", 0) != 1 || sig_json.value("algorithm", std::string{}) != "ed25519") {
        return {false, "Unsupported CudaKernelExpression CUDA manifest signature metadata."};
    }

    const std::string normalized_trusted_key = normalizeEd25519PublicKey(trusted_public_key);
    std::vector<unsigned char> public_key = hexDecodeRaw(normalized_trusted_key);
    const std::string expected_public_key_fingerprint = sig_json.value("public_key_fingerprint", std::string{});
    if (expected_public_key_fingerprint.empty()) {
        return {false, "Serialized CudaKernelExpression manifest signature does not contain a public_key_fingerprint."};
    }
    if (publicKeyFingerprintContainsPublicKeyMaterial(expected_public_key_fingerprint, public_key)) {
        return {false,
                "Serialized CudaKernelExpression public_key_fingerprint contains public key material instead of a SHA-256 digest; refusing "
                "to compile loaded CUDA source."};
    }
    if (expected_public_key_fingerprint != publicKeyFingerprintFromRaw(public_key)) {
        return {false,
                "The trusted Ed25519 public key provided by the caller does not match the public-key fingerprint recorded with the saved "
                "model's CudaKernelExpression manifest signature."};
    }

    std::vector<unsigned char> signature = hexDecodeRaw(sig_json.at("signature").get<std::string>());
    if (signature.size() != 64) {
        return {false, "CudaKernelExpression Ed25519 signature must be 64 raw bytes encoded as hex."};
    }

    const json manifest = cudaKernelManifestFromExpressionJson(expression_json);
    const std::string manifest_bytes = cudaKernelManifestCanonicalBytes(manifest);
    const std::string expected_manifest_hash = sig_json.value("manifest_sha256", std::string{});
    if (!expected_manifest_hash.empty() && expected_manifest_hash != sha256Hex(manifest_bytes)) {
        return {false, "CudaKernelExpression CUDA manifest SHA-256 hash does not match the signed metadata."};
    }

    EvpPkeyPtr key(EVP_PKEY_new_raw_public_key(EVP_PKEY_ED25519, nullptr, public_key.data(), public_key.size()), EVP_PKEY_free);
    if (!key) {
        return {false, opensslErrorPrefix("EVP_PKEY_new_raw_public_key")};
    }
    EvpMdCtxPtr md_ctx(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    if (!md_ctx || EVP_DigestVerifyInit(md_ctx.get(), nullptr, nullptr, nullptr, key.get()) <= 0) {
        return {false, opensslErrorPrefix("EVP_DigestVerifyInit")};
    }

    const int ok = EVP_DigestVerify(md_ctx.get(),
                                    signature.data(),
                                    signature.size(),
                                    reinterpret_cast<const unsigned char*>(manifest_bytes.data()),
                                    manifest_bytes.size());
    if (ok != 1) {
        return {false,
                "CudaKernelExpression CUDA manifest signature verification failed. The encrypted CUDA source, ABI, launch policy, or "
                "expression graph may have been modified, or the wrong public key was supplied."};
    }
    return {true, "CudaKernelExpression CUDA manifest signature verified with the trusted Ed25519 public key."};
}

std::vector<std::string> collectCudaKernelSigningPublicKeys(const json& j) {
    std::set<std::string> keys;
    collectCudaKernelSigningPublicKeysRecursive(j, keys);
    return std::vector<std::string>(keys.begin(), keys.end());
}

std::vector<CudaKernelSourceInspection> collectCudaKernelSourceInfo(const json& j) {
    std::vector<CudaKernelSourceInspection> out;
    collectCudaKernelSourceInfoRecursive(j, out);
    return out;
}

json collectCudaKernelSourceInfoJson(const json& j) { return cudaKernelSourceInspectionListToJson(collectCudaKernelSourceInfo(j)); }

std::string cudaKernelLoadedModelSafetyDisclaimer() {
    return "CudaKernelExpression CUDA source is unsafe, trusted code execution. Ed25519 signature verification only proves that the "
           "encrypted CUDA source, ABI, launch policy, and expression graph match what was signed; AES-256-GCM decryption only restores "
           "the source for inspection/compilation by a holder of the out-of-band decryption key. Neither mechanism makes the CUDA code "
           "safe, sandboxed, bounds-checked, or warranted. The serialized model records only key fingerprints, never signing public keys "
           "or source decryption keys. Inspect all CUDA source that will be compiled before enabling loaded CUDA kernel compilation.";
}

}  // namespace ThorImplementation
