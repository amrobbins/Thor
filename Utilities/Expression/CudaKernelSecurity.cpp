#include "Utilities/Expression/CudaKernelSecurity.h"

#include <openssl/crypto.h>
#include <openssl/evp.h>
#include <openssl/rand.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using json = nlohmann::json;

namespace ThorImplementation {
namespace {

using EvpPkeyPtr = std::unique_ptr<EVP_PKEY, decltype(&EVP_PKEY_free)>;
using EvpPkeyCtxPtr = std::unique_ptr<EVP_PKEY_CTX, decltype(&EVP_PKEY_CTX_free)>;
using EvpMdCtxPtr = std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)>;

std::string opensslErrorPrefix(const std::string& what) { return what + " failed while handling CudaKernelExpression Ed25519 signatures."; }

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

std::string normalizeEd25519PublicKey(const std::string& public_key) {
    std::vector<unsigned char> raw = hexDecodeRaw(public_key);
    if (raw.size() != 32) {
        throw std::runtime_error(
            "CudaKernelExpression trusted Ed25519 public key must be 32 raw bytes encoded as hex, optionally prefixed with 'ed25519:'.");
    }
    return "ed25519:" + hexEncode(raw.data(), raw.size());
}

std::mutex& signingPublicKeyRegistryMutex() {
    static std::mutex mutex;
    return mutex;
}

std::unordered_map<std::string, std::string>& signingPublicKeyRegistry() {
    static std::unordered_map<std::string, std::string> registry;
    return registry;
}

std::string publicKeyFingerprintFromRaw(const std::vector<unsigned char>& public_key) { return sha256Hex(public_key); }

bool publicKeyFingerprintContainsPublicKeyMaterial(const std::string& public_key_fingerprint,
                                                   const std::vector<unsigned char>& public_key) {
    try {
        return hexDecodeRaw(public_key_fingerprint) == public_key;
    } catch (const std::exception&) {
        return false;
    }
}

void requirePublicKeyFingerprintDoesNotContainPublicKeyMaterial(const std::string& public_key_fingerprint,
                                                                const std::vector<unsigned char>& public_key,
                                                                const char* error_message) {
    if (publicKeyFingerprintContainsPublicKeyMaterial(public_key_fingerprint, public_key)) {
        throw std::logic_error(error_message);
    }
}

std::string publicKeyFingerprintFromText(const std::string& public_key) {
    return publicKeyFingerprintFromRaw(hexDecodeRaw(normalizeEd25519PublicKey(public_key)));
}

void rememberEphemeralSigningPublicKey(const std::string& public_key_text) {
    const std::string fingerprint = publicKeyFingerprintFromText(public_key_text);
    std::lock_guard<std::mutex> lock(signingPublicKeyRegistryMutex());
    signingPublicKeyRegistry()[fingerprint] = public_key_text;
}

std::string lookupEphemeralSigningPublicKeyByFingerprint(const std::string& fingerprint) {
    std::lock_guard<std::mutex> lock(signingPublicKeyRegistryMutex());
    auto it = signingPublicKeyRegistry().find(fingerprint);
    if (it == signingPublicKeyRegistry().end()) {
        return {};
    }
    return it->second;
}

json canonicalKernelJson(const json& kernel_json) {
    json kernel;
    kernel["schema_version"] = kernel_json.at("schema_version");
    kernel["name"] = kernel_json.at("name");
    kernel["source"] = kernel_json.at("source");
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
                info.source = kernel.value("source", std::string{});
                info.compiled_source_hash = kernel.value("compiled_source_hash", std::string{});
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
               {"loaded_source_compilation_allowed", info.loaded_source_compilation_allowed}};
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

std::string cudaKernelGenerateAndAttachManifestSignature(json& expression_json) {
    if (!expression_json.contains("cuda_kernels") || expression_json.at("cuda_kernels").empty()) {
        return {};
    }

    // Signing keys are intentionally generated internally and kept ephemeral. The private key is
    // never accepted from users, serialized, returned, or stored outside this stack frame.
    if (RAND_status() != 1) {
        throw std::runtime_error("OpenSSL CSPRNG is not seeded; refusing to generate CudaKernelExpression signing key.");
    }

    EvpPkeyCtxPtr keygen_ctx(EVP_PKEY_CTX_new_id(EVP_PKEY_ED25519, nullptr), EVP_PKEY_CTX_free);
    if (!keygen_ctx || EVP_PKEY_keygen_init(keygen_ctx.get()) <= 0) {
        throw std::runtime_error(opensslErrorPrefix("EVP_PKEY_keygen_init"));
    }

    EVP_PKEY* raw_key = nullptr;
    if (EVP_PKEY_keygen(keygen_ctx.get(), &raw_key) <= 0 || raw_key == nullptr) {
        throw std::runtime_error(opensslErrorPrefix("EVP_PKEY_keygen"));
    }
    EvpPkeyPtr key(raw_key, EVP_PKEY_free);

    std::vector<unsigned char> public_key(32);
    size_t public_key_len = public_key.size();
    if (EVP_PKEY_get_raw_public_key(key.get(), public_key.data(), &public_key_len) <= 0 || public_key_len != public_key.size()) {
        throw std::runtime_error(opensslErrorPrefix("EVP_PKEY_get_raw_public_key"));
    }

    const json manifest = cudaKernelManifestFromExpressionJson(expression_json);
    const std::string manifest_bytes = cudaKernelManifestCanonicalBytes(manifest);

    EvpMdCtxPtr md_ctx(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    if (!md_ctx || EVP_DigestSignInit(md_ctx.get(), nullptr, nullptr, nullptr, key.get()) <= 0) {
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

    const std::string public_key_text = "ed25519:" + hexEncode(public_key.data(), public_key.size());
    const std::string public_key_fingerprint = publicKeyFingerprintFromRaw(public_key);
    requirePublicKeyFingerprintDoesNotContainPublicKeyMaterial(
        public_key_fingerprint,
        public_key,
        "Internal CudaKernelExpression signing error: public_key_fingerprint contains public key material instead of a SHA-256 digest.");
    rememberEphemeralSigningPublicKey(public_key_text);

    expression_json["cuda_kernel_manifest_signature"] = json{{"schema_version", 1},
                                                             {"algorithm", "ed25519"},
                                                             {"public_key_fingerprint", public_key_fingerprint},
                                                             {"manifest_sha256", sha256Hex(manifest_bytes)},
                                                             {"signature", "ed25519:" + hexEncode(signature.data(), signature.size())},
                                                             {"disclaimer", cudaKernelLoadedModelSafetyDisclaimer()}};
    return public_key_text;
}

CudaKernelSignatureVerificationResult cudaKernelVerifyManifestSignature(const json& expression_json,
                                                                        const std::string& trusted_public_key) {
    if (!expression_json.contains("cuda_kernels") || expression_json.at("cuda_kernels").empty()) {
        return {true, "Expression contains no CudaKernelExpression CUDA source."};
    }
    if (trusted_public_key.empty()) {
        return {false,
                "A trusted Ed25519 public key is required to compile CudaKernelExpression CUDA source loaded from a saved model. Load "
                "without opt-in to inspect the source, then provide the out-of-band public key printed when the model was saved through "
                "the load API only after applying your own security policy."};
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
                "CudaKernelExpression CUDA manifest signature verification failed. The CUDA source, ABI, launch policy, or expression "
                "graph may have been modified, or the wrong public key was supplied."};
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
    return "CudaKernelExpression CUDA source is unsafe, trusted code execution. Ed25519 signature verification only proves that the CUDA "
           "source, ABI, launch policy, and expression graph match what was signed; it does not make the CUDA code safe, sandboxed, "
           "bounds-checked, or warranted. The serialized model records only a public-key fingerprint, not the public key itself. Inspect "
           "all CUDA source that will be compiled before providing the trusted public key and enabling loaded CUDA kernel compilation.";
}

}  // namespace ThorImplementation
