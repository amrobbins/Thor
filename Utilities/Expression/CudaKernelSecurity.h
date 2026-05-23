#pragma once

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

namespace ThorImplementation {

struct CudaKernelSignatureVerificationResult {
    bool verified = false;
    std::string message;
};

[[nodiscard]] nlohmann::json cudaKernelManifestFromExpressionJson(const nlohmann::json& expression_json);
[[nodiscard]] std::string cudaKernelManifestCanonicalBytes(const nlohmann::json& manifest);
[[nodiscard]] std::string cudaKernelGenerateAndAttachManifestSignature(nlohmann::json& expression_json);
[[nodiscard]] CudaKernelSignatureVerificationResult cudaKernelVerifyManifestSignature(
    const nlohmann::json& expression_json,
    const std::string& trusted_public_key);
[[nodiscard]] std::vector<std::string> collectCudaKernelSigningPublicKeys(const nlohmann::json& j);
[[nodiscard]] nlohmann::json collectCudaKernelSourceInfoJson(const nlohmann::json& j);
[[nodiscard]] std::string cudaKernelLoadedModelSafetyDisclaimer();

}  // namespace ThorImplementation
