#include "Utilities/Common/CudnnExecutionWorkspace.h"

#include <stdexcept>
#include <string>

using namespace ThorImplementation;
using namespace std;

namespace {

string workspacePrefix(string_view operationName) {
    string prefix = "cuDNN";
    if (!operationName.empty()) {
        prefix += " ";
        prefix += operationName;
    }
    prefix += " workspace";
    return prefix;
}

}  // namespace

uint64_t ThorImplementation::checkedCudnnWorkspaceSizeInBytes(int64_t reportedBytes, string_view operationName) {
    if (reportedBytes < 0) {
        throw runtime_error(workspacePrefix(operationName) + " size reported by cuDNN is negative: " + to_string(reportedBytes) + ".");
    }
    return static_cast<uint64_t>(reportedBytes);
}

void ThorImplementation::validateCudnnExecutionWorkspace(const optional<Tensor>& workspace,
                                                         uint64_t requiredBytes,
                                                         int gpuNum,
                                                         string_view operationName) {
    const string prefix = workspacePrefix(operationName);

    if (!workspace.has_value()) {
        if (requiredBytes != 0) {
            throw invalid_argument(prefix + " requires at least " + to_string(requiredBytes) + " bytes, but no workspace was provided.");
        }
        return;
    }

    const Tensor& tensor = workspace.value();
    if (!tensor.isInitialized()) {
        throw invalid_argument(prefix + " tensor is not initialized.");
    }
    if (tensor.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
        throw invalid_argument(prefix + " must be a GPU tensor.");
    }
    if (tensor.getPlacement().getDeviceNum() != gpuNum) {
        throw invalid_argument(prefix + " is on GPU " + to_string(tensor.getPlacement().getDeviceNum()) +
                               ", expected GPU " + to_string(gpuNum) + ".");
    }
    if (tensor.getDataType() != DataType::UINT8) {
        throw invalid_argument(prefix + " must have UINT8 dtype.");
    }
    if (tensor.getArraySizeInBytes() < requiredBytes) {
        throw invalid_argument(prefix + " is too small. Required at least " + to_string(requiredBytes) + " bytes, got " +
                               to_string(tensor.getArraySizeInBytes()) + ".");
    }
}

void* ThorImplementation::cudnnExecutionWorkspacePointer(optional<Tensor>& workspace,
                                                          uint64_t requiredBytes,
                                                          int gpuNum,
                                                          string_view operationName) {
    validateCudnnExecutionWorkspace(workspace, requiredBytes, gpuNum, operationName);
    if (requiredBytes == 0)
        return nullptr;
    return workspace.value().getMemPtr<void>();
}
