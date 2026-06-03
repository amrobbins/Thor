#include "Utilities/Expression/CudaKernelExpression.h"

#include <cuda_runtime.h>
#include <nvrtc.h>

#include <algorithm>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Expression/CudaHelpers.h"
#include "Utilities/Expression/EquationCompiler.h"
#include "Utilities/Expression/FusedEquation.h"

using json = nlohmann::json;

namespace ThorImplementation {
namespace {

std::string dtypeName(DataType dtype) { return TensorDescriptor::getElementTypeName(dtype); }

std::string inputKindName(CudaKernelExpression::TensorParamSpec::Kind kind) {
    switch (kind) {
        case CudaKernelExpression::TensorParamSpec::Kind::Tensor:
            return "tensor";
        case CudaKernelExpression::TensorParamSpec::Kind::TensorRuntimeScalar:
            return "tensor_runtime_scalar";
        case CudaKernelExpression::TensorParamSpec::Kind::HostRuntimeScalar:
            return "host_runtime_scalar";
        default:
            return "unknown";
    }
}

void validateHostRuntimeScalarDataType(DataType type) {
    if (type != DataType::FP32) {
        throw std::invalid_argument("CudaKernelExpression host runtime scalar dtype '" + dtypeName(type) +
                                    "' is not supported. Host runtime scalars are currently bound as fp32 values.");
    }
}

void validateScalarDataType(DataType type) {
    switch (type) {
        case DataType::INT32:
        case DataType::UINT32:
        case DataType::INT64:
        case DataType::UINT64:
        case DataType::FP32:
        case DataType::FP64:
            return;
        default:
            throw std::invalid_argument("CudaKernelExpression scalar dtype '" + dtypeName(type) +
                                        "' is not supported for by-value kernel scalar arguments. Supported scalar dtypes are int32, "
                                        "uint32, int64, uint64, fp32, and fp64.");
    }
}

void validateName(const std::string& name, const char* what) {
    if (name.empty()) {
        throw std::invalid_argument(std::string("CudaKernelExpression ") + what + " name cannot be empty.");
    }
    if (name.rfind("__", 0) == 0) {
        throw std::invalid_argument(std::string("CudaKernelExpression ") + what + " name cannot start with reserved prefix '__': " + name);
    }
}

uint64_t checkedNumel(const std::vector<uint64_t>& dims) {
    uint64_t n = 1;
    for (uint64_t d : dims) {
        if (d != 0 && n > std::numeric_limits<uint64_t>::max() / d) {
            throw std::runtime_error("CudaKernelExpression tensor numel overflow.");
        }
        n *= d;
    }
    return n;
}

std::string stableSourceHash(const std::string& text) {
    constexpr uint64_t kOffset = 1469598103934665603ULL;
    constexpr uint64_t kPrime = 1099511628211ULL;

    uint64_t hash = kOffset;
    for (unsigned char c : text) {
        hash ^= static_cast<uint64_t>(c);
        hash *= kPrime;
    }

    std::ostringstream ss;
    ss << "fnv1a64:" << std::hex << std::setw(16) << std::setfill('0') << hash;
    return ss.str();
}

void ensureCudaContextCurrentForCudaKernelExpression(int device_num) {
    CU_CHECK(cuInit(0));

    CUdevice device;
    CU_CHECK(cuDeviceGet(&device, device_num));

    CUcontext ctx = nullptr;
    CU_CHECK(cuCtxGetCurrent(&ctx));

    if (ctx == nullptr) {
        CUcontext primary;
        CU_CHECK(cuDevicePrimaryCtxRetain(&primary, device));
        CU_CHECK(cuCtxSetCurrent(primary));
        return;
    }

    CUdevice current_device;
    CU_CHECK(cuCtxGetDevice(&current_device));
    if (static_cast<int>(current_device) != device_num) {
        CUcontext primary;
        CU_CHECK(cuDevicePrimaryCtxRetain(&primary, device));
        CU_CHECK(cuCtxSetCurrent(primary));
    }
}

EquationSignature buildSignature(uint32_t num_params, int device_num) {
    cudaDeviceProp prop{};
    cudaError_t cuda_status = cudaGetDeviceProperties(&prop, device_num);
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error(std::string("cudaGetDeviceProperties failed: ") + cudaGetErrorString(cuda_status));
    }

    EquationSignature sig{};
    sig.num_inputs = num_params;
    sig.sm_major = prop.major;
    sig.sm_minor = prop.minor;
    sig.device_num = device_num;
    sig.use_fast_math = false;
    return sig;
}

std::string customKernelCompiledSource(const std::string& user_source) {
    // CudaKernelExpression is intended to feel like writing a normal CUDA kernel.
    // NVRTC does not implicitly provide the fixed-width integer typedefs, and in
    // this runtime compilation path it may not have a host standard include path
    // that can satisfy <stdint.h>. Provide the small ABI prelude directly instead
    // of requiring every custom kernel to carry boilerplate includes.
    //
    // Keep this intentionally minimal: users still own CUDA-specific includes
    // they need for half/bfloat16/etc. Reset the line mapping before the user
    // source so NVRTC diagnostics point at the kernel they wrote.
    static constexpr const char* prelude = R"cuda(
#ifndef THOR_CUDA_KERNEL_EXPRESSION_FIXED_WIDTH_TYPES
#define THOR_CUDA_KERNEL_EXPRESSION_FIXED_WIDTH_TYPES
typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef signed short int16_t;
typedef unsigned short uint16_t;
typedef signed int int32_t;
typedef unsigned int uint32_t;
typedef signed long long int64_t;
typedef unsigned long long uint64_t;
#endif
#line 1 "fused.cu"
)cuda";
    return std::string(prelude) + user_source;
}

std::string customKernelCacheKey(const std::string& name,
                                 const std::string& compiled_source,
                                 const std::string& entry,
                                 const EquationSignature& sig) {
    std::ostringstream oss;
    oss << "cuda_kernel_expression:v4\n";
    oss << "name=" << name << "\n";
    oss << "entry=" << entry << "\n";
    oss << "sm=" << sig.sm_major << sig.sm_minor << "\n";
    oss << "source_hash=" << stableSourceHash(compiled_source) << "\n";
    oss << compiled_source;
    return oss.str();
}

std::mutex compiled_kernel_cache_mutex;
std::unordered_map<std::string, std::weak_ptr<CompiledCudaKernel>> compiled_kernel_cache;

std::shared_ptr<CompiledCudaKernel> cacheLookup(const std::string& key) {
    std::lock_guard<std::mutex> lock(compiled_kernel_cache_mutex);
    auto it = compiled_kernel_cache.find(key);
    if (it == compiled_kernel_cache.end()) {
        return nullptr;
    }
    std::shared_ptr<CompiledCudaKernel> hit = it->second.lock();
    if (!hit) {
        compiled_kernel_cache.erase(it);
    }
    return hit;
}

void cacheInsert(const std::string& key, const std::shared_ptr<CompiledCudaKernel>& compiled) {
    std::lock_guard<std::mutex> lock(compiled_kernel_cache_mutex);
    compiled_kernel_cache[key] = compiled;
}

void validateDenseContiguous(const Tensor& tensor, const std::string& name, const char* what) {
    if (!tensor.isInitialized()) {
        throw std::invalid_argument(std::string("CudaKernelExpression ") + what + " tensor '" + name + "' is not initialized.");
    }
    if (tensor.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
        throw std::invalid_argument(std::string("CudaKernelExpression ") + what + " tensor '" + name + "' is not on GPU.");
    }
    if (!tensor.isDenseContiguous()) {
        throw std::invalid_argument(std::string("CudaKernelExpression ") + what + " tensor '" + name +
                                    "' is not dense-contiguous. CudaKernelExpression raw pointer ABI requires contiguous tensors.");
    }
}

const Tensor& lookupTensor(const std::unordered_map<std::string, Tensor>& tensors, const std::string& name, const char* where) {
    auto it = tensors.find(name);
    if (it == tensors.end()) {
        throw std::runtime_error(std::string("CudaKernelExpression missing tensor '") + name + "' while resolving " + where + ".");
    }
    return it->second;
}

std::string joinNames(const std::vector<std::string>& names) {
    std::ostringstream oss;
    for (size_t i = 0; i < names.size(); ++i) {
        if (i)
            oss << ", ";
        oss << names[i];
    }
    return oss.str();
}

const char* tensorParamKindName(CudaKernelExpression::TensorParamSpec::Kind kind) {
    switch (kind) {
        case CudaKernelExpression::TensorParamSpec::Kind::Tensor:
            return "tensor";
        case CudaKernelExpression::TensorParamSpec::Kind::TensorRuntimeScalar:
            return "tensor_runtime_scalar";
        case CudaKernelExpression::TensorParamSpec::Kind::HostRuntimeScalar:
            return "host_runtime_scalar";
    }
    throw std::runtime_error("Unknown CudaKernelExpression tensor param kind.");
}

CudaKernelExpression::TensorParamSpec::Kind tensorParamKindFromName(const std::string& kind) {
    if (kind == "tensor")
        return CudaKernelExpression::TensorParamSpec::Kind::Tensor;
    if (kind == "tensor_runtime_scalar")
        return CudaKernelExpression::TensorParamSpec::Kind::TensorRuntimeScalar;
    if (kind == "host_runtime_scalar")
        return CudaKernelExpression::TensorParamSpec::Kind::HostRuntimeScalar;
    throw std::runtime_error("Unknown CudaKernelExpression tensor param kind in serialized expression: " + kind);
}

const char* dimExprKindName(CudaKernelExpression::DimExpr::Kind kind) {
    switch (kind) {
        case CudaKernelExpression::DimExpr::Kind::Constant:
            return "constant";
        case CudaKernelExpression::DimExpr::Kind::TensorDim:
            return "tensor_dim";
        case CudaKernelExpression::DimExpr::Kind::TensorNumel:
            return "tensor_numel";
    }
    throw std::runtime_error("Unknown CudaKernelExpression DimExpr kind.");
}

const char* launchSpecKindName(CudaKernelExpression::LaunchSpec::Kind kind) {
    switch (kind) {
        case CudaKernelExpression::LaunchSpec::Kind::Grid1D:
            return "grid_1d";
    }
    throw std::runtime_error("Unknown CudaKernelExpression launch spec kind.");
}

}  // namespace

uint64_t CudaKernelExpression::DimExpr::resolve(const std::unordered_map<std::string, Tensor>& tensors) const {
    switch (kind_) {
        case Kind::Constant:
            return value_;
        case Kind::TensorDim: {
            const Tensor& tensor = lookupTensor(tensors, tensor_name_, "dimension expression");
            const auto dims = tensor.getDimensions();
            if (axis_ >= dims.size()) {
                throw std::runtime_error("CudaKernelExpression dimension expression axis " + std::to_string(axis_) +
                                         " is out of range for tensor '" + tensor_name_ + "' with rank " + std::to_string(dims.size()) +
                                         ".");
            }
            return dims[axis_];
        }
        case Kind::TensorNumel: {
            const Tensor& tensor = lookupTensor(tensors, tensor_name_, "numel expression");
            return checkedNumel(tensor.getDimensions());
        }
    }
    throw std::runtime_error("CudaKernelExpression encountered unknown dimension expression kind.");
}

uint64_t CudaKernelExpression::DimExpr::resolve(const std::unordered_map<std::string, std::vector<uint64_t>>& tensor_shapes) const {
    switch (kind_) {
        case Kind::Constant:
            return value_;
        case Kind::TensorDim: {
            auto it = tensor_shapes.find(tensor_name_);
            if (it == tensor_shapes.end()) {
                throw std::runtime_error("CudaKernelExpression dimension expression references unknown tensor shape '" + tensor_name_ +
                                         "'.");
            }
            if (axis_ >= it->second.size()) {
                throw std::runtime_error("CudaKernelExpression dimension expression axis " + std::to_string(axis_) +
                                         " is out of range for tensor shape '" + tensor_name_ + "'.");
            }
            return it->second[axis_];
        }
        case Kind::TensorNumel: {
            auto it = tensor_shapes.find(tensor_name_);
            if (it == tensor_shapes.end()) {
                throw std::runtime_error("CudaKernelExpression numel expression references unknown tensor shape '" + tensor_name_ + "'.");
            }
            return checkedNumel(it->second);
        }
    }
    throw std::runtime_error("CudaKernelExpression encountered unknown dimension expression kind.");
}

std::string CudaKernelExpression::DimExpr::describe() const {
    switch (kind_) {
        case Kind::Constant:
            return std::to_string(value_);
        case Kind::TensorDim:
            return "dim(" + tensor_name_ + ", " + std::to_string(axis_) + ")";
        case Kind::TensorNumel:
            return "numel(" + tensor_name_ + ")";
    }
    return "<unknown>";
}

json CudaKernelExpression::DimExpr::architectureJson() const {
    json j;
    j["kind"] = dimExprKindName(kind_);
    j["tensor_name"] = tensor_name_;
    j["axis"] = axis_;
    j["value"] = value_;
    return j;
}

CudaKernelExpression::DimExpr CudaKernelExpression::DimExpr::deserialize(const json& j) {
    const std::string kind = j.at("kind").get<std::string>();
    if (kind == "constant") {
        return constant(j.at("value").get<uint64_t>());
    }
    if (kind == "tensor_dim") {
        return dim(j.at("tensor_name").get<std::string>(), j.at("axis").get<uint32_t>());
    }
    if (kind == "tensor_numel") {
        return numel(j.at("tensor_name").get<std::string>());
    }
    throw std::runtime_error("Unknown CudaKernelExpression DimExpr kind in serialized expression: " + kind);
}

CudaKernelExpression::LaunchSpec CudaKernelExpression::LaunchSpec::grid1D(DimExpr elements,
                                                                          uint32_t block_size,
                                                                          uint32_t dynamic_shared_bytes) {
    if (block_size == 0) {
        throw std::invalid_argument("CudaKernelExpression grid_1d launch block_size must be nonzero.");
    }
    return LaunchSpec{Kind::Grid1D, std::move(elements), block_size, dynamic_shared_bytes};
}

CudaKernelLaunchConfig CudaKernelExpression::LaunchSpec::resolve(const LaunchContext& launch_context) const {
    switch (kind) {
        case Kind::Grid1D: {
            if (block_size == 0) {
                throw std::runtime_error("CudaKernelExpression grid_1d launch block_size must be nonzero.");
            }
            std::unordered_map<std::string, Tensor> tensors = launch_context.inputs;
            for (const auto& [name, tensor] : launch_context.outputs) {
                tensors.emplace(name, tensor);
            }
            uint64_t n = elements.resolve(tensors);
            uint64_t grid_x = (n + block_size - 1) / block_size;
            if (grid_x == 0) {
                grid_x = 1;
            }
            if (grid_x > std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error("CudaKernelExpression grid_1d launch grid.x exceeds uint32_t range.");
            }
            return CudaKernelLaunchConfig{dim3(static_cast<uint32_t>(grid_x), 1, 1), dim3(block_size, 1, 1), dynamic_shared_bytes};
        }
    }
    throw std::runtime_error("Unknown CudaKernelExpression launch spec kind.");
}

json CudaKernelExpression::LaunchSpec::architectureJson() const {
    json j;
    j["kind"] = launchSpecKindName(kind);
    j["elements"] = elements.architectureJson();
    j["block_size"] = block_size;
    j["dynamic_shared_bytes"] = dynamic_shared_bytes;
    return j;
}

CudaKernelExpression::LaunchSpec CudaKernelExpression::LaunchSpec::deserialize(const json& j) {
    const std::string kind_name = j.at("kind").get<std::string>();
    if (kind_name == "grid_1d") {
        return grid1D(
            DimExpr::deserialize(j.at("elements")), j.value("block_size", uint32_t{256}), j.value("dynamic_shared_bytes", uint32_t{0}));
    }
    throw std::runtime_error("Unknown CudaKernelExpression launch spec kind in serialized expression: " + kind_name);
}

const Tensor& CudaKernelExpression::LaunchContext::input(const std::string& name) const {
    return lookupTensor(inputs, name, "launch input");
}

const Tensor& CudaKernelExpression::LaunchContext::output(const std::string& name) const {
    return lookupTensor(outputs, name, "launch output");
}

uint64_t CudaKernelExpression::LaunchContext::dim(const std::string& tensor_name, uint32_t axis) const {
    auto input_it = inputs.find(tensor_name);
    const Tensor* tensor = nullptr;
    if (input_it != inputs.end()) {
        tensor = &input_it->second;
    } else {
        auto output_it = outputs.find(tensor_name);
        if (output_it != outputs.end()) {
            tensor = &output_it->second;
        }
    }
    if (!tensor) {
        throw std::runtime_error("CudaKernelExpression launch context has no tensor named '" + tensor_name + "'.");
    }
    const auto dims = tensor->getDimensions();
    if (axis >= dims.size()) {
        throw std::runtime_error("CudaKernelExpression launch context dim axis out of range for tensor '" + tensor_name + "'.");
    }
    return dims[axis];
}

uint64_t CudaKernelExpression::LaunchContext::numel(const std::string& tensor_name) const {
    auto input_it = inputs.find(tensor_name);
    if (input_it != inputs.end()) {
        return checkedNumel(input_it->second.getDimensions());
    }
    auto output_it = outputs.find(tensor_name);
    if (output_it != outputs.end()) {
        return checkedNumel(output_it->second.getDimensions());
    }
    throw std::runtime_error("CudaKernelExpression launch context has no tensor named '" + tensor_name + "'.");
}

DataType CudaKernelExpression::LaunchContext::dtype(const std::string& tensor_name) const {
    auto input_it = inputs.find(tensor_name);
    if (input_it != inputs.end()) {
        return input_it->second.getDataType();
    }
    auto output_it = outputs.find(tensor_name);
    if (output_it != outputs.end()) {
        return output_it->second.getDataType();
    }
    throw std::runtime_error("CudaKernelExpression launch context has no tensor named '" + tensor_name + "'.");
}

CudaKernelExpression::Builder::Builder(std::string name) : name_(std::move(name)) { validateName(name_, "kernel"); }

CudaKernelExpression::Builder& CudaKernelExpression::Builder::source(std::string cuda_source) {
    source_ = std::move(cuda_source);
    return *this;
}

CudaKernelExpression::Builder& CudaKernelExpression::Builder::entry(std::string entrypoint) {
    validateName(entrypoint, "entrypoint");
    entry_ = std::move(entrypoint);
    return *this;
}

CudaKernelExpression::Builder& CudaKernelExpression::Builder::input(std::string name, DataType dtype) {
    validateName(name, "input");
    inputs_.push_back(TensorParamSpec{std::move(name), dtype, TensorParamSpec::Kind::Tensor});
    return *this;
}

CudaKernelExpression::Builder& CudaKernelExpression::Builder::tensorRuntimeScalarInput(std::string name, DataType dtype) {
    validateName(name, "tensor runtime scalar input");
    inputs_.push_back(TensorParamSpec{std::move(name), dtype, TensorParamSpec::Kind::TensorRuntimeScalar});
    return *this;
}

CudaKernelExpression::Builder& CudaKernelExpression::Builder::hostRuntimeScalarInput(std::string name, DataType dtype) {
    validateName(name, "host runtime scalar input");
    validateHostRuntimeScalarDataType(dtype);
    inputs_.push_back(TensorParamSpec{std::move(name), dtype, TensorParamSpec::Kind::HostRuntimeScalar});
    return *this;
}

CudaKernelExpression::Builder& CudaKernelExpression::Builder::output(std::string name, DataType dtype, std::vector<DimExpr> shape) {
    validateName(name, "output");
    if (shape.empty()) {
        throw std::invalid_argument("CudaKernelExpression output '" + name +
                                    "' requires a non-empty shape. Use scalar outputs as shape {1}.");
    }
    outputs_.push_back(OutputParamSpec{std::move(name), dtype, std::move(shape), {}});
    return *this;
}

CudaKernelExpression::Builder& CudaKernelExpression::Builder::outputLike(std::string name, DataType dtype, const std::string& input_name) {
    validateName(name, "output");
    validateName(input_name, "input");
    outputs_.push_back(OutputParamSpec{std::move(name), dtype, {}, input_name});
    return *this;
}

CudaKernelExpression::Builder& CudaKernelExpression::Builder::scalar(
    std::string name, DataType type, std::variant<int32_t, uint32_t, int64_t, uint64_t, float, double, DimExpr> value) {
    validateName(name, "scalar");
    validateScalarDataType(type);
    scalars_.push_back(ScalarParamSpec{std::move(name), type, std::move(value)});
    return *this;
}

CudaKernelExpression::Builder& CudaKernelExpression::Builder::launch(LaunchFn launch_fn) {
    launch_fn_ = std::move(launch_fn);
    launch_spec_ = std::nullopt;
    return *this;
}

CudaKernelExpression::Builder& CudaKernelExpression::Builder::launchGrid1D(DimExpr elements,
                                                                           uint32_t block_size,
                                                                           uint32_t dynamic_shared_bytes) {
    launch_spec_ = LaunchSpec::grid1D(std::move(elements), block_size, dynamic_shared_bytes);
    launch_fn_ = nullptr;
    return *this;
}

CudaKernelExpression CudaKernelExpression::Builder::build() const {
    if (source_.empty()) {
        throw std::invalid_argument("CudaKernelExpression requires CUDA source.");
    }
    if (entry_.empty()) {
        throw std::invalid_argument("CudaKernelExpression requires an entrypoint.");
    }
    if (inputs_.empty()) {
        throw std::invalid_argument("CudaKernelExpression requires at least one input.");
    }
    if (outputs_.empty()) {
        throw std::invalid_argument("CudaKernelExpression requires at least one output tensor.");
    }
    if (!launch_fn_ && !launch_spec_.has_value()) {
        throw std::invalid_argument("CudaKernelExpression requires a launch function or serializable launch spec.");
    }

    std::unordered_set<std::string> names;
    std::unordered_set<std::string> tensor_input_names;
    auto insert = [&](const std::string& name, const char* kind) {
        if (!names.insert(name).second) {
            throw std::invalid_argument(std::string("CudaKernelExpression duplicate parameter name for ") + kind + ": " + name);
        }
    };
    for (const auto& input : inputs_) {
        insert(input.name, "input");
        if (input.kind == TensorParamSpec::Kind::Tensor) {
            tensor_input_names.insert(input.name);
        }
    }
    for (const auto& output : outputs_) {
        insert(output.name, "output");
        if (!output.like_input_name.empty() && !tensor_input_names.contains(output.like_input_name)) {
            throw std::invalid_argument("CudaKernelExpression output '" + output.name +
                                        "' must be declared like a tensor input, got unknown/non-tensor input '" + output.like_input_name +
                                        "'.");
        }
    }
    for (const auto& scalar : scalars_)
        insert(scalar.name, "scalar");

    return CudaKernelExpression(name_, source_, entry_, inputs_, outputs_, scalars_, launch_fn_, launch_spec_, true);
}

CudaKernelExpression::CudaKernelExpression(std::string name,
                                           std::string source,
                                           std::string entry,
                                           std::vector<TensorParamSpec> inputs,
                                           std::vector<OutputParamSpec> outputs,
                                           std::vector<ScalarParamSpec> scalars,
                                           LaunchFn launch_fn,
                                           std::optional<LaunchSpec> launch_spec,
                                           bool loaded_source_compilation_allowed)
    : name_(std::move(name)),
      source_(std::move(source)),
      entry_(std::move(entry)),
      inputs_(std::move(inputs)),
      outputs_(std::move(outputs)),
      scalars_(std::move(scalars)),
      launch_fn_(std::move(launch_fn)),
      launch_spec_(std::move(launch_spec)),
      loaded_source_compilation_allowed_(loaded_source_compilation_allowed) {}

std::string CudaKernelExpression::cacheSignature() const {
    std::ostringstream oss;
    oss << "name=" << name_ << "\n";
    oss << "entry=" << entry_ << "\n";
    oss << "source_hash=" << stableSourceHash(source_) << "\n";
    oss << "inputs=";
    for (const TensorParamSpec& input : inputs_) {
        oss << input.name << ":" << inputKindName(input.kind) << ":" << dtypeName(input.dtype) << ";";
    }
    oss << "\noutputs=";
    for (const OutputParamSpec& output : outputs_) {
        oss << output.name << ":" << dtypeName(output.dtype) << ":";
        if (!output.like_input_name.empty()) {
            oss << "like(" << output.like_input_name << ")";
        } else {
            oss << "shape(";
            for (const DimExpr& dim : output.shape) {
                oss << dim.describe() << ",";
            }
            oss << ")";
        }
        oss << ";";
    }
    oss << "\nscalars=";
    for (const ScalarParamSpec& scalar : scalars_) {
        oss << scalar.name << ":" << static_cast<int>(scalar.type) << ";";
    }
    if (launch_spec_.has_value()) {
        oss << "\nlaunch=" << launch_spec_->architectureJson().dump();
    } else {
        oss << "\nlaunch=<runtime_callback>";
    }
    return oss.str();
}

std::string cudaKernelExpressionCompiledSourceForInspection(const std::string& user_source) {
    return customKernelCompiledSource(user_source);
}

std::string CudaKernelExpression::compiledSource() const { return customKernelCompiledSource(source_); }

CudaKernelExpression::SourceInfo CudaKernelExpression::sourceInfo() const {
    const std::string compiled = compiledSource();
    return SourceInfo{
        .name = name_,
        .entrypoint = entry_,
        .source = source_,
        .compiled_source = compiled,
        .source_hash = stableSourceHash(compiled),
        .loaded_source_compilation_allowed = loaded_source_compilation_allowed_,
    };
}

json CudaKernelExpression::architectureJson() const {
    if (!launch_spec_.has_value()) {
        throw std::runtime_error("CudaKernelExpression '" + name_ +
                                 "' was built with a non-serializable launch callback. Use launchGrid1D(...) for CudaKernelExpressions "
                                 "that may be saved/loaded.");
    }

    json j;
    j["schema_version"] = 1;
    j["name"] = name_;
    j["source"] = source_;
    j["entry"] = entry_;
    j["compiled_source_hash"] = stableSourceHash(compiledSource());
    j["loaded_source_compilation_allowed"] = loaded_source_compilation_allowed_;

    j["inputs"] = json::array();
    for (const TensorParamSpec& input : inputs_) {
        j["inputs"].push_back(json{{"name", input.name}, {"dtype", input.dtype}, {"kind", tensorParamKindName(input.kind)}});
    }

    j["outputs"] = json::array();
    for (const OutputParamSpec& output : outputs_) {
        json output_json{{"name", output.name}, {"dtype", output.dtype}};
        if (!output.like_input_name.empty()) {
            output_json["like_input_name"] = output.like_input_name;
        } else {
            output_json["shape"] = json::array();
            for (const DimExpr& dim : output.shape) {
                output_json["shape"].push_back(dim.architectureJson());
            }
        }
        j["outputs"].push_back(std::move(output_json));
    }

    auto scalar_value_to_json = [](const ScalarParamSpec& scalar) {
        return std::visit(
            [](const auto& v) -> json {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, DimExpr>) {
                    return json{{"kind", "dim_expr"}, {"value", v.architectureJson()}};
                } else {
                    return json{{"kind", "literal"}, {"value", v}};
                }
            },
            scalar.value);
    };

    j["scalars"] = json::array();
    for (const ScalarParamSpec& scalar : scalars_) {
        j["scalars"].push_back(json{{"name", scalar.name}, {"dtype", scalar.type}, {"value", scalar_value_to_json(scalar)}});
    }

    j["launch"] = launch_spec_->architectureJson();
    return j;
}

CudaKernelExpression CudaKernelExpression::deserialize(const json& j, bool allow_unsafe_loaded_cuda_source) {
    const int schema_version = j.value("schema_version", 1);
    if (schema_version != 1) {
        throw std::runtime_error("Unsupported CudaKernelExpression schema_version: " + std::to_string(schema_version));
    }

    Builder builder(j.at("name").get<std::string>());
    if (!j.contains("source")) {
        throw std::runtime_error(
            "CudaKernelExpression serialized kernel does not contain plaintext CUDA source. Encrypted CUDA source must be verified and "
            "decrypted by ExpressionDefinition::deserialize before individual kernels are deserialized.");
    }
    builder.source(j.at("source").get<std::string>());
    builder.entry(j.at("entry").get<std::string>());

    for (const json& input_json : j.at("inputs")) {
        const std::string name = input_json.at("name").get<std::string>();
        const DataType dtype = input_json.at("dtype").get<DataType>();
        const TensorParamSpec::Kind kind = tensorParamKindFromName(input_json.value("kind", std::string("tensor")));
        switch (kind) {
            case TensorParamSpec::Kind::Tensor:
                builder.input(name, dtype);
                break;
            case TensorParamSpec::Kind::TensorRuntimeScalar:
                builder.tensorRuntimeScalarInput(name, dtype);
                break;
            case TensorParamSpec::Kind::HostRuntimeScalar:
                builder.hostRuntimeScalarInput(name, dtype);
                break;
        }
    }

    for (const json& output_json : j.at("outputs")) {
        const std::string name = output_json.at("name").get<std::string>();
        const DataType dtype = output_json.at("dtype").get<DataType>();
        if (output_json.contains("like_input_name")) {
            builder.outputLike(name, dtype, output_json.at("like_input_name").get<std::string>());
        } else {
            std::vector<DimExpr> shape;
            for (const json& dim_json : output_json.at("shape")) {
                shape.push_back(DimExpr::deserialize(dim_json));
            }
            builder.output(name, dtype, std::move(shape));
        }
    }

    auto parse_scalar_value = [](DataType dtype,
                                 const json& value_json) -> std::variant<int32_t, uint32_t, int64_t, uint64_t, float, double, DimExpr> {
        const std::string kind = value_json.at("kind").get<std::string>();
        if (kind == "dim_expr") {
            return DimExpr::deserialize(value_json.at("value"));
        }
        if (kind != "literal") {
            throw std::runtime_error("Unknown CudaKernelExpression scalar value kind in serialized expression: " + kind);
        }
        switch (dtype) {
            case DataType::INT32:
                return value_json.at("value").get<int32_t>();
            case DataType::UINT32:
                return value_json.at("value").get<uint32_t>();
            case DataType::INT64:
                return value_json.at("value").get<int64_t>();
            case DataType::UINT64:
                return value_json.at("value").get<uint64_t>();
            case DataType::FP32:
                return value_json.at("value").get<float>();
            case DataType::FP64:
                return value_json.at("value").get<double>();
            default:
                throw std::runtime_error("Unsupported CudaKernelExpression scalar dtype in serialized expression: " + dtypeName(dtype));
        }
    };

    for (const json& scalar_json : j.at("scalars")) {
        const std::string name = scalar_json.at("name").get<std::string>();
        const DataType dtype = scalar_json.at("dtype").get<DataType>();
        builder.scalar(name, dtype, parse_scalar_value(dtype, scalar_json.at("value")));
    }

    LaunchSpec launch_spec = LaunchSpec::deserialize(j.at("launch"));
    builder.launchGrid1D(launch_spec.elements, launch_spec.block_size, launch_spec.dynamic_shared_bytes);

    CudaKernelExpression result = builder.build();
    result.loaded_source_compilation_allowed_ = allow_unsafe_loaded_cuda_source;

    const std::string expected_hash = j.value("compiled_source_hash", std::string{});
    if (!expected_hash.empty() && expected_hash != stableSourceHash(result.compiledSource())) {
        throw std::runtime_error("CudaKernelExpression compiled_source_hash mismatch for loaded kernel '" + result.name_ + "'.");
    }

    return result;
}

std::vector<std::vector<uint64_t>> CudaKernelExpression::inferOutputShapesFromInputShapes(
    const std::unordered_map<std::string, std::vector<uint64_t>>& input_shapes) const {
    std::vector<std::vector<uint64_t>> result;
    result.reserve(outputs_.size());
    std::unordered_map<std::string, std::vector<uint64_t>> shapes = input_shapes;

    for (const OutputParamSpec& output : outputs_) {
        std::vector<uint64_t> dims;
        if (!output.like_input_name.empty()) {
            auto it = shapes.find(output.like_input_name);
            if (it == shapes.end()) {
                throw std::runtime_error("CudaKernelExpression output '" + output.name + "' is like unknown input shape '" +
                                         output.like_input_name + "'.");
            }
            dims = it->second;
        } else {
            dims.reserve(output.shape.size());
            for (const DimExpr& dim : output.shape) {
                dims.push_back(dim.resolve(shapes));
            }
        }
        shapes.emplace(output.name, dims);
        result.push_back(std::move(dims));
    }

    return result;
}

std::shared_ptr<CompiledCudaKernel> CudaKernelExpression::compile(int device_num) const {
    if (!loaded_source_compilation_allowed_) {
        throw std::runtime_error(
            "Refusing to compile CudaKernelExpression '" + name_ +
            "' because its CUDA source was loaded from a serialized model. Custom CUDA source in loaded models is unsafe code execution. "
            "If you will be running it, you should then load it with "
            "allow_unsafe_loaded_cuda_source=true, the trusted Ed25519 public key, and the AES-256-GCM source decryption "
            "key that were provided when the model was saved. Signature verification provides evidence of the identity of the signed CUDA "
            "manifest. You should inspect the resultant CUDA source.");
    }

    EquationSignature sig =
        buildSignature(static_cast<uint32_t>(inputs_.size() + outputs_.size() + scalars_.size()), device_num);
    const std::string compiled_source = customKernelCompiledSource(source_);
    const std::string key = customKernelCacheKey(name_, compiled_source, entry_, sig);

    if (auto hit = cacheLookup(key)) {
        return hit;
    }

    ScopedGpu scoped_gpu(device_num);
    ensureCudaContextCurrentForCudaKernelExpression(device_num);

    std::vector<char> ltoir = EquationCompiler::compileToLtoIr(compiled_source, entry_, sig);
    std::vector<char> cubin = EquationCompiler::linkToCubin(ltoir, sig);

    auto compiled = std::make_shared<CompiledCudaKernel>();
    compiled->cache_key = key;
    compiled->kernel_name = entry_;
    compiled->device_num = device_num;
    CU_CHECK(cuModuleLoadData(&compiled->module, cubin.data()));
    CU_CHECK(cuModuleGetFunction(&compiled->kernel, compiled->module, entry_.c_str()));
    cacheInsert(key, compiled);
    return compiled;
}

std::unordered_map<std::string, Tensor> CudaKernelExpression::allocateAndValidateOutputs(
    const std::unordered_map<std::string, Tensor>& inputs,
    const std::unordered_map<std::string, Tensor>& preallocated_outputs,
    const std::unordered_map<std::string, std::vector<uint64_t>>& requested_output_shapes,
    const TensorPlacement& placement) const {
    std::unordered_map<std::string, Tensor> all_tensors = inputs;
    std::unordered_map<std::string, Tensor> outputs;
    outputs.reserve(outputs_.size());

    for (const OutputParamSpec& spec : outputs_) {
        std::vector<uint64_t> dims;
        if (!spec.like_input_name.empty()) {
            const Tensor& like = lookupTensor(inputs, spec.like_input_name, "output-like shape");
            dims = like.getDimensions();
        } else {
            dims.reserve(spec.shape.size());
            for (const DimExpr& dim : spec.shape) {
                dims.push_back(dim.resolve(all_tensors));
            }
        }

        auto requested_it = requested_output_shapes.find(spec.name);
        if (requested_it != requested_output_shapes.end() && !requested_it->second.empty()) {
            if (requested_it->second != dims) {
                throw std::runtime_error("CudaKernelExpression requested output shape for '" + spec.name + "' does not match spec.");
            }
        }

        auto pre_it = preallocated_outputs.find(spec.name);
        Tensor output;
        if (pre_it != preallocated_outputs.end()) {
            output = pre_it->second;
            validateDenseContiguous(output, spec.name, "output");
            if (output.getDataType() != spec.dtype) {
                throw std::runtime_error("CudaKernelExpression output '" + spec.name + "' dtype mismatch. Got " +
                                         dtypeName(output.getDataType()) + ", expected " + dtypeName(spec.dtype) + ".");
            }
            if (output.getDimensions() != dims) {
                throw std::runtime_error("CudaKernelExpression output '" + spec.name + "' shape mismatch.");
            }
        } else {
            output = Tensor(placement, TensorDescriptor(spec.dtype, dims));
        }

        outputs.emplace(spec.name, output);
        all_tensors.emplace(spec.name, output);
    }

    for (const auto& [name, _] : preallocated_outputs) {
        bool known = std::any_of(outputs_.begin(), outputs_.end(), [&](const OutputParamSpec& spec) { return spec.name == name; });
        if (!known) {
            throw std::runtime_error("CudaKernelExpression received unknown preallocated output: " + name);
        }
    }

    return outputs;
}

CudaKernelScalarValue CudaKernelExpression::resolveScalar(const ScalarParamSpec& scalar,
                                                          const std::unordered_map<std::string, Tensor>& tensors) const {
    auto numeric_u64 = [&]() -> uint64_t {
        return std::visit(
            [&](const auto& v) -> uint64_t {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, DimExpr>) {
                    return v.resolve(tensors);
                } else {
                    if constexpr (std::is_floating_point_v<T>) {
                        if (v < 0.0) {
                            throw std::runtime_error("CudaKernelExpression cannot convert negative floating scalar to unsigned value.");
                        }
                    } else if constexpr (std::is_signed_v<T>) {
                        if (v < 0) {
                            throw std::runtime_error("CudaKernelExpression cannot convert negative signed scalar to unsigned value.");
                        }
                    }
                    return static_cast<uint64_t>(v);
                }
            },
            scalar.value);
    };

    auto numeric_i64 = [&]() -> int64_t {
        return std::visit(
            [&](const auto& v) -> int64_t {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, DimExpr>) {
                    const uint64_t u = v.resolve(tensors);
                    if (u > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
                        throw std::runtime_error("CudaKernelExpression scalar expression exceeds int64_t range for scalar '" + scalar.name +
                                                 "'.");
                    }
                    return static_cast<int64_t>(u);
                } else {
                    return static_cast<int64_t>(v);
                }
            },
            scalar.value);
    };

    switch (scalar.type) {
        case DataType::INT32: {
            const int64_t v = numeric_i64();
            if (v < std::numeric_limits<int32_t>::min() || v > std::numeric_limits<int32_t>::max()) {
                throw std::runtime_error("CudaKernelExpression scalar '" + scalar.name + "' is outside int32 range.");
            }
            return static_cast<int32_t>(v);
        }
        case DataType::UINT32: {
            const uint64_t v = numeric_u64();
            if (v > std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error("CudaKernelExpression scalar '" + scalar.name + "' is outside uint32 range.");
            }
            return static_cast<uint32_t>(v);
        }
        case DataType::INT64:
            return numeric_i64();
        case DataType::UINT64:
            return numeric_u64();
        case DataType::FP32:
            return std::visit(
                [&](const auto& v) -> float {
                    using T = std::decay_t<decltype(v)>;
                    if constexpr (std::is_same_v<T, DimExpr>) {
                        return static_cast<float>(v.resolve(tensors));
                    } else {
                        return static_cast<float>(v);
                    }
                },
                scalar.value);
        case DataType::FP64:
            return std::visit(
                [&](const auto& v) -> double {
                    using T = std::decay_t<decltype(v)>;
                    if constexpr (std::is_same_v<T, DimExpr>) {
                        return static_cast<double>(v.resolve(tensors));
                    } else {
                        return static_cast<double>(v);
                    }
                },
                scalar.value);
        default:
            throw std::runtime_error("CudaKernelExpression scalar '" + scalar.name + "' has unsupported dtype '" + dtypeName(scalar.type) +
                                     "'.");
    }
}

std::shared_ptr<StampedCudaKernel> CudaKernelExpression::stampCompiled(
    const std::shared_ptr<CompiledCudaKernel>& compiled,
    const std::unordered_map<std::string, Tensor>& bound_inputs,
    const std::unordered_map<std::string, Tensor>& preallocated_outputs,
    const std::unordered_map<std::string, std::vector<uint64_t>>& requested_output_shapes,
    const Stream& stream,
    std::unordered_map<std::string, Tensor>& resolved_outputs,
    const std::unordered_map<std::string, TensorScalarBinding>& tensor_scalar_inputs) const {
    if (!compiled) {
        throw std::runtime_error("CudaKernelExpression::stampCompiled requires a compiled kernel.");
    }
    size_t expected_tensor_inputs = 0;
    size_t expected_tensor_runtime_scalars = 0;
    size_t expected_host_runtime_scalars = 0;
    std::unordered_set<std::string> expected_tensor_names;
    std::unordered_set<std::string> expected_tensor_scalar_names;
    std::unordered_set<std::string> expected_host_scalar_names;
    for (const TensorParamSpec& input : inputs_) {
        switch (input.kind) {
            case TensorParamSpec::Kind::Tensor:
                ++expected_tensor_inputs;
                expected_tensor_names.insert(input.name);
                break;
            case TensorParamSpec::Kind::TensorRuntimeScalar:
                ++expected_tensor_runtime_scalars;
                expected_tensor_scalar_names.insert(input.name);
                break;
            case TensorParamSpec::Kind::HostRuntimeScalar:
                ++expected_host_runtime_scalars;
                expected_host_scalar_names.insert(input.name);
                break;
            default:
                throw std::runtime_error("CudaKernelExpression encountered unknown input parameter kind.");
        }
    }
    if (bound_inputs.size() != expected_tensor_inputs) {
        throw std::invalid_argument("CudaKernelExpression tensor input count mismatch. Expected {" +
                                    joinNames(std::vector<std::string>(expected_tensor_names.begin(), expected_tensor_names.end())) + "}.");
    }
    if (tensor_scalar_inputs.size() != expected_tensor_runtime_scalars) {
        throw std::invalid_argument(
            "CudaKernelExpression tensor runtime scalar input count mismatch. Expected {" +
            joinNames(std::vector<std::string>(expected_tensor_scalar_names.begin(), expected_tensor_scalar_names.end())) + "}.");
    }
    for (const auto& [name, _] : bound_inputs) {
        if (!expected_tensor_names.contains(name)) {
            throw std::invalid_argument("CudaKernelExpression received unexpected input tensor: " + name);
        }
    }
    for (const auto& [name, _] : tensor_scalar_inputs) {
        if (!expected_tensor_scalar_names.contains(name)) {
            throw std::invalid_argument("CudaKernelExpression received unexpected tensor runtime scalar input: " + name);
        }
    }

    TensorPlacement placement(TensorPlacement::MemDevices::GPU, stream.getGpuNum());
    resolved_outputs = allocateAndValidateOutputs(bound_inputs, preallocated_outputs, requested_output_shapes, placement);

    LaunchContext launch_context{bound_inputs, resolved_outputs, stream.getGpuNum()};
    CudaKernelLaunchConfig launch_config = launch_fn_ ? launch_fn_(launch_context) : launch_spec_->resolve(launch_context);

    std::unordered_map<std::string, Tensor> all_tensors = bound_inputs;
    for (const auto& [name, tensor] : resolved_outputs) {
        all_tensors.emplace(name, tensor);
    }

    std::vector<Tensor> ordered_inputs;
    ordered_inputs.reserve(expected_tensor_inputs);
    std::vector<TensorScalarBinding> ordered_tensor_scalars;
    ordered_tensor_scalars.reserve(expected_tensor_runtime_scalars);
    std::vector<Tensor> ordered_outputs;
    ordered_outputs.reserve(outputs_.size());
    (void)expected_host_runtime_scalars;

    std::vector<StampedCudaKernelParam> params;
    params.reserve(inputs_.size() + outputs_.size() + scalars_.size());

    int expected_device = stream.getGpuNum();
    for (const TensorParamSpec& spec : inputs_) {
        if (spec.kind == TensorParamSpec::Kind::Tensor) {
            const Tensor& input = lookupTensor(bound_inputs, spec.name, "kernel input parameter");
            validateDenseContiguous(input, spec.name, "input");
            if (input.getDataType() != spec.dtype) {
                throw std::runtime_error("CudaKernelExpression input '" + spec.name + "' dtype mismatch. Got " +
                                         dtypeName(input.getDataType()) + ", expected " + dtypeName(spec.dtype) + ".");
            }
            if (input.getPlacement().getDeviceNum() != expected_device) {
                throw std::runtime_error("CudaKernelExpression input '" + spec.name + "' GPU does not match stream GPU.");
            }
            const size_t idx = ordered_inputs.size();
            ordered_inputs.push_back(input);
            params.push_back(StampedCudaKernelParam{StampedCudaKernelParam::Kind::TensorInput, spec.name, idx, int32_t{0}});
        } else if (spec.kind == TensorParamSpec::Kind::HostRuntimeScalar) {
            validateHostRuntimeScalarDataType(spec.dtype);
            params.push_back(StampedCudaKernelParam{StampedCudaKernelParam::Kind::HostRuntimeScalar, spec.name, 0, float{0.0f}});
        } else {
            auto scalar_it = tensor_scalar_inputs.find(spec.name);
            if (scalar_it == tensor_scalar_inputs.end()) {
                throw std::runtime_error("CudaKernelExpression missing tensor runtime scalar input: " + spec.name);
            }
            const TensorScalarBinding& binding = scalar_it->second;
            if (!binding.buffer.isInitialized()) {
                throw std::runtime_error("CudaKernelExpression tensor runtime scalar '" + spec.name + "' buffer is not initialized.");
            }
            if (binding.buffer.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
                throw std::runtime_error("CudaKernelExpression tensor runtime scalar '" + spec.name + "' buffer must be on GPU.");
            }
            if (binding.buffer.getPlacement().getDeviceNum() != expected_device) {
                throw std::runtime_error("CudaKernelExpression tensor runtime scalar '" + spec.name + "' GPU does not match stream GPU.");
            }
            if (binding.sourceDType != spec.dtype) {
                throw std::runtime_error("CudaKernelExpression tensor runtime scalar '" + spec.name + "' dtype mismatch. Got " +
                                         dtypeName(binding.sourceDType) + ", expected " + dtypeName(spec.dtype) + ".");
            }
            const size_t elem_bytes = static_cast<size_t>(TensorDescriptor::getElementSizeInBytes(spec.dtype));
            if (binding.byteOffset + elem_bytes > binding.buffer.getArraySizeInBytes()) {
                throw std::runtime_error("CudaKernelExpression tensor runtime scalar '" + spec.name +
                                         "' binding exceeds backing buffer size.");
            }
            const size_t idx = ordered_tensor_scalars.size();
            ordered_tensor_scalars.push_back(binding);
            params.push_back(StampedCudaKernelParam{StampedCudaKernelParam::Kind::TensorRuntimeScalar, spec.name, idx, int32_t{0}});
        }
    }

    for (const OutputParamSpec& spec : outputs_) {
        const Tensor& output = lookupTensor(resolved_outputs, spec.name, "kernel output parameter");
        const size_t idx = ordered_outputs.size();
        ordered_outputs.push_back(output);
        params.push_back(StampedCudaKernelParam{StampedCudaKernelParam::Kind::TensorOutput, spec.name, idx, int32_t{0}});
    }

    for (const ScalarParamSpec& spec : scalars_) {
        params.push_back(StampedCudaKernelParam{StampedCudaKernelParam::Kind::Scalar, spec.name, 0, resolveScalar(spec, all_tensors)});
    }

    return std::make_shared<StampedCudaKernel>(
        compiled, ordered_inputs, ordered_tensor_scalars, ordered_outputs, params, launch_config, stream);
}

StampedExecutionPlan CudaKernelExpression::stamp(const std::unordered_map<std::string, Tensor>& bound_inputs,
                                                 const std::unordered_map<std::string, Tensor>& preallocated_outputs,
                                                 Stream& stream,
                                                 const std::unordered_map<std::string, TensorScalarBinding>& tensor_scalar_inputs) const {
    std::unordered_map<std::string, std::vector<uint64_t>> requested_output_shapes;
    std::unordered_map<std::string, Tensor> outputs;
    std::shared_ptr<StampedCudaKernel> stamped = stampCompiled(
        compile(stream.getGpuNum()), bound_inputs, preallocated_outputs, requested_output_shapes, stream, outputs, tensor_scalar_inputs);
    std::vector<StampedExecutionStage> stages;
    stages.emplace_back(stamped);
    return StampedExecutionPlan(std::move(stages), std::move(outputs), stream);
}

Outputs CudaKernelExpression::apply(const std::unordered_map<std::string, Expression>& input_exprs) const {
    if (input_exprs.size() != inputs_.size()) {
        std::vector<std::string> expected;
        expected.reserve(inputs_.size());
        for (const TensorParamSpec& input : inputs_) {
            expected.push_back(input.name);
        }
        throw std::invalid_argument("CudaKernelExpression apply input count mismatch. Expected {" + joinNames(expected) + "}.");
    }

    std::vector<PhysicalExpression> input_physical_exprs;
    input_physical_exprs.reserve(inputs_.size());
    for (const TensorParamSpec& input : inputs_) {
        auto it = input_exprs.find(input.name);
        if (it == input_exprs.end()) {
            throw std::invalid_argument("CudaKernelExpression apply missing input expression: " + input.name);
        }
        PhysicalExpression physical = it->second.expression();
        if (physical.output_node >= physical.nodes.size()) {
            throw std::invalid_argument("CudaKernelExpression apply input expression has invalid output node: " + input.name);
        }
        const ExprOp actual_op = physical.nodes[physical.output_node].op;
        if (input.kind == TensorParamSpec::Kind::TensorRuntimeScalar && actual_op != ExprOp::TENSOR_RUNTIME_SCALAR) {
            throw std::invalid_argument("CudaKernelExpression input '" + input.name + "' expects Expression::tensorRuntimeScalar.");
        }
        if (input.kind == TensorParamSpec::Kind::HostRuntimeScalar && actual_op != ExprOp::RUNTIME_SCALAR) {
            throw std::invalid_argument("CudaKernelExpression input '" + input.name + "' expects Expression::runtimeScalar.");
        }
        if (input.kind == TensorParamSpec::Kind::Tensor &&
            (actual_op == ExprOp::TENSOR_RUNTIME_SCALAR || actual_op == ExprOp::RUNTIME_SCALAR)) {
            throw std::invalid_argument("CudaKernelExpression tensor input '" + input.name + "' received a runtime scalar expression.");
        }
        input_physical_exprs.push_back(std::move(physical));
    }
    for (const auto& [name, _] : input_exprs) {
        const bool known = std::any_of(inputs_.begin(), inputs_.end(), [&](const TensorParamSpec& spec) { return spec.name == name; });
        if (!known) {
            throw std::invalid_argument("CudaKernelExpression apply received unexpected input expression: " + name);
        }
    }

    auto dst = std::make_shared<PhysicalExpression>();
    std::unordered_map<std::string, uint32_t> dst_input_slots_by_name;
    std::vector<uint32_t> kernel_input_nodes;
    kernel_input_nodes.reserve(inputs_.size());

    for (size_t i = 0; i < input_physical_exprs.size(); ++i) {
        uint32_t remapped_output = Expression::cloneInto(input_physical_exprs[i], *dst, dst_input_slots_by_name);
        if (inputs_[i].kind == TensorParamSpec::Kind::TensorRuntimeScalar || inputs_[i].kind == TensorParamSpec::Kind::HostRuntimeScalar) {
            ExprNode& node = dst->nodes.at(remapped_output);
            node.input_tensor_dtype = inputs_[i].dtype;
            node.output_dtype = inputs_[i].dtype;
            node.compute_dtype = inputs_[i].dtype;
            node.backward_output_dtype = inputs_[i].dtype;
            node.backward_compute_dtype = inputs_[i].dtype;
        }
        kernel_input_nodes.push_back(remapped_output);
    }

    const uint32_t kernel_spec_index = static_cast<uint32_t>(dst->cuda_kernel_expressions.size());
    dst->cuda_kernel_expressions.push_back(std::make_shared<CudaKernelExpression>(*this));

    std::vector<NamedOutput> named_outputs;
    named_outputs.reserve(outputs_.size());
    for (uint32_t output_idx = 0; output_idx < outputs_.size(); ++output_idx) {
        ExprNode node;
        node.op = ExprOp::CUDA_KERNEL_OUTPUT;
        node.cuda_kernel_spec_index = kernel_spec_index;
        node.cuda_kernel_output_index = output_idx;
        node.cuda_kernel_input_nodes = kernel_input_nodes;
        node.output_dtype = outputs_[output_idx].dtype;
        node.compute_dtype = outputs_[output_idx].dtype;
        node.backward_output_dtype = outputs_[output_idx].dtype;
        node.backward_compute_dtype = outputs_[output_idx].dtype;

        const uint32_t node_idx = static_cast<uint32_t>(dst->nodes.size());
        dst->nodes.push_back(std::move(node));
        named_outputs.push_back(NamedOutput{outputs_[output_idx].name, node_idx});
    }

    return Outputs::fromPhysicalOutputs(PhysicalOutputs{.expr = dst, .outputs = std::move(named_outputs)});
}

DynamicExpression CudaKernelExpression::asDynamicExpression() const {
    std::unordered_map<std::string, Expression> expression_inputs;
    expression_inputs.reserve(inputs_.size());
    std::vector<std::string> expected_inputs;
    expected_inputs.reserve(inputs_.size());
    for (const TensorParamSpec& input : inputs_) {
        if (input.kind == TensorParamSpec::Kind::TensorRuntimeScalar) {
            expression_inputs.emplace(input.name, Expression::tensorRuntimeScalar(input.name, input.dtype, input.dtype));
        } else if (input.kind == TensorParamSpec::Kind::HostRuntimeScalar) {
            expression_inputs.emplace(input.name, Expression::runtimeScalar(input.name, input.dtype, input.dtype));
        } else {
            expression_inputs.emplace(input.name, Expression::input(input.name, input.dtype, input.dtype));
        }
        expected_inputs.push_back(input.name);
    }

    Outputs outputs = apply(expression_inputs);
    ExpressionDefinition definition = ExpressionDefinition::fromOutputs(outputs);
    definition.expected_input_names = expected_inputs;
    definition.expected_output_names.clear();
    definition.expected_output_names.reserve(outputs_.size());
    for (const OutputParamSpec& output : outputs_) {
        definition.expected_output_names.push_back(output.name);
    }
    return DynamicExpression::fromExpressionDefinition(definition);
}

}  // namespace ThorImplementation
