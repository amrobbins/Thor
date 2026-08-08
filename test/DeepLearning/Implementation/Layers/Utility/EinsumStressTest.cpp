#include "DeepLearning/Implementation/Layers/Utility/EinsumLayer.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkOutput.h"
#include "Utilities/TensorOperations/Einsum/Einsum.h"
#include "Utilities/TensorOperations/Einsum/EinsumParser.h"
#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"
#include "test/DeepLearning/Implementation/Layers/NoOpLayer.h"

#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace ThorImplementation;

#define REQUIRE_CUDA_DEVICE()                                                                                          \
    do {                                                                                                               \
        int device_count = 0;                                                                                          \
        const cudaError_t status = cudaGetDeviceCount(&device_count);                                                 \
        if (status != cudaSuccess || device_count == 0) {                                                             \
            GTEST_SKIP() << "CUDA device required for einsum stress test";                                           \
        }                                                                                                              \
    } while (false)

namespace {

constexpr char kPhysicalBatchLabel = 'Z';

struct RandomizedLayerEinsumCase {
    std::string equation;
    std::vector<std::vector<uint64_t>> feature_dimensions;
};

struct ForwardRunResult {
    std::vector<uint64_t> dimensions;
    std::vector<float> values;
};

struct BackwardRunResult {
    std::vector<Tensor> gradients_cpu;
};

class EinsumStressGradientCaptureLayer final : public NoOpLayer {
   public:
    void backward(std::optional<Tensor> errorInput, uint32_t batchSize = 0) override {
        captured_errors.push_back(errorInput);
        captured_batch_sizes.push_back(batchSize);
    }

    std::vector<std::optional<Tensor>> captured_errors;
    std::vector<uint32_t> captured_batch_sizes;
};

uint32_t stressSeed(const char* environment_name, uint32_t default_seed) {
    const char* value = std::getenv(environment_name);
    return value != nullptr ? static_cast<uint32_t>(std::stoul(value, nullptr, 0)) : default_seed;
}

int positiveStressSetting(const char* environment_name, int default_value) {
    const char* value = std::getenv(environment_name);
    const int parsed = value != nullptr ? std::stoi(value) : default_value;
    if (parsed <= 0) {
        throw std::invalid_argument(std::string(environment_name) + " must be positive.");
    }
    return parsed;
}

std::string describeDimensions(const std::vector<std::vector<uint64_t>>& dimensions) {
    std::ostringstream out;
    out << '[';
    for (size_t operand = 0; operand < dimensions.size(); ++operand) {
        if (operand != 0) out << ',';
        out << '[';
        for (size_t axis = 0; axis < dimensions[operand].size(); ++axis) {
            if (axis != 0) out << ',';
            out << dimensions[operand][axis];
        }
        out << ']';
    }
    out << ']';
    return out.str();
}

std::vector<float> randomSmallValues(size_t count, std::mt19937& rng) {
    std::uniform_int_distribution<int> distribution(-8, 8);
    std::vector<float> values(count);
    for (float& value : values) {
        value = static_cast<float>(distribution(rng)) * 0.125f;
    }
    return values;
}

size_t elementCount(const std::vector<uint64_t>& dimensions) {
    size_t count = 1;
    for (uint64_t dimension : dimensions) {
        if (dimension == 0 || count > std::numeric_limits<size_t>::max() / dimension) {
            throw std::overflow_error("Randomized einsum stress tensor element count overflowed size_t.");
        }
        count *= static_cast<size_t>(dimension);
    }
    return count;
}

RandomizedLayerEinsumCase makeRandomizedLayerEinsumCase(size_t operand_count, int trial, std::mt19937& rng) {
    if (operand_count < 2 || operand_count > 10) {
        throw std::invalid_argument("Randomized layer einsum stress cases support two through ten operands.");
    }

    // Z is deliberately reserved for the implicit physical batch label used by
    // the independent whole-equation generic reference below.
    static const std::string label_pool = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXY";
    size_t next_label_index = 0;
    const auto next_label = [&]() -> char {
        if (next_label_index >= label_pool.size()) {
            throw std::logic_error("Randomized layer einsum stress case exhausted the label pool.");
        }
        return label_pool[next_label_index++];
    };

    std::vector<std::vector<char>> operand_labels(operand_count);
    std::vector<std::vector<std::pair<char, uint64_t>>> local_dimensions(operand_count);
    const auto add_label = [&](size_t operand, char label, uint64_t dimension) {
        operand_labels.at(operand).push_back(label);
        auto& dimensions_for_operand = local_dimensions.at(operand);
        const auto existing = std::find_if(dimensions_for_operand.begin(),
                                           dimensions_for_operand.end(),
                                           [&](const auto& item) { return item.first == label; });
        if (existing == dimensions_for_operand.end()) {
            dimensions_for_operand.emplace_back(label, dimension);
        } else if (existing->second != dimension) {
            throw std::logic_error("Randomized layer einsum stress case assigned inconsistent local dimensions.");
        }
    };
    const auto local_dimension = [&](size_t operand, char label) -> uint64_t {
        const auto& dimensions_for_operand = local_dimensions.at(operand);
        const auto existing = std::find_if(dimensions_for_operand.begin(),
                                           dimensions_for_operand.end(),
                                           [&](const auto& item) { return item.first == label; });
        if (existing == dimensions_for_operand.end()) {
            throw std::logic_error("Randomized layer einsum stress case is missing a local dimension.");
        }
        return existing->second;
    };

    // Every case has a connected chain backbone.  Additional structure below
    // exercises local reductions, broadcasted surviving labels, cross-links,
    // repeated-label diagonals, axis permutations, and both exact and beam
    // multi-operand planning while keeping tensors intentionally tiny.
    std::vector<char> connector_labels(operand_count + 1);
    for (char& label : connector_labels) label = next_label();
    for (size_t operand = 0; operand < operand_count; ++operand) {
        add_label(operand, connector_labels[operand], 2);
        add_label(operand, connector_labels[operand + 1], 2);
    }

    std::vector<char> output_labels = {connector_labels.front(), connector_labels.back()};

    if ((operand_count % 2) == 1) {
        const char shared_surviving_label = next_label();
        output_labels.push_back(shared_surviving_label);
        std::bernoulli_distribution singleton_distribution(0.35);
        bool any_non_singleton = false;
        for (size_t operand = 0; operand < operand_count; ++operand) {
            const uint64_t dimension = singleton_distribution(rng) ? 1 : 2;
            any_non_singleton = any_non_singleton || dimension == 2;
            add_label(operand, shared_surviving_label, dimension);
        }
        if (!any_non_singleton) {
            auto& first_dimensions = local_dimensions.front();
            auto item = std::find_if(first_dimensions.begin(), first_dimensions.end(), [&](const auto& candidate) {
                return candidate.first == shared_surviving_label;
            });
            item->second = 2;
        }
    }

    if ((operand_count % 3) != 1) {
        std::uniform_int_distribution<size_t> operand_distribution(0, operand_count - 1);
        add_label(operand_distribution(rng), next_label(), 2);
    }

    if (operand_count >= 4 && (operand_count % 2) == 0) {
        std::uniform_int_distribution<size_t> first_distribution(0, operand_count - 3);
        const size_t first = first_distribution(rng);
        std::uniform_int_distribution<size_t> second_distribution(first + 2, operand_count - 1);
        const size_t second = second_distribution(rng);
        const char cross_label = next_label();
        add_label(first, cross_label, 2);
        add_label(second, cross_label, 2);
    }

    if (operand_count >= 3 && (operand_count % 3) == 0) {
        std::uniform_int_distribution<size_t> operand_distribution(0, operand_count - 1);
        const size_t operand = operand_distribution(rng);
        std::uniform_int_distribution<size_t> axis_distribution(0, operand_labels[operand].size() - 1);
        const char diagonal_label = operand_labels[operand][axis_distribution(rng)];
        add_label(operand, diagonal_label, local_dimension(operand, diagonal_label));
    }

    if (trial > 0 && operand_count >= 4) {
        std::uniform_int_distribution<size_t> operand_distribution(0, operand_count - 1);
        add_label(operand_distribution(rng), next_label(), 2);
    }

    std::vector<std::vector<uint64_t>> dimensions(operand_count);
    std::vector<std::string> subscripts(operand_count);
    for (size_t operand = 0; operand < operand_count; ++operand) {
        std::shuffle(operand_labels[operand].begin(), operand_labels[operand].end(), rng);
        subscripts[operand].assign(operand_labels[operand].begin(), operand_labels[operand].end());
        dimensions[operand].reserve(operand_labels[operand].size());
        for (char label : operand_labels[operand]) {
            dimensions[operand].push_back(local_dimension(operand, label));
        }
    }
    std::shuffle(output_labels.begin(), output_labels.end(), rng);

    std::ostringstream equation;
    for (size_t operand = 0; operand < operand_count; ++operand) {
        if (operand != 0) equation << ',';
        equation << subscripts[operand];
    }
    equation << "->";
    for (char label : output_labels) equation << label;

    return RandomizedLayerEinsumCase{equation.str(), std::move(dimensions)};
}

std::string addPhysicalBatchLabel(const std::string& feature_equation) {
    const size_t arrow = feature_equation.find("->");
    if (arrow == std::string::npos) {
        throw std::invalid_argument("Einsum layer stress equations must use an explicit output.");
    }

    std::ostringstream physical;
    size_t operand_begin = 0;
    bool first = true;
    while (operand_begin < arrow) {
        const size_t comma = feature_equation.find(',', operand_begin);
        const size_t operand_end = comma == std::string::npos || comma > arrow ? arrow : comma;
        if (!first) physical << ',';
        first = false;
        physical << kPhysicalBatchLabel << feature_equation.substr(operand_begin, operand_end - operand_begin);
        if (operand_end == arrow) break;
        operand_begin = operand_end + 1;
    }
    physical << "->" << kPhysicalBatchLabel << feature_equation.substr(arrow + 2);
    return physical.str();
}

std::vector<Tensor> makeRandomCpuInputs(const RandomizedLayerEinsumCase& test_case,
                                        uint32_t batch,
                                        std::mt19937& rng) {
    TensorPlacement cpu_placement(TensorPlacement::MemDevices::CPU);
    std::vector<Tensor> inputs;
    inputs.reserve(test_case.feature_dimensions.size());
    for (const auto& feature_dimensions : test_case.feature_dimensions) {
        std::vector<uint64_t> physical_dimensions;
        physical_dimensions.reserve(feature_dimensions.size() + 1);
        physical_dimensions.push_back(batch);
        physical_dimensions.insert(physical_dimensions.end(), feature_dimensions.begin(), feature_dimensions.end());
        inputs.emplace_back(cpu_placement, TensorDescriptor(DataType::FP32, physical_dimensions));
        const std::vector<float> values = randomSmallValues(elementCount(physical_dimensions), rng);
        std::copy(values.begin(), values.end(), inputs.back().getMemPtr<float>());
    }
    return inputs;
}

ForwardRunResult runLayerForward(const std::string& equation,
                                 const std::vector<Tensor>& input_cpu,
                                 uint32_t valid_example_count) {
    TensorPlacement cpu_placement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpu_placement(TensorPlacement::MemDevices::GPU, 0);

    std::vector<std::shared_ptr<NetworkInput>> network_inputs;
    std::vector<std::shared_ptr<Layer>> layers;
    network_inputs.reserve(input_cpu.size());
    layers.reserve(input_cpu.size() + 2);

    for (const Tensor& input : input_cpu) {
        const std::vector<uint64_t> input_dimensions = input.getDimensions();
        std::vector<unsigned long> dimensions(input_dimensions.begin(), input_dimensions.end());
        auto network_input = std::make_shared<NetworkInput>(gpu_placement, DataType::FP32, dimensions);
        network_inputs.push_back(network_input);
        layers.push_back(network_input);
    }

    auto einsum = std::make_shared<EinsumLayer>(equation);
    auto output = std::make_shared<NetworkOutput>(cpu_placement);
    layers.push_back(einsum);
    layers.push_back(output);

    for (size_t operand = 0; operand < network_inputs.size(); ++operand) {
        network_inputs[operand]->connectToNextLayer(einsum.get(), 0, static_cast<int>(operand));
    }
    einsum->connectToNextLayer(output.get());

    LayerTestHelper::initializeNetwork(layers);
    for (size_t operand = 0; operand < network_inputs.size(); ++operand) {
        network_inputs[operand]->forward(input_cpu[operand], false, valid_example_count);
    }
    Stream stream = einsum->getStream();
    stream.waitEvent(output->getOutputReadyEvent());
    stream.synchronize();

    if (!output->getFeatureOutput().has_value()) {
        LayerTestHelper::tearDownNetwork(layers);
        throw std::runtime_error("Einsum layer stress forward did not produce an output tensor.");
    }
    const Tensor output_cpu = output->getFeatureOutput().value();
    ForwardRunResult result;
    result.dimensions = output_cpu.getDimensions();
    result.values.resize(output_cpu.getTotalNumElements());
    std::copy(output_cpu.getMemPtr<float>(),
              output_cpu.getMemPtr<float>() + output_cpu.getTotalNumElements(),
              result.values.begin());
    LayerTestHelper::tearDownNetwork(layers);
    return result;
}

ForwardRunResult runWholeEquationGenericReference(const std::string& feature_equation,
                                                  const std::vector<Tensor>& input_cpu) {
    TensorPlacement cpu_placement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpu_placement(TensorPlacement::MemDevices::GPU, 0);
    Stream stream(0);

    std::vector<Tensor> inputs_gpu;
    inputs_gpu.reserve(input_cpu.size());
    for (const Tensor& input : input_cpu) {
        inputs_gpu.emplace_back(gpu_placement, input.getDescriptor());
        inputs_gpu.back().copyFromAsync(input, stream);
    }

    auto reference = Einsum(addPhysicalBatchLabel(feature_equation)).stampGenericReference(inputs_gpu, stream);
    reference->run();
    Tensor output_cpu = reference->getOutputTensor().clone(cpu_placement);
    output_cpu.copyFromAsync(reference->getOutputTensor(), stream);
    stream.synchronize();

    ForwardRunResult result;
    result.dimensions = output_cpu.getDimensions();
    result.values.resize(output_cpu.getTotalNumElements());
    std::copy(output_cpu.getMemPtr<float>(),
              output_cpu.getMemPtr<float>() + output_cpu.getTotalNumElements(),
              result.values.begin());
    return result;
}

BackwardRunResult runLayerBackward(const std::string& equation,
                                   const std::vector<Tensor>& input_cpu,
                                   const Tensor& upstream_gradient_cpu,
                                   uint32_t batch) {
    TensorPlacement cpu_placement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpu_placement(TensorPlacement::MemDevices::GPU, 0);

    std::vector<Stream> streams;
    std::vector<Tensor> input_gpu;
    std::vector<std::unique_ptr<EinsumStressGradientCaptureLayer>> captures;
    streams.reserve(input_cpu.size());
    input_gpu.reserve(input_cpu.size());
    captures.reserve(input_cpu.size());

    auto einsum = std::make_unique<EinsumLayer>(equation);
    for (size_t operand = 0; operand < input_cpu.size(); ++operand) {
        streams.emplace_back(0);
        input_gpu.emplace_back(gpu_placement, input_cpu[operand].getDescriptor());
        input_gpu.back().copyFromAsync(input_cpu[operand], streams.back());
        captures.push_back(std::make_unique<EinsumStressGradientCaptureLayer>());
        einsum->connectToPreviousLayer(captures.back().get(),
                                       input_gpu.back(),
                                       streams.back(),
                                       true,
                                       static_cast<int>(operand));
    }
    for (size_t operand = 1; operand < streams.size(); ++operand) {
        streams[0].waitEvent(streams[operand].putEvent());
    }

    NoOpLayer sink;
    einsum->connectToNextLayer(&sink);
    einsum->compile();
    einsum->initialize();

    std::vector<std::optional<Tensor>> error_inputs = einsum->getErrorInputs();
    if (error_inputs.size() != 1 || !error_inputs[0].has_value()) {
        throw std::runtime_error("Einsum layer stress backward expected one downstream gradient tensor.");
    }
    error_inputs[0]->copyFromAsync(upstream_gradient_cpu, streams[0]);
    einsum->backward(error_inputs[0], batch);

    BackwardRunResult result;
    const std::vector<std::optional<Tensor>> error_outputs = einsum->getErrorOutputs();
    result.gradients_cpu.reserve(error_outputs.size());
    for (size_t operand = 0; operand < error_outputs.size(); ++operand) {
        if (!error_outputs[operand].has_value()) {
            throw std::runtime_error("Einsum layer stress backward unexpectedly pruned an operand gradient.");
        }
        Tensor gradient_cpu = error_outputs[operand]->clone(cpu_placement);
        gradient_cpu.copyFromAsync(error_outputs[operand].value(), streams[operand]);
        streams[operand].synchronize();
        result.gradients_cpu.push_back(std::move(gradient_cpu));
    }

    einsum->cleanup();
    return result;
}

double genericReferenceLoss(const std::string& feature_equation,
                            const std::vector<Tensor>& input_cpu,
                            const Tensor& upstream_gradient_cpu) {
    const ForwardRunResult forward = runWholeEquationGenericReference(feature_equation, input_cpu);
    if (forward.values.size() != upstream_gradient_cpu.getTotalNumElements()) {
        throw std::logic_error("Einsum backward stress reference output size does not match upstream gradient.");
    }
    const float* upstream = upstream_gradient_cpu.getMemPtr<float>();
    double loss = 0.0;
    for (size_t element = 0; element < forward.values.size(); ++element) {
        loss += static_cast<double>(forward.values[element]) * static_cast<double>(upstream[element]);
    }
    return loss;
}

std::vector<size_t> sampledGradientElements(size_t element_count, int requested_samples, std::mt19937& rng) {
    const size_t target = std::min(element_count, static_cast<size_t>(requested_samples));
    std::vector<size_t> result;
    result.reserve(target);
    const auto add = [&](size_t index) {
        if (result.size() < target && std::find(result.begin(), result.end(), index) == result.end()) {
            result.push_back(index);
        }
    };
    if (element_count != 0) {
        add(0);
        if (element_count > 1) add(1);  // Often exercises an off-diagonal repeated-label element.
        add(element_count - 1);
    }
    if (result.size() < target) {
        std::uniform_int_distribution<size_t> distribution(0, element_count - 1);
        while (result.size() < target) add(distribution(rng));
    }
    return result;
}

}  // namespace

// These stress tests are intentionally disabled by default. Run them explicitly with:
//   ./thor_tests --gtest_also_run_disabled_tests \
//     --gtest_filter='EinsumLayerStress.DISABLED_*'
// The environment variables printed on failure can be used to reproduce or
// scale a run without changing the normal Thor test-suite cost.
TEST(EinsumLayerStress, DISABLED_RandomizedForwardMatchesWholeEquationGenericReference) {
    REQUIRE_CUDA_DEVICE();
    const uint32_t seed = stressSeed("THOR_EINSUM_LAYER_FORWARD_STRESS_SEED", 0x315F0A11u);
    const int runs_per_operand_count =
        positiveStressSetting("THOR_EINSUM_LAYER_FORWARD_STRESS_RUNS_PER_OPERAND_COUNT", 10);
    constexpr uint32_t batch = 2;
    std::mt19937 rng(seed);

    for (size_t operand_count = 2; operand_count <= 10; ++operand_count) {
        for (int trial = 0; trial < runs_per_operand_count; ++trial) {
            const RandomizedLayerEinsumCase test_case = makeRandomizedLayerEinsumCase(operand_count, trial, rng);
            try {
                const std::vector<Tensor> inputs = makeRandomCpuInputs(test_case, batch, rng);
                const ForwardRunResult actual = runLayerForward(test_case.equation, inputs, batch);
                const ForwardRunResult expected = runWholeEquationGenericReference(test_case.equation, inputs);

                ASSERT_EQ(actual.dimensions, expected.dimensions)
                    << "equation=" << test_case.equation
                    << " feature_dimensions=" << describeDimensions(test_case.feature_dimensions);
                ASSERT_EQ(actual.values.size(), expected.values.size());
                for (size_t element = 0; element < actual.values.size(); ++element) {
                    const float tolerance = 7.5e-4f + 7.5e-4f * std::abs(expected.values[element]);
                    if (std::abs(actual.values[element] - expected.values[element]) > tolerance) {
                        FAIL() << "Randomized EinsumLayer forward stress mismatch. Reproduce with "
                               << "THOR_EINSUM_LAYER_FORWARD_STRESS_SEED=" << seed << ' '
                               << "THOR_EINSUM_LAYER_FORWARD_STRESS_RUNS_PER_OPERAND_COUNT="
                               << runs_per_operand_count << ". operand_count=" << operand_count
                               << " trial=" << trial << " equation=" << test_case.equation
                               << " feature_dimensions=" << describeDimensions(test_case.feature_dimensions)
                               << " element=" << element << " actual=" << actual.values[element]
                               << " expected=" << expected.values[element] << " tolerance=" << tolerance;
                    }
                }
            } catch (const std::exception& error) {
                FAIL() << "Randomized EinsumLayer forward stress threw. Reproduce with "
                       << "THOR_EINSUM_LAYER_FORWARD_STRESS_SEED=" << seed << ' '
                       << "THOR_EINSUM_LAYER_FORWARD_STRESS_RUNS_PER_OPERAND_COUNT="
                       << runs_per_operand_count << ". operand_count=" << operand_count
                       << " trial=" << trial << " equation=" << test_case.equation
                       << " feature_dimensions=" << describeDimensions(test_case.feature_dimensions)
                       << " exception=" << error.what();
            }
        }
    }
}

TEST(EinsumLayerStress, DISABLED_RandomizedBackwardMatchesFiniteDifferenceReference) {
    REQUIRE_CUDA_DEVICE();
    const uint32_t seed = stressSeed("THOR_EINSUM_LAYER_BACKWARD_STRESS_SEED", 0x315BAC4Du);
    const int runs_per_operand_count =
        positiveStressSetting("THOR_EINSUM_LAYER_BACKWARD_STRESS_RUNS_PER_OPERAND_COUNT", 2);
    const int samples_per_operand =
        positiveStressSetting("THOR_EINSUM_LAYER_BACKWARD_STRESS_SAMPLES_PER_OPERAND", 3);
    constexpr uint32_t batch = 2;
    constexpr float epsilon = 1.0e-2f;
    std::mt19937 rng(seed);

    for (size_t operand_count = 2; operand_count <= 10; ++operand_count) {
        for (int trial = 0; trial < runs_per_operand_count; ++trial) {
            const RandomizedLayerEinsumCase test_case = makeRandomizedLayerEinsumCase(operand_count, trial, rng);
            try {
                std::vector<Tensor> inputs = makeRandomCpuInputs(test_case, batch, rng);
                const ResolvedEinsumEquation resolved =
                    EinsumParser::parseAndResolve(test_case.equation, test_case.feature_dimensions);
                std::vector<uint64_t> upstream_dimensions;
                upstream_dimensions.reserve(resolved.output_dimensions.size() + 1);
                upstream_dimensions.push_back(batch);
                upstream_dimensions.insert(upstream_dimensions.end(),
                                           resolved.output_dimensions.begin(),
                                           resolved.output_dimensions.end());
                TensorPlacement cpu_placement(TensorPlacement::MemDevices::CPU);
                Tensor upstream(cpu_placement, TensorDescriptor(DataType::FP32, upstream_dimensions));
                const std::vector<float> upstream_values = randomSmallValues(upstream.getTotalNumElements(), rng);
                std::copy(upstream_values.begin(), upstream_values.end(), upstream.getMemPtr<float>());

                const BackwardRunResult analytic = runLayerBackward(test_case.equation, inputs, upstream, batch);
                ASSERT_EQ(analytic.gradients_cpu.size(), inputs.size());

                for (size_t operand = 0; operand < inputs.size(); ++operand) {
                    Tensor& input = inputs[operand];
                    const Tensor& gradient = analytic.gradients_cpu[operand];
                    ASSERT_EQ(input.getDimensions(), gradient.getDimensions());
                    const std::vector<size_t> sample_indices =
                        sampledGradientElements(input.getTotalNumElements(), samples_per_operand, rng);
                    for (size_t element : sample_indices) {
                        float* input_values = input.getMemPtr<float>();
                        const float original = input_values[element];
                        input_values[element] = original + epsilon;
                        const double plus = genericReferenceLoss(test_case.equation, inputs, upstream);
                        input_values[element] = original - epsilon;
                        const double minus = genericReferenceLoss(test_case.equation, inputs, upstream);
                        input_values[element] = original;

                        const double finite_difference = (plus - minus) / (2.0 * epsilon);
                        const double actual = gradient.getMemPtr<float>()[element];
                        const double tolerance = 5.0e-3 + 3.0e-2 * std::abs(finite_difference);
                        if (std::abs(actual - finite_difference) > tolerance) {
                            FAIL() << "Randomized EinsumLayer backward stress mismatch. Reproduce with "
                                   << "THOR_EINSUM_LAYER_BACKWARD_STRESS_SEED=" << seed << ' '
                                   << "THOR_EINSUM_LAYER_BACKWARD_STRESS_RUNS_PER_OPERAND_COUNT="
                                   << runs_per_operand_count << ' '
                                   << "THOR_EINSUM_LAYER_BACKWARD_STRESS_SAMPLES_PER_OPERAND="
                                   << samples_per_operand << ". operand_count=" << operand_count
                                   << " trial=" << trial << " operand=" << operand
                                   << " element=" << element << " equation=" << test_case.equation
                                   << " feature_dimensions=" << describeDimensions(test_case.feature_dimensions)
                                   << " actual=" << actual << " finite_difference=" << finite_difference
                                   << " tolerance=" << tolerance;
                        }
                    }
                }
            } catch (const std::exception& error) {
                FAIL() << "Randomized EinsumLayer backward stress threw. Reproduce with "
                       << "THOR_EINSUM_LAYER_BACKWARD_STRESS_SEED=" << seed << ' '
                       << "THOR_EINSUM_LAYER_BACKWARD_STRESS_RUNS_PER_OPERAND_COUNT="
                       << runs_per_operand_count << ' '
                       << "THOR_EINSUM_LAYER_BACKWARD_STRESS_SAMPLES_PER_OPERAND="
                       << samples_per_operand << ". operand_count=" << operand_count
                       << " trial=" << trial << " equation=" << test_case.equation
                       << " feature_dimensions=" << describeDimensions(test_case.feature_dimensions)
                       << " exception=" << error.what();
            }
        }
    }
}
