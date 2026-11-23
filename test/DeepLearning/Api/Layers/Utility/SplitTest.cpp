#include "DeepLearning/Api/Layers/Utility/Split.h"
#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Layers/Loss/RaggedCustomLoss.h"
#include "DeepLearning/Api/Layers/Utility/Concatenate.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkOutput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/Layers/Loss/RaggedCustomLoss.h"
#include "DeepLearning/Implementation/Layers/Utility/RaggedSplit.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"
#include "test/DeepLearning/Api/Helpers/GradientRivet.h"

#include "gtest/gtest.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

using namespace Thor;
using namespace std;
using json = nlohmann::json;
namespace Impl = ThorImplementation;

namespace {

void expectLogicError(const std::function<void()> &build, const std::string &fragment) {
    try {
        build();
        FAIL() << "Expected a logic error containing: " << fragment;
    } catch (const std::logic_error &error) {
        EXPECT_NE(std::string(error.what()).find(fragment), std::string::npos) << error.what();
    }
}

}  // namespace

TEST(UtilityApiLayers, SplitBuildsPiecesOfTheirSizesAndTypes) {
    Network network("split_builds");
    Tensor input(DataType::FP32, {3, 7, 5});

    Split same = Split::Builder().network(network).featureInput(input).axis(1).sizes({2, 4, 1}).build();
    const vector<Tensor> pieces = same.getFeatureOutputs();
    ASSERT_EQ(pieces.size(), 3U);
    EXPECT_EQ(pieces[0].getDimensions(), (vector<uint64_t>{3, 2, 5}));
    EXPECT_EQ(pieces[1].getDimensions(), (vector<uint64_t>{3, 4, 5}));
    EXPECT_EQ(pieces[2].getDimensions(), (vector<uint64_t>{3, 1, 5}));
    for (const Tensor &piece : pieces) EXPECT_EQ(piece.getDataType(), DataType::FP32);
    EXPECT_EQ(same.getAxis(), 1U);
    EXPECT_EQ(same.getSizes(), (vector<uint64_t>{2, 4, 1}));
    EXPECT_EQ(same.getOutputsFromInput(input), pieces);
    for (uint32_t i = 0; i < pieces.size(); ++i) {
        EXPECT_EQ(same.getConnectionType(pieces[i]), static_cast<int>(i));
        EXPECT_EQ(same.getFeatureInput(pieces[i]), input);
    }
    EXPECT_EQ(same.getConnectionType(input), 0);

    Tensor other(DataType::FP32, {3, 7, 5});
    Split typed = Split::Builder()
                      .network(network)
                      .featureInput(other)
                      .axis(2)
                      .sizes({1, 4})
                      .outputDataTypes({DataType::BF16, DataType::FP8_E5M2})
                      .build();
    EXPECT_EQ(typed.getFeatureOutputs()[0].getDataType(), DataType::BF16);
    EXPECT_EQ(typed.getFeatureOutputs()[1].getDataType(), DataType::FP8_E5M2);
    EXPECT_EQ(typed.getFeatureOutputs()[1].getDimensions(), (vector<uint64_t>{3, 7, 4}));

    auto builder = [&](uint32_t axis, vector<uint64_t> sizes) {
        Split::Builder b;
        b.network(network).featureInput(Tensor(DataType::FP16, {3, 7, 5})).axis(axis).sizes(std::move(sizes));
        return b;
    };
    expectLogicError([&] { builder(1, {2, 4}).build(); }, "sum to 6");
    expectLogicError([&] { builder(1, {7, 0}).build(); }, "positive");
    expectLogicError([&] { builder(1, {7}).build(); }, "two or more");
    expectLogicError([&] { builder(3, {2, 3}).build(); }, "out of range");
    expectLogicError([&] { builder(1, {3, 4}).outputDataTypes({DataType::FP32}).build(); }, "one output data type per piece");
}

TEST(UtilityApiLayers, SplitSerializesAndDeserializes) {
    Network network("split_serializes");
    NetworkInput x = NetworkInput::Builder().network(network).name("x").dimensions({4, 6}).dataType(DataType::FP32).build();
    Split split = Split::Builder()
                      .network(network)
                      .featureInput(x.getFeatureOutput().value())
                      .axis(1)
                      .sizes({1, 5})
                      .outputDataTypes({DataType::INT32, DataType::FP16})
                      .build();
    const json j = split.architectureJson();
    EXPECT_EQ(j.at("layer_type").get<string>(), "split");
    EXPECT_EQ(j.at("version").get<string>(), "1.0.0");
    EXPECT_EQ(j.at("axis").get<uint32_t>(), 1U);
    EXPECT_EQ(j.at("sizes").get<vector<uint64_t>>(), (vector<uint64_t>{1, 5}));
    ASSERT_EQ(j.at("outputs").size(), 2U);
    EXPECT_EQ(j.at("outputs").at(0).at("data_type").get<DataType>(), DataType::INT32);
    EXPECT_EQ(j.at("outputs").at(1).at("data_type").get<DataType>(), DataType::FP16);

    ASSERT_NO_THROW(Split::deserialize(j, &network));
    auto restored = dynamic_pointer_cast<Split>(network.getLayer(network.getNumLayers() - 1));
    ASSERT_NE(restored, nullptr);
    EXPECT_EQ(restored->getSizes(), (vector<uint64_t>{1, 5}));
    ASSERT_EQ(restored->getFeatureOutputs().size(), 2U);
    EXPECT_EQ(restored->getFeatureOutputs()[0].getDataType(), DataType::INT32);
    EXPECT_EQ(restored->getFeatureOutputs()[1].getDimensions(), (vector<uint64_t>{4, 5}));
    // A deserialized piece is a new tensor that keeps the saved one's id as its original id; the input is the network's
    // own tensor.
    for (uint32_t i = 0; i < 2; ++i)
        EXPECT_EQ(restored->getFeatureOutputs()[i].getOriginalId(), j.at("outputs").at(i).at("id").get<uint64_t>());
    EXPECT_EQ(restored->getFeatureInputs()[0], x.getFeatureOutput().value());
    json again = restored->architectureJson();
    for (uint32_t i = 0; i < 2; ++i) again["outputs"][i]["id"] = j.at("outputs").at(i).at("id");
    EXPECT_EQ(again, j);

    json bad = j;
    bad["sizes"] = vector<uint64_t>{2, 5};
    EXPECT_THROW(Split::deserialize(bad, &network), std::runtime_error);
    bad = j;
    bad["outputs"].erase(1);
    EXPECT_THROW(Split::deserialize(bad, &network), std::runtime_error);
    bad = j;
    bad["outputs"][1]["dimensions"] = vector<uint64_t>{4, 4};
    EXPECT_THROW(Split::deserialize(bad, &network), std::runtime_error);
}

// Placed: the network connects the pieces to their consumers in its own order; each consumer must get its piece, in
// its type.
TEST(UtilityApiLayers, SplitPlacedSendsEachPieceToItsConsumer) {
    constexpr uint32_t batchSize = 4;
    Network network("split_placed");
    NetworkInput x = NetworkInput::Builder().network(network).name("x").dimensions({3, 6}).dataType(DataType::FP32).build();
    const vector<uint64_t> sizes{2, 3, 1};
    const vector<DataType> types{DataType::BF16, DataType::FP32, DataType::INT32};
    Split split = Split::Builder()
                      .network(network)
                      .featureInput(x.getFeatureOutput().value())
                      .axis(1)
                      .sizes(sizes)
                      .outputDataTypes(types)
                      .build();
    // Consumers declared in reverse order of the pieces.
    const vector<string> names{"a", "b", "c"};
    for (int i = 2; i >= 0; --i) {
        NetworkOutput::Builder()
            .network(network)
            .name(names[i])
            .inputTensor(split.getFeatureOutputs()[i])
            .dataType(types[i])
            .build();
    }

    vector<Event> initDoneEvents;
    shared_ptr<PlacedNetwork> placed = network.place(batchSize, initDoneEvents);
    ASSERT_NE(placed, nullptr);
    for (Event &event : initDoneEvents) event.synchronize();
    ThorImplementation::TensorPlacement cpuPlacement(ThorImplementation::TensorPlacement::MemDevices::CPU);
    ThorImplementation::Tensor inputCpu(cpuPlacement, ThorImplementation::TensorDescriptor(DataType::FP32, {batchSize, 3, 6}));
    // Exact in every type (the INT32 column holds whole numbers).
    auto value = [](uint64_t example, uint64_t row, uint64_t column) -> float {
        return column == 5 ? static_cast<float>(example * 100 + row * 10) - 3.0f
                           : static_cast<float>(example * 16 + row * 4 + column) * 0.5f - 7.0f;
    };
    float *inputValues = inputCpu.getMemPtr<float>();
    for (uint64_t e = 0; e < batchSize; ++e)
        for (uint64_t r = 0; r < 3; ++r)
            for (uint64_t c = 0; c < 6; ++c) inputValues[(e * 3 + r) * 6 + c] = value(e, r, c);
    map<string, ThorImplementation::Tensor> outputs;
    map<string, Event> outputReadyEvents;
    Event done = placed->submitBatch(0, {{"x", inputCpu}}, outputs, outputReadyEvents, /*isInferenceOnly=*/true);
    done.synchronize();

    const vector<uint64_t> firstColumn{0, 2, 5};
    for (uint32_t i = 0; i < 3; ++i) {
        ASSERT_TRUE(outputs.contains(names[i]));
        outputReadyEvents.at(names[i]).synchronize();
        const ThorImplementation::Tensor piece = outputs.at(names[i]);
        ASSERT_EQ(piece.getDescriptor().getDataType(), types[i]);
        ASSERT_EQ(piece.getDimensions(), (vector<uint64_t>{batchSize, 3, sizes[i]}));
        ThorImplementation::Tensor pieceCpu(cpuPlacement, piece.getDescriptor());
        Stream stream(0);
        pieceCpu.copyFromAsync(piece, stream);
        stream.synchronize();
        for (uint64_t e = 0; e < batchSize; ++e) {
            for (uint64_t r = 0; r < 3; ++r) {
                for (uint64_t c = 0; c < sizes[i]; ++c) {
                    const uint64_t index = (e * 3 + r) * sizes[i] + c;
                    const float expected = value(e, r, firstColumn[i] + c);
                    float observed = 0.0f;
                    if (types[i] == DataType::BF16)
                        observed = __bfloat162float(pieceCpu.getMemPtr<__nv_bfloat16>()[index]);
                    else if (types[i] == DataType::INT32)
                        observed = static_cast<float>(pieceCpu.getMemPtr<int32_t>()[index]);
                    else
                        observed = pieceCpu.getMemPtr<float>()[index];
                    EXPECT_EQ(observed, expected) << "piece " << i << " example " << e << " row " << r << " column " << c;
                }
            }
        }
    }
}

namespace {

RaggedTensor raggedHistory(Network &network, const string &name = "history") {
    return RaggedNetworkInput::Builder()
        .network(network)
        .name(name)
        .valuesDataType(DataType::FP32)
        .offsetsDataType(DataType::UINT32)
        .trailingDimensions({6, 3})
        .maxValuesPerRow(4)
        .build();
}

// Rows 0 and 3 of every placement are empty; the rest hold one or two values. The active prefix is short of capacity.
vector<uint64_t> raggedSplitOffsets(uint32_t batchSize) {
    vector<uint64_t> offsets(batchSize + 1, 0);
    for (uint32_t row = 0; row < batchSize; ++row)
        offsets[row + 1] = offsets[row] + (row == 0 || row % 3 == 0 ? 0 : 1 + row % 2);
    return offsets;
}

Impl::RowPartitionRuntime raggedSplitPartition(const RaggedTensor &input, uint32_t batchSize, const vector<uint64_t> &offsets) {
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor offsetData(cpu, input.getDescriptorForPlacement(batchSize).getOffsetsDescriptor());
    for (uint32_t i = 0; i <= batchSize; ++i) offsetData.getMemPtr<uint32_t>()[i] = static_cast<uint32_t>(offsets[i]);
    Impl::RowPartitionRuntime partition(offsetData, input.getDescriptorForPlacement(batchSize).getRowPartition());
    partition.setHostOffsets(offsets);
    return partition;
}

float hostValue(const Impl::Tensor &tensor, uint64_t index) {
    switch (tensor.getDescriptor().getDataType()) {
        case DataType::FP16:
            return __half2float(tensor.getMemPtr<half>()[index]);
        case DataType::INT32:
            return static_cast<float>(tensor.getMemPtr<int32_t>()[index]);
        default:
            return tensor.getMemPtr<float>()[index];
    }
}

}  // namespace

TEST(UtilityApiLayers, RaggedSplitBuildsPiecesThatShareThePartition) {
    Network network("ragged_split_builds");
    const RaggedTensor input = raggedHistory(network);
    Split split = Split::Builder()
                      .network(network)
                      .featureInput(input)
                      .axis(1)
                      .sizes({1, 2})
                      .outputDataTypes({DataType::BF16, DataType::FP32})
                      .build();
    ASSERT_TRUE(split.getUseRagged());
    const vector<RaggedTensor> pieces = split.getRaggedFeatureOutputs();
    ASSERT_EQ(pieces.size(), 2U);
    EXPECT_EQ(pieces[0].getValuesDimensions(), (vector<uint64_t>{4, 6, 1}));
    EXPECT_EQ(pieces[1].getValuesDimensions(), (vector<uint64_t>{4, 6, 2}));
    EXPECT_EQ(pieces[0].getValuesDataType(), DataType::BF16);
    EXPECT_EQ(pieces[1].getValuesDataType(), DataType::FP32);
    for (uint32_t i = 0; i < pieces.size(); ++i) {
        EXPECT_TRUE(pieces[i].sharesPartitionWith(input));
        EXPECT_EQ(pieces[i].getOffsets(), input.getOffsets());
        EXPECT_EQ(pieces[i].getMaxValuesPerRow(), 4U);
        EXPECT_EQ(pieces[i].getValues(), split.getFeatureOutputs()[i]);
        EXPECT_EQ(split.getConnectionType(pieces[i].getValues()), static_cast<int>(i));
    }
    ASSERT_EQ(split.getFeatureInputs().size(), 2U);
    EXPECT_EQ(split.getFeatureInputs()[0], input.getValues());
    EXPECT_EQ(split.getFeatureInputs()[1], input.getOffsets());
    EXPECT_EQ(split.getConnectionType(input.getValues()), 0);
    EXPECT_EQ(split.getConnectionType(input.getOffsets()), 1);
    EXPECT_EQ(split.getRaggedPartitionRequirementForInput(input.getOffsets()),
              ThorImplementation::RaggedPartitionRequirement::DEVICE_ACTIVE_COUNT);

    // The pieces are driven once the values and the partition are both connected.
    EXPECT_TRUE(split.mustConnectAllInputsToDriveOutput());
    split.resetGraphTraversalState();
    split.informThatInputConnectionMade(input.getOffsets());
    EXPECT_TRUE(split.getOutputsFromInput(input.getOffsets()).empty());
    split.informThatInputConnectionMade(input.getValues());
    EXPECT_EQ(split.getOutputsFromInput(input.getValues()), split.getFeatureOutputs());
    split.resetGraphTraversalState();

    // The axis counts the trailing dimensions.
    expectLogicError([&] { Split::Builder().network(network).featureInput(input).axis(2).sizes({1, 2}).build(); },
                     "out of range");
    expectLogicError([&] { Split::Builder().network(network).featureInput(input).axis(0).sizes({2, 3}).build(); },
                     "sum to 5");

    const json j = split.architectureJson();
    EXPECT_TRUE(j.at("use_ragged").get<bool>());
    EXPECT_FALSE(j.contains("inputs"));
    EXPECT_EQ(j.at("ragged_input").at("values").at("id").get<uint64_t>(), input.getValues().getOriginalId());
    ASSERT_EQ(j.at("ragged_outputs").size(), 2U);
    for (uint32_t i = 0; i < 2; ++i) {
        EXPECT_EQ(j.at("ragged_outputs").at(i).at("offsets").at("id"), j.at("ragged_input").at("offsets").at("id"));
        EXPECT_EQ(j.at("ragged_outputs").at(i).at("values").at("id"), j.at("outputs").at(i).at("id"));
    }
    // Logical M, not a placement's capacity.
    EXPECT_EQ(j.at("ragged_outputs").at(1).at("values").at("dimensions").get<vector<uint64_t>>(),
              (vector<uint64_t>{4, 6, 2}));

    ASSERT_NO_THROW(Split::deserialize(j, &network));
    auto restored = dynamic_pointer_cast<Split>(network.getLayer(network.getNumLayers() - 1));
    ASSERT_NE(restored, nullptr);
    ASSERT_TRUE(restored->getUseRagged());
    ASSERT_EQ(restored->getFeatureInputs().size(), 2U);
    EXPECT_EQ(restored->getFeatureInputs()[0], input.getValues());
    EXPECT_EQ(restored->getFeatureInputs()[1], input.getOffsets());
    const vector<RaggedTensor> restoredPieces = restored->getRaggedFeatureOutputs();
    ASSERT_EQ(restoredPieces.size(), 2U);
    for (uint32_t i = 0; i < 2; ++i) {
        EXPECT_TRUE(restoredPieces[i].sharesPartitionWith(input));
        EXPECT_EQ(restoredPieces[i].getValues().getOriginalId(), j.at("outputs").at(i).at("id").get<uint64_t>());
        EXPECT_EQ(restoredPieces[i].getValuesDataType(), pieces[i].getValuesDataType());
    }
    json again = restored->architectureJson();
    for (uint32_t i = 0; i < 2; ++i) {
        again["outputs"][i]["id"] = j.at("outputs").at(i).at("id");
        again["ragged_outputs"][i]["values"]["id"] = j.at("outputs").at(i).at("id");
    }
    EXPECT_EQ(again, j);

    json bad = j;
    bad["ragged_outputs"][1]["offsets"]["id"] = input.getValues().getOriginalId();
    EXPECT_ANY_THROW(Split::deserialize(bad, &network));
    bad = j;
    bad["outputs"][1]["dimensions"] = vector<uint64_t>{4, 6, 3};
    bad["ragged_outputs"][1]["values"]["dimensions"] = vector<uint64_t>{4, 6, 3};
    EXPECT_ANY_THROW(Split::deserialize(bad, &network));
    bad = j;
    bad["ragged_outputs"].erase(1);
    EXPECT_ANY_THROW(Split::deserialize(bad, &network));
    bad = j;
    bad["inputs"] = json::array({input.getValues().architectureJson()});
    EXPECT_ANY_THROW(Split::deserialize(bad, &network));
    bad = j;
    bad["ragged_input"].erase("layer_ragged_reference_version");
    EXPECT_ANY_THROW(Split::deserialize(bad, &network));
}

// Saved and loaded, then placed at two batch sizes: every piece holds its columns of the active packed values, in its
// type, and the input's host partition; the inactive capacity is never read.
TEST(UtilityApiLayers, RaggedSplitSavedForwardAtTwoAndSeven) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) GTEST_SKIP() << "CUDA device unavailable";

    const filesystem::path archive =
        filesystem::temp_directory_path() / ("ragged_split_forward_" + to_string(chrono::steady_clock::now().time_since_epoch().count()));
    filesystem::remove_all(archive);
    const vector<uint64_t> sizes{2, 1, 3};
    const vector<uint64_t> firstRow{0, 2, 3};
    const vector<DataType> types{DataType::FP32, DataType::FP16, DataType::FP32};
    const vector<string> names{"a", "b", "c"};
    uint64_t partitionId = 0;
    vector<uint64_t> pieceIds;
    {
        Network network("ragged_split_forward");
        const RaggedTensor input = raggedHistory(network);
        Split split = Split::Builder().network(network).featureInput(input).axis(0).sizes(sizes).outputDataTypes(types).build();
        const vector<RaggedTensor> pieces = split.getRaggedFeatureOutputs();
        ASSERT_EQ(pieces.size(), 3U);
        // Consumers declared in reverse order of the pieces.
        for (int i = 2; i >= 0; --i)
            (void)RaggedNetworkOutput::Builder().network(network).name(names[i]).inputTensor(pieces[i]).build();
        partitionId = input.getRowPartitionToken().getOriginalId();
        for (const Tensor &piece : split.getFeatureOutputs()) pieceIds.push_back(piece.getOriginalId());
        network.save(archive.string(), true);
    }

    // Exact in FP16: below 512, in quarters.
    auto value = [](uint64_t token, uint32_t row, uint32_t column) {
        return static_cast<float>(10 * row + column) + static_cast<float>(token) * 0.25f;
    };
    for (uint32_t b : {2U, 7U}) {
        SCOPED_TRACE(::testing::Message() << "placement B=" << b);
        Network network("ragged_split_forward");
        ASSERT_NO_THROW(network.load(archive.string()));
        const auto external = network.getExternalRaggedNetworkInputs();
        ASSERT_EQ(external.size(), 1U);
        const RaggedTensor input = external.front().raggedTensor;
        EXPECT_EQ(input.getRowPartitionToken().getOriginalId(), partitionId);
        shared_ptr<Split> layer;
        for (uint32_t i = 0; i < network.getNumLayers(); ++i)
            if (auto candidate = dynamic_pointer_cast<Split>(network.getLayer(i))) layer = candidate;
        ASSERT_NE(layer, nullptr);
        ASSERT_TRUE(layer->getUseRagged());
        ASSERT_EQ(layer->getFeatureOutputs().size(), 3U);
        for (uint32_t i = 0; i < 3; ++i) {
            EXPECT_EQ(layer->getFeatureOutputs()[i].getOriginalId(), pieceIds[i]);
            EXPECT_TRUE(layer->getRaggedFeatureOutputs()[i].sharesPartitionWith(input));
        }
        const json architectureBefore = network.architectureJson();

        vector<Event> initEvents;
        const auto placed = network.place(b, initEvents, /*inferenceOnly=*/true);
        ASSERT_NE(placed, nullptr);
        for (Event &event : initEvents) event.synchronize();
        auto &stamp = placed->getStampedNetwork(0);
        EXPECT_EQ(stamp.getManagedPartitionOffsetsInputForTest(input.getRowPartitionId()), nullptr);
        ASSERT_NE(stamp.getManagedPartitionActiveCountInputForTest(input.getRowPartitionId()), nullptr);
        auto physical = dynamic_pointer_cast<Impl::RaggedSplit>(stamp.getPhysicalLayerFromApiLayer(layer->getId()));
        ASSERT_NE(physical, nullptr);
        EXPECT_EQ(physical->getType(), "RaggedSplit");
        ASSERT_EQ(physical->getFeatureInputs().size(), 2U);
        ASSERT_TRUE(physical->getFeatureInputs()[1].has_value());
        EXPECT_EQ(physical->getFeatureInputs()[1]->getDimensions(), (vector<uint64_t>{1}));
        ASSERT_EQ(physical->getFeatureOutputs().size(), 3U);
        for (uint32_t i = 0; i < 3; ++i) {
            ASSERT_TRUE(physical->getFeatureOutputs()[i].has_value());
            EXPECT_EQ(physical->getFeatureOutputs()[i]->getDimensions(), (vector<uint64_t>{b * 4U, sizes[i], 3}));
            EXPECT_EQ(physical->getFeatureOutputs()[i]->getDataType(), types[i]);
        }
        EXPECT_EQ(network.architectureJson(), architectureBefore);

        const vector<uint64_t> offsets = raggedSplitOffsets(b);
        const uint64_t active = offsets.back();
        ASSERT_GT(active, 0U);
        ASSERT_LT(active, b * 4U);
        const Impl::RowPartitionRuntime partition = raggedSplitPartition(input, b, offsets);
        Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
        Impl::Tensor data(cpu, input.getDescriptorForPlacement(b).getValuesDescriptor());
        for (uint64_t t = 0; t < b * 4U; ++t)
            for (uint32_t row = 0; row < 6; ++row)
                for (uint32_t c = 0; c < 3; ++c)
                    data.getMemPtr<float>()[t * 18 + row * 3 + c] = t < active ? value(t, row, c) : numeric_limits<float>::quiet_NaN();
        Batch batch;
        batch.insert("history", Impl::RaggedTensor(data, partition));
        const auto outputs = placed->inferLogical(batch);
        for (uint32_t i = 0; i < 3; ++i) {
            SCOPED_TRACE(::testing::Message() << "piece " << names[i]);
            ASSERT_TRUE(outputs.contains(names[i]));
            ASSERT_TRUE(holds_alternative<Impl::RaggedTensor>(outputs.at(names[i])));
            const auto piece = get<Impl::RaggedTensor>(outputs.at(names[i]));
            const auto hostOffsets = piece.getHostOffsetsIfAvailable();
            ASSERT_TRUE(hostOffsets.has_value());
            EXPECT_EQ(hostOffsets.value(), offsets);
            ASSERT_EQ(piece.getValues().getDimensions(), (vector<uint64_t>{b * 4U, sizes[i], 3}));
            ASSERT_EQ(piece.getValues().getDescriptor().getDataType(), types[i]);
            for (uint64_t t = 0; t < active; ++t)
                for (uint32_t row = 0; row < sizes[i]; ++row)
                    for (uint32_t c = 0; c < 3; ++c)
                        EXPECT_EQ(hostValue(piece.getValues(), (t * sizes[i] + row) * 3 + c), value(t, firstRow[i] + row, c))
                            << "token " << t << " row " << row << " column " << c;
        }
    }
    filesystem::remove_all(archive);
}

namespace {

Impl::DynamicExpression splitSquaredError(bool gradient) {
    const auto predictions = Impl::Expression::input("predictions", DataType::FP32, DataType::FP32);
    const auto labels = Impl::Expression::input("labels", DataType::FP32, DataType::FP32);
    const auto difference = predictions - labels;
    auto result = gradient ? difference * Impl::Expression(2.0 * Impl::Loss::getLossScalingFactor()) : difference * difference;
    result = result.withOutputDType(DataType::FP32);
    return Impl::DynamicExpression::fromExpressionDefinition(Impl::ExpressionDefinition::fromOutputs(
        Impl::Expression::outputs({{gradient ? "predictions_grad" : "loss", result}})));
}

vector<float> downloadActive(const Impl::Tensor &source, uint64_t elements) {
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor host = source.clone(cpu);
    Impl::Tensor deviceHandle = source;
    Stream stream = Stream::getNextDownloadStream(source.getPlacement().getDeviceNum());
    host.downloadSection(deviceHandle, stream, 0, 0, elements * sizeof(float));
    stream.synchronize();
    return vector<float>(host.getMemPtr<float>(), host.getMemPtr<float>() + elements);
}

}  // namespace

// Split, then concatenate the pieces back and take a squared error: the input's gradient is the loss gradient on the
// rows of every floating-point piece, and zero on the rows of the INT32 piece, which takes none.
TEST(UtilityApiLayers, RaggedSplitBackpropagatesEachPieceAndZerosForAnIntegralPiece) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) GTEST_SKIP() << "CUDA device unavailable";
    constexpr uint32_t b = 5;
    constexpr uint64_t capacity = b * 4;

    Network network("ragged_split_backward");
    const RaggedTensor input = raggedHistory(network);
    const RaggedTensor targets = RaggedNetworkInput::Builder()
                                     .network(network)
                                     .name("targets")
                                     .partition(input)
                                     .valuesDataType(DataType::FP32)
                                     .trailingDimensions({6, 3})
                                     .build();
    const auto rivet = GradientRivet::Builder().network(network).tensor(input.getValues()).build();
    ASSERT_TRUE(rivet.getFeatureOutput().has_value());
    const RaggedTensor riveted = input.withValues(rivet.getFeatureOutput().value());
    Split split = Split::Builder()
                      .network(network)
                      .featureInput(riveted)
                      .axis(0)
                      .sizes({2, 1, 3})
                      .outputDataTypes({DataType::FP32, DataType::INT32, DataType::FP32})
                      .build();
    const vector<RaggedTensor> pieces = split.getRaggedFeatureOutputs();
    const Concatenate joined = Concatenate::Builder()
                                   .network(network)
                                   .featureInput(pieces[0])
                                   .featureInput(pieces[1])
                                   .featureInput(pieces[2])
                                   .concatenationAxis(0)
                                   .outputDataType(DataType::FP32)
                                   .build();
    ASSERT_TRUE(joined.getRaggedFeatureOutput().has_value());
    const auto loss = RaggedCustomLoss::Builder()
                          .network(network)
                          .predictions(joined.getRaggedFeatureOutput().value())
                          .labels(targets)
                          .lossExpression(splitSquaredError(false))
                          .gradientExpression(splitSquaredError(true))
                          .build();
    (void)NetworkOutput::Builder().network(network).name("raw_loss").inputTensor(loss.getRaggedRawLoss().getValues()).dataType(DataType::FP32).build();

    vector<Event> initialization;
    auto placed = network.place(b, initialization, /*inferenceOnly=*/false);
    ASSERT_NE(placed, nullptr);
    for (auto &event : initialization) event.synchronize();
    auto &stamp = placed->getStampedNetwork(0);
    auto physical = dynamic_pointer_cast<Impl::RaggedSplit>(stamp.getPhysicalLayerFromApiLayer(split.getId()));
    auto physicalLoss = dynamic_pointer_cast<Impl::RaggedCustomLoss>(stamp.getPhysicalLayerFromApiLayer(loss.getId()));
    ASSERT_NE(physical, nullptr);
    ASSERT_NE(physicalLoss, nullptr);
    const auto gradients = physical->getErrorOutputs();
    ASSERT_EQ(gradients.size(), 2U);
    ASSERT_TRUE(gradients[0].has_value());
    EXPECT_EQ(gradients[0]->getDimensions(), (vector<uint64_t>{capacity, 6, 3}));
    EXPECT_FALSE(gradients[1].has_value()) << "the partition carrier takes no gradient";

    const vector<uint64_t> offsets = raggedSplitOffsets(b);
    const uint64_t active = offsets.back();
    ASSERT_GT(active, 0U);
    ASSERT_LT(active, capacity);
    const Impl::RowPartitionRuntime partition = raggedSplitPartition(input, b, offsets);
    // Row 2 is the INT32 piece: whole numbers, so its conversion is exact.
    auto value = [](uint64_t token, uint32_t row, uint32_t column) {
        if (row == 2) return static_cast<float>(token + 3 * column) - 4.0f;
        return 0.16f * static_cast<float>(token) + 0.13f * static_cast<float>(row) - 0.23f * static_cast<float>(column) + 0.5f;
    };
    auto difference = [](uint64_t token, uint32_t row, uint32_t column) {
        const float magnitude = 0.11f + 0.07f * static_cast<float>((token + 2 * row) % 5) + 0.035f * static_cast<float>(row) +
                                0.055f * static_cast<float>(column);
        return ((token + row + column) % 2 == 0 ? 1.0f : -1.0f) * magnitude;
    };
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor data(cpu, input.getDescriptorForPlacement(b).getValuesDescriptor());
    Impl::Tensor labels(cpu, targets.getDescriptorForPlacement(b).getValuesDescriptor());
    for (uint64_t t = 0; t < capacity; ++t)
        for (uint32_t row = 0; row < 6; ++row)
            for (uint32_t c = 0; c < 3; ++c) {
                const uint64_t index = t * 18 + row * 3 + c;
                data.getMemPtr<float>()[index] = t < active ? value(t, row, c) : numeric_limits<float>::quiet_NaN();
                labels.getMemPtr<float>()[index] =
                    t < active ? value(t, row, c) - difference(t, row, c) : numeric_limits<float>::quiet_NaN();
            }
    Batch batch;
    batch.insert("history", Impl::RaggedTensor(data, partition));
    batch.insert("targets", labels);
    batch.setValidExampleCount(b);
    map<string, Impl::Tensor> results;
    map<string, Event> ready;
    auto done = placed->submitBatch(0, batch, results, ready, false);
    done.synchronize();
    ASSERT_TRUE(ready.contains("raw_loss"));
    ready.at("raw_loss").synchronize();
    placed->synchronize();

    ASSERT_TRUE(physicalLoss->getErrorOutput().has_value());
    const auto upstream = downloadActive(physicalLoss->getErrorOutput().value(), active * 18);
    const auto gradient = downloadActive(gradients[0].value(), active * 18);
    for (uint64_t t = 0; t < active; ++t)
        for (uint32_t row = 0; row < 6; ++row)
            for (uint32_t c = 0; c < 3; ++c) {
                const uint64_t index = t * 18 + row * 3 + c;
                if (row == 2) {
                    EXPECT_EQ(gradient[index], 0.0f) << "INT32 piece, token " << t << " column " << c;
                    continue;
                }
                const float expected = 2.0f * difference(t, row, c) * Impl::Loss::getLossScalingFactor();
                EXPECT_NEAR(upstream[index], expected, 5e-3f) << "upstream at " << index;
                EXPECT_NEAR(gradient[index], upstream[index], 6e-3f) << "token " << t << " row " << row << " column " << c;
            }
}
