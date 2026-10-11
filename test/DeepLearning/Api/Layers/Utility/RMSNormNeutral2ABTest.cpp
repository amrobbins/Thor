#include "DeepLearning/Api/Layers/Utility/RMSNorm.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Layers/Loss/RaggedCustomLoss.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Optimizers/Sgd.h"
#include "test/DeepLearning/Api/Helpers/GradientRivet.h"
#include "DeepLearning/Implementation/Layers/NeuralNetwork/RaggedRMSNorm.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Common/Stream.h"
#include "gtest/gtest.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <limits>
#include <map>
#include <vector>

using namespace Thor;
using namespace std;
using json = nlohmann::json;
namespace Impl = ThorImplementation;

namespace {
constexpr uint64_t M2AB = 4, H2AB = 3;
constexpr float EPS2AB = 1e-5f;

RaggedTensor input2AB(Network& network) {
    return RaggedNetworkInput::Builder().network(network).name("tokens")
        .valuesDataType(DataType::FP32).offsetsDataType(DataType::UINT64)
        .trailingDimensions({H2AB}).maxValuesPerRow(M2AB).build();
}
RaggedTensor auxiliary2AB(Network& network, const RaggedTensor& feature) {
    return RaggedNetworkInput::Builder().network(network).name("aux")
        .valuesDataType(DataType::FP32).trailingDimensions({H2AB})
        .partition(feature).build();
}
RMSNorm layer2AB(Network& network, const RaggedTensor& feature, const RaggedTensor& auxiliary,
                 bool auxiliaryFirst = false) {
    RMSNorm::Builder builder;
    builder.network(network).normalizedShape({H2AB}).epsilon(EPS2AB)
        .parameterDataType(DataType::FP32);
    if (auxiliaryFirst) builder.epilogueInput("auxiliary", auxiliary);
    builder.featureInput(feature);
    if (!auxiliaryFirst) builder.epilogueInput("auxiliary", auxiliary);
    builder.epilogue(RMSNorm::epilogueInput() + RMSNorm::epilogueAuxInput("auxiliary") * Impl::Expression(0.5));
    return builder.build();
}
shared_ptr<RMSNorm> find2AB(Network& network) {
    for (uint32_t i = 0; i < network.getNumLayers(); ++i) {
        auto rms = dynamic_pointer_cast<RMSNorm>(network.getLayer(i));
        if (rms) return rms;
    }
    return {};
}
filesystem::path archive2AB(const string& prefix) {
    return filesystem::temp_directory_path() /
        (prefix + "_" + to_string(chrono::steady_clock::now().time_since_epoch().count()));
}
vector<uint64_t> offsets2AB(uint32_t b) {
    vector<uint64_t> ends(b + 1, 0);
    for (uint32_t r = 0; r < b; ++r)
        ends[r + 1] = ends[r] + (r % 3 == 0 ? 0u : r % 3 == 1 ? 2u : 1u);
    return ends;
}
float x2AB(uint64_t t, uint32_t c) {
    return 0.25f + 0.13f * static_cast<float>(t) + 0.15f * static_cast<float>(c);
}
float a2AB(uint64_t t, uint32_t c) {
    return 0.19f + 0.08f * static_cast<float>(t) - 0.07f * static_cast<float>(c);
}
Batch batch2AB(const RaggedTensor& input, uint32_t b, bool labels = false) {
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    // Build physical values and offsets from the same runtime partition; the API
    // tensor shape is deliberately never used as the physical B*M geometry.
    const auto descriptor = input.getDescriptorForPlacement(b);
    const auto valuesDescriptor = descriptor.getValuesDescriptor();
    const auto offsetsDescriptor = descriptor.getOffsetsDescriptor();
    Impl::Tensor x(cpu, valuesDescriptor), aux(cpu, valuesDescriptor), offsets(cpu, offsetsDescriptor);
    const auto ends = offsets2AB(b);
    for (uint64_t t = 0; t < b * M2AB; ++t) {
        for (uint32_t c = 0; c < H2AB; ++c) {
            x.getMemPtr<float>()[t * H2AB + c] = t < ends.back() ? x2AB(t,c) : numeric_limits<float>::quiet_NaN();
            aux.getMemPtr<float>()[t * H2AB + c] = t < ends.back() ? a2AB(t,c) : numeric_limits<float>::quiet_NaN();
        }
    }
    for (uint32_t i = 0; i <= b; ++i) offsets.getMemPtr<uint64_t>()[i] = ends[i];
    // Get the canonical runtime row-partition descriptor from a neutral input.
    // Both independent values connectors use that *same* partition.
    Impl::RowPartitionRuntime partition(offsets, descriptor.getRowPartition());
    partition.setHostOffsets(ends);
    Batch result;
    result.insert("tokens", Impl::RaggedTensor(x, partition));
    result.insert("aux", Impl::RaggedTensor(aux, partition));
    if (labels) {
        Impl::Tensor truth(cpu, valuesDescriptor);
        for (uint64_t i = 0; i < b * M2AB * H2AB; ++i) truth.getMemPtr<float>()[i] = 0.f;
        result.insert("labels", Impl::RaggedTensor(truth, partition));
    }
    result.setValidExampleCount(b);
    return result;
}
vector<float> download2AB(Impl::Tensor value) {
    const uint64_t bytes = value.getArraySizeInBytes();
    EXPECT_EQ(bytes % sizeof(float), 0u);
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor host = value.clone(cpu);
    Stream stream = Stream::getNextDownloadStream(value.getPlacement().getDeviceNum());
    host.downloadSection(value, stream, 0, 0, bytes);
    stream.synchronize();
    return vector<float>(host.getMemPtr<float>(), host.getMemPtr<float>() + bytes / sizeof(float));
}
float tokenLoss2AB(const vector<float>& x, const vector<float>& aux, const vector<float>& weights) {
    float sum = 0.f;
    for (float v : x) sum += v * v;
    const float invRms = 1.f / sqrtf(sum / static_cast<float>(H2AB) + EPS2AB);
    float loss = 0.f;
    for (uint32_t c = 0; c < H2AB; ++c) {
        const float y = x[c] * invRms * weights[c] + 0.5f * aux[c];
        loss += y*y;
    }
    return loss;
}
} // namespace

TEST(RaggedRMSNormNeutral2AB, StrictAuxiliaryReferencesPartitionAndLegacyCompatibility) {
    Network network("rms_2ab_semantics");
    const auto feature = input2AB(network);
    const auto aux = auxiliary2AB(network, feature);
    const auto rms = layer2AB(network, feature, aux, true);
    ASSERT_TRUE(rms.getRaggedFeatureOutput().has_value());
    EXPECT_TRUE(rms.getRaggedFeatureOutput()->sharesPartitionWith(feature));
    const json record = rms.architectureJson();
    EXPECT_EQ(record.at("version"), "1.1.0");
    EXPECT_EQ(record.at("epilogue_inputs").size(), 1u);
    const json& binding = record.at("epilogue_inputs").at(0);
    EXPECT_EQ(binding.at("name"), "auxiliary");
    const json& ragged = binding.at("ragged_tensor");
    EXPECT_FALSE(ragged.contains("geometry_mode"));
    EXPECT_FALSE(ragged.contains("batch_size"));
    EXPECT_FALSE(ragged.contains("max_total_values"));
    EXPECT_EQ(ragged.at("max_values_per_row"), M2AB);
    EXPECT_EQ(ragged.at("offsets").at("dimensions"), (vector<uint64_t>{1}));
    EXPECT_EQ(ragged.at("offsets").at("id"), record.at("ragged_inputs").at(0).at("offsets").at("id"));
    for (uint32_t b : {2u,7u})
        EXPECT_EQ(rms.getOutputTensorBytes(b), b*M2AB*H2AB*sizeof(float));
    shared_ptr<thor_file::TarReader> reader;
    auto reject = [&](json bad) { EXPECT_ANY_THROW(RMSNorm::deserialize(reader,bad,&network)); };
    json bad = record; bad["version"] = "1.4.0";
    EXPECT_NO_THROW(RMSNorm::deserialize(reader,bad,&network)); // Archived canonical auxiliary grammar.
    bad = record; bad["version"] = "1.0.0"; reject(bad);
    bad=record; bad["version"] = "1.3.0"; reject(bad);
    bad=record; bad["epilogue"] = nullptr; reject(bad);
    bad=record; bad["epilogue_inputs"] = json::array(); reject(bad);
    bad=record; bad["epilogue_inputs"][0]["name"]="__internal"; reject(bad);
    bad=record; bad["epilogue_inputs"][0]["tensor"]["data_type"]=DataType::BF16; reject(bad);
    bad=record; bad["epilogue_inputs"][0]["ragged_tensor"]["values"]["id"]=feature.getValues().getOriginalId(); reject(bad);
    bad=record; bad["epilogue_inputs"][0]["ragged_tensor"]["offsets"]["id"]=feature.getValues().getOriginalId(); reject(bad);
    bad=record; bad["epilogue_inputs"][0]["ragged_tensor"]["max_values_per_row"]=M2AB+1; reject(bad);
    bad=record; bad["epilogue_inputs"][0]["ragged_tensor"]["batch_size"]=2; reject(bad);
    bad=record; bad["epilogue_inputs"][0]["ragged_tensor"]["max_total_values"]=8; reject(bad);
    bad=record; bad["epilogue_inputs"][0]["ragged_tensor"]["values"]["dimensions"]=vector<uint64_t>{M2AB,4}; reject(bad);
    bad=record; bad["ragged_outputs"][0]["offsets"]["id"]=aux.getValues().getOriginalId(); reject(bad);
    // Historical packed references may carry obsolete placement B and total
    // capacity. Neither is authoritative model geometry: semantic M comes
    // from the already-reconstructed graph source, even for version 1.1.0.
    for (uint32_t archivedB : {2u, 7u}) {
        SCOPED_TRACE(::testing::Message() << "archived B=" << archivedB);
        json historical = record;
        historical["batch_size"] = archivedB;
        historical["max_total_values"] = archivedB * M2AB;
        auto packHistoricalReference = [&](json& reference) {
            reference.erase("layer_ragged_reference_version");
            reference.erase("geometry_mode");
            reference["batch_size"] = archivedB;
            reference["max_total_values"] = archivedB * M2AB;
            reference["values"]["dimensions"][0] = archivedB * M2AB;
            reference["offsets"]["dimensions"][0] = archivedB + 1;
        };
        packHistoricalReference(historical["ragged_inputs"][0]);
        packHistoricalReference(historical["ragged_outputs"][0]);
        packHistoricalReference(historical["epilogue_inputs"][0]["ragged_tensor"]);
        historical["inputs"][0]["dimensions"][0] = archivedB * M2AB;
        historical["outputs"][0]["dimensions"][0] = archivedB * M2AB;
        historical["epilogue_inputs"][0]["tensor"]["dimensions"][0] = archivedB * M2AB;
        const uint32_t layersBefore = network.getNumLayers();
        ASSERT_NO_THROW(RMSNorm::deserialize(reader, historical, &network));
        ASSERT_EQ(network.getNumLayers(), layersBefore + 1);
        auto restored = dynamic_pointer_cast<RMSNorm>(network.getLayer(layersBefore));
        ASSERT_NE(restored, nullptr);
        ASSERT_TRUE(restored->getRaggedFeatureInput().has_value());
        ASSERT_TRUE(restored->getRaggedFeatureOutput().has_value());
        EXPECT_EQ(restored->getRaggedFeatureInput()->getValuesDimensions(),
                  (vector<uint64_t>{M2AB, H2AB}));
        EXPECT_EQ(restored->getRaggedFeatureOutput()->getValuesDimensions(),
                  (vector<uint64_t>{M2AB, H2AB}));
        EXPECT_TRUE(restored->getRaggedFeatureInput()->sharesPartitionWith(feature));
        // Both runtime placements are valid regardless of the archived B.
        for (uint32_t placementB : {2u, 7u})
            EXPECT_EQ(restored->getOutputTensorBytes(placementB),
                      placementB * M2AB * H2AB * sizeof(float));
    }
    const auto different = RaggedNetworkInput::Builder().network(network).name("different_partition")
        .valuesDataType(DataType::FP32).offsetsDataType(DataType::UINT64)
        .trailingDimensions({H2AB}).maxValuesPerRow(M2AB).build();
    EXPECT_ANY_THROW((void)layer2AB(network,feature,different));
    bad=record;
    bad["epilogue_inputs"][0]["ragged_tensor"]["offsets"]["id"] = different.getOffsets().getOriginalId();
    reject(bad);
    // Canonical logical construction preserves the established layer schema version.
    Network legacy("rms_2ab_legacy");
    const auto fixed = RaggedNetworkInput::Builder().network(legacy).name("fixed")
        .valuesDataType(DataType::FP32).trailingDimensions({H2AB})
        .maxValuesPerRow(M2AB).build();
    const auto fixedAux = RaggedNetworkInput::Builder().network(legacy).name("fixed_aux")
        .valuesDataType(DataType::FP32).trailingDimensions({H2AB}).partition(fixed).build();
    EXPECT_EQ(layer2AB(legacy,fixed,fixedAux).architectureJson().at("version"), "1.1.0");
}

TEST(RaggedRMSNormNeutral2AB, SavedInferenceB2B7WithNontrivialAuxiliaryAndEmptyRows) {
    int devices=0;
    if (cudaGetDeviceCount(&devices)!=cudaSuccess || devices==0) GTEST_SKIP()<<"CUDA device unavailable";
    const auto archive=archive2AB("rms_2ab_inference");
    {
        Network network("rms_2ab_inference");
        const auto feature=input2AB(network), aux=auxiliary2AB(network,feature);
        const auto rms=layer2AB(network,feature,aux);
        (void)RaggedNetworkOutput::Builder().network(network).name("normalized")
            .inputTensor(rms.getRaggedFeatureOutput().value()).build();
        // The auxiliary also has an independent reporting consumer. It must not
        // change which canonical partition reaches the RMSNorm epilogue.
        (void)RaggedNetworkOutput::Builder().network(network).name("aux_echo")
            .inputTensor(aux).build();
        network.save(archive.string(),true);
    }
    for (uint32_t b : {2u,7u}) {
        SCOPED_TRACE(::testing::Message()<<"B="<<b);
        Network network("rms_2ab_inference");
        ASSERT_NO_THROW(network.load(archive.string()));
        auto rms=find2AB(network);
        ASSERT_NE(rms,nullptr);
        const auto before=network.architectureJson();
        vector<Event> events;
        auto placed=network.place(b,events,true);
        ASSERT_NE(placed,nullptr);
        for (auto& e:events) e.synchronize();
        auto physical=dynamic_pointer_cast<Impl::RaggedRMSNorm>(
            placed->getStampedNetwork(0).getPhysicalLayerFromApiLayer(rms->getId()));
        ASSERT_NE(physical,nullptr);
        ASSERT_TRUE(physical->getFeatureOutput().has_value());
        EXPECT_EQ(physical->getFeatureOutput()->getDimensions(),(vector<uint64_t>{b*M2AB,H2AB}));
        const auto weights=download2AB(physical->getParameterStorage("weights"));
        const auto inputs=network.getExternalRaggedNetworkInputs();
        auto tokens=find_if(inputs.begin(),inputs.end(),[](const auto& ref){ return ref.name=="tokens"; });
        ASSERT_NE(tokens,inputs.end());
        const auto outputs=placed->inferLogical(batch2AB(tokens->raggedTensor,b));
        ASSERT_TRUE(outputs.contains("normalized"));
        ASSERT_TRUE(outputs.contains("aux_echo"));
        ASSERT_TRUE(holds_alternative<Impl::RaggedTensor>(outputs.at("normalized")));
        ASSERT_TRUE(holds_alternative<Impl::RaggedTensor>(outputs.at("aux_echo")));
        const auto result=get<Impl::RaggedTensor>(outputs.at("normalized"));
        const auto hostOffsets=result.getHostOffsetsIfAvailable();
        ASSERT_TRUE(hostOffsets.has_value());
        EXPECT_EQ(hostOffsets.value(),offsets2AB(b));
        const float* actual=result.getValues().getMemPtr<float>();
        const auto auxEcho=get<Impl::RaggedTensor>(outputs.at("aux_echo"));
        EXPECT_EQ(auxEcho.getHostOffsetsIfAvailable(),hostOffsets);
        const float* echoed=auxEcho.getValues().getMemPtr<float>();
        for (uint64_t t=0; t<hostOffsets->back(); ++t) {
            for (uint32_t c=0;c<H2AB;++c)
                EXPECT_NEAR(echoed[t*H2AB+c],a2AB(t,c),1e-5f);
            vector<float> x(H2AB), aux(H2AB);
            for (uint32_t c=0;c<H2AB;++c) { x[c]=x2AB(t,c); aux[c]=a2AB(t,c); }
            float squared=0.f;
            for (float v:x) squared+=v*v;
            const float denom=sqrtf(squared/H2AB+EPS2AB);
            for (uint32_t c=0;c<H2AB;++c)
                EXPECT_NEAR(actual[t*H2AB+c],x[c]*weights[c]/denom+0.5f*aux[c],2e-3f);
        }
        EXPECT_EQ(network.architectureJson(),before);
    }
    filesystem::remove_all(archive);
}

TEST(RaggedRMSNormNeutral2AB, SavedTrainingB2B7RoutesBothGradientsAndUpdatesWeights) {
    int devices=0;
    if (cudaGetDeviceCount(&devices)!=cudaSuccess || devices==0) GTEST_SKIP()<<"CUDA device unavailable";
    const auto archive=archive2AB("rms_2ab_training");
    {
        Network network("rms_2ab_training");
        (void)Sgd::Builder().network(network).initialLearningRate(.001f).decay(0.f).momentum(0.f).build();
        const auto feature=input2AB(network), aux=auxiliary2AB(network,feature);
        const auto labels=RaggedNetworkInput::Builder().network(network).name("labels")
            .valuesDataType(DataType::FP32).trailingDimensions({H2AB}).partition(feature).build();
        const auto featureRivet=GradientRivet::Builder().network(network).tensor(feature.getValues()).build();
        const auto auxRivet=GradientRivet::Builder().network(network).tensor(aux.getValues()).build();
        const auto rms=layer2AB(network,
            feature.withValues(featureRivet.getFeatureOutput().value()),
            aux.withValues(auxRivet.getFeatureOutput().value()),true);
        const auto prediction=Impl::Expression::input("predictions",DataType::FP32,DataType::FP32);
        const auto truth=Impl::Expression::input("labels",DataType::FP32,DataType::FP32);
        const auto diff=prediction-truth;
        auto loss=Impl::DynamicExpression::fromExpressionDefinition(
            Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({{
                "loss",(diff*diff).withOutputDType(DataType::FP32)}})));
        auto gradient=Impl::DynamicExpression::fromExpressionDefinition(
            Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({{
                "predictions_grad",(diff*Impl::Expression(2.0*Impl::Loss::getLossScalingFactor()))
                    .withOutputDType(DataType::FP32)}})));
        const auto raw=RaggedCustomLoss::Builder().network(network)
            .predictions(rms.getRaggedFeatureOutput().value()).labels(labels)
            .lossExpression(loss).gradientExpression(gradient).build();
        (void)NetworkOutput::Builder().network(network).name("raw_loss")
            .inputTensor(raw.getRaggedRawLoss().getValues()).dataType(DataType::FP32).build();
        network.save(archive.string(),true);
    }
    for (uint32_t b : {2u,7u}) {
        SCOPED_TRACE(::testing::Message()<<"B="<<b);
        Network network("rms_2ab_training");
        ASSERT_NO_THROW(network.load(archive.string()));
        const auto before=network.architectureJson();
        auto rms=find2AB(network);
        ASSERT_NE(rms,nullptr);
        vector<Event> events;
        auto placed=network.place(b,events,false);
        ASSERT_NE(placed,nullptr);
        for (auto& e:events) e.synchronize();
        auto physical=dynamic_pointer_cast<Impl::RaggedRMSNorm>(
            placed->getStampedNetwork(0).getPhysicalLayerFromApiLayer(rms->getId()));
        ASSERT_NE(physical,nullptr);
        const auto oldWeights=download2AB(physical->getParameterStorage("weights"));
        const auto inputs=network.getExternalRaggedNetworkInputs();
        auto tokens=find_if(inputs.begin(),inputs.end(),[](const auto& ref){ return ref.name=="tokens"; });
        ASSERT_NE(tokens,inputs.end());
        map<string,Impl::Tensor> outputs;
        map<string,Event> ready;
        auto finished=placed->submitBatch(0,batch2AB(tokens->raggedTensor,b,true),outputs,ready,false);
        finished.synchronize();
        placed->synchronize();
        ASSERT_TRUE(outputs.contains("raw_loss"));
        const auto& errors=physical->getErrorOutputs();
        ASSERT_GE(errors.size(),3u);
        ASSERT_TRUE(errors[0].has_value());
        ASSERT_TRUE(errors[2].has_value());
        const auto dx=download2AB(errors[0].value());
        const auto dAux=download2AB(errors[2].value());
        const auto newWeights=download2AB(physical->getParameterStorage("weights"));
        const auto lossScale=Impl::Loss::getLossScalingFactor();
        vector<float> dWeights(H2AB,0.f);
        for (uint64_t t=0;t<offsets2AB(b).back();++t) {
            vector<float> x(H2AB),aux(H2AB);
            for (uint32_t c=0;c<H2AB;++c) {x[c]=x2AB(t,c);aux[c]=a2AB(t,c);}
            float sq=0.f;
            for (float v:x) sq+=v*v;
            const float invRms=1.f/sqrtf(sq/H2AB+EPS2AB);
            for (uint32_t c=0;c<H2AB;++c) {
                const float y=x[c]*oldWeights[c]*invRms+0.5f*aux[c];
                EXPECT_NEAR(dAux.at(t*H2AB+c)/lossScale, y,2e-2f);
                dWeights[c]+=2.f*y*x[c]*invRms;
                const float step=1e-3f;
                auto plus=x,minus=x;
                plus[c]+=step;minus[c]-=step;
                const float expected=(tokenLoss2AB(plus,aux,oldWeights)-
                    tokenLoss2AB(minus,aux,oldWeights))/(2.f*step);
                ASSERT_GT(dx.size(),t*H2AB+c);
                EXPECT_NEAR(dx[t*H2AB+c]/lossScale,expected,3e-2f);
            }
        }
        bool changed=false;
        for (uint32_t c=0;c<H2AB;++c) {
            const float delta=newWeights[c]-oldWeights[c];
            changed|=fabs(delta)>1e-8f;
            if (fabs(dWeights[c])>1e-4f) EXPECT_LE(delta*dWeights[c],1e-7f);
        }
        EXPECT_TRUE(changed);
        EXPECT_EQ(network.architectureJson(),before);
    }
    filesystem::remove_all(archive);
}
