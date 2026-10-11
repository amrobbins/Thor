#include "DeepLearning/Api/Layers/Learning/FullyConnected.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedPrimitiveCommon.h"
#include "DeepLearning/Api/Layers/Utility/RMSNorm.h"
#include "DeepLearning/Api/Layers/Activations/Sigmoid.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Layers/Loss/RaggedCustomLoss.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Optimizers/Sgd.h"
#include "test/DeepLearning/Api/Helpers/GradientRivet.h"
#include "DeepLearning/Implementation/Layers/NeuralNetwork/RaggedFullyConnected.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/TarFile/TarWriter.h"
#include "gtest/gtest.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cmath>
#include <filesystem>
#include <map>
#include <memory>
#include <limits>
#include <vector>

using namespace Thor;
using namespace std;
using json = nlohmann::json;
namespace Impl = ThorImplementation;

namespace {
constexpr uint64_t kM1X=4, kIn1X=3, kOut1X=2;
RaggedTensor tokens1X(Network& network) {
    return RaggedNetworkInput::Builder().network(network).name("tokens")
        .valuesDataType(DataType::FP32).offsetsDataType(DataType::UINT64)
        .trailingDimensions({kIn1X}).maxValuesPerRow(kM1X).build();
}
FullyConnected build1X(Network& network, const RaggedTensor& input, bool bias) {
    return FullyConnected::Builder().network(network).featureInput(input)
        .numOutputFeatures(kOut1X).hasBias(bias).weightsDataType(DataType::FP32)
        .computeDataType(DataType::FP32).outputDataType(DataType::FP32)
        .noActivation().build();
}
shared_ptr<FullyConnected> find1X(Network& network) {
    for (uint32_t i=0; i<network.getNumLayers(); ++i) {
        auto fc = dynamic_pointer_cast<FullyConnected>(network.getLayer(i));
        if (fc) return fc;
    }
    return {};
}
filesystem::path archive1X(const string& prefix) {
    return filesystem::temp_directory_path() /
        (prefix + "_" + to_string(chrono::steady_clock::now().time_since_epoch().count()));
}
vector<uint64_t> offsets1X(uint32_t b) {
    vector<uint64_t> result(b+1,0);
    for (uint32_t r=0; r<b; ++r) result[r+1]=result[r]+(r%3==0?0u:r%3==1?2u:1u);
    return result;
}
float token1X(uint64_t t, uint32_t c) {
    return .25f+.11f*static_cast<float>(t)+.2f*static_cast<float>(c);
}
Batch batch1X(const RaggedTensor& input, uint32_t b, bool withLabels=false) {
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    const auto descriptor=input.getDescriptorForPlacement(b);
    const auto offsets=offsets1X(b);
    Impl::Tensor values(cpu, descriptor.getValuesDescriptor());
    Impl::Tensor boundaries(cpu, descriptor.getOffsetsDescriptor());
    for (uint64_t t=0; t<b*kM1X; ++t)
        for (uint32_t c=0; c<kIn1X; ++c)
            values.getMemPtr<float>()[t*kIn1X+c] =
                t<offsets.back()?token1X(t,c):numeric_limits<float>::quiet_NaN();
    for (uint32_t i=0; i<=b; ++i) boundaries.getMemPtr<uint64_t>()[i]=offsets[i];
    Impl::RowPartitionRuntime runtime(boundaries,descriptor.getRowPartition());
    runtime.setHostOffsets(offsets);
    Batch batch;
    batch.insert("tokens",Impl::RaggedTensor(values,runtime));
    if (withLabels) {
        Impl::Tensor targets(cpu,Impl::TensorDescriptor(DataType::FP32,{b*kM1X,kOut1X}));
        for (uint64_t t=0; t<b*kM1X*kOut1X; ++t) targets.getMemPtr<float>()[t]=0.f;
        batch.insert("labels",Impl::RaggedTensor(targets,runtime));
    }
    batch.setValidExampleCount(b);
    return batch;
}
vector<float> download1X(Impl::Tensor value) {
    const auto bytes=value.getArraySizeInBytes();
    EXPECT_EQ(bytes%sizeof(float),0u);
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor host=value.clone(cpu);
    Stream stream=Stream::getNextDownloadStream(value.getPlacement().getDeviceNum());
    host.downloadSection(value,stream,0,0,bytes);
    stream.synchronize();
    const auto n=bytes/sizeof(float);
    return vector<float>(host.getMemPtr<float>(),host.getMemPtr<float>()+n);
}
} // namespace

TEST(RaggedFullyConnectedNeutral1X, SemanticSchemaMemoryAndFixedCompatibility) {
    Network network("fc_1x_schema");
    const auto input=tokens1X(network);
    const auto fc=build1X(network,input,true);
    ASSERT_TRUE(fc.getRaggedFeatureOutput().has_value());
    const auto output=fc.getRaggedFeatureOutput().value();
    EXPECT_TRUE(output.sharesPartitionWith(input));
    EXPECT_NE(output.getValues().getOriginalId(),input.getValues().getOriginalId());
    EXPECT_EQ(output.getValuesDimensions(),(vector<uint64_t>{kM1X,kOut1X}));
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    for (uint32_t b : {2u,7u}) {
        EXPECT_EQ(fc.getOutputTensorBytes(b),b*kM1X*kOut1X*sizeof(float));
        EXPECT_EQ(fc.getNonFirstInstanceMemRequirementInBytes(b,cpu),fc.getOutputTensorBytes(b));
        EXPECT_EQ(fc.getFirstInstanceMemRequirementInBytes(b,cpu),
                  fc.getOutputTensorBytes(b)+(kIn1X*kOut1X+kOut1X)*sizeof(float));
    }
    EXPECT_ANY_THROW((void)fc.getOutputTensorBytes(0));
    const json record=fc.architectureJson();
    ASSERT_EQ(record.at("version"),"1.3.0");
    for (const char* name : {"ragged_inputs","ragged_outputs"}) {
        const json& r=record.at(name).at(0);
        EXPECT_FALSE(r.contains("geometry_mode"));
        EXPECT_FALSE(r.contains("batch_size"));
        EXPECT_FALSE(r.contains("max_total_values"));
        EXPECT_EQ(r.at("offsets").at("dimensions"),(vector<uint64_t>{1}));
    }
    shared_ptr<thor_file::TarReader> reader;
    auto reject=[&](json broken) { EXPECT_ANY_THROW(FullyConnected::deserialize(reader,broken,&network)); };
    // 1.2.0 is a supported historical feature/schema version, not an
    // unsupported geometry mode. Use a genuinely unknown version here.
    json bad=record; bad["version"]="9.9.9"; reject(bad);
    bad=record; bad["batch_size"]=7; reject(bad);
    bad=record; bad["ragged_inputs"][0]["batch_size"]=7; reject(bad);
    bad=record; bad["ragged_outputs"][0]["max_total_values"]=28; reject(bad);
    bad=record; bad["ragged_inputs"][0]["max_values_per_row"]=5; reject(bad);
    bad=record; bad["ragged_inputs"][0]["values"]["dimensions"]=vector<uint64_t>{7,kIn1X}; reject(bad);
    bad=record; bad["ragged_inputs"][0]["offsets"]["id"]=input.getValues().getOriginalId(); reject(bad);
    bad=record; bad["ragged_outputs"][0]["offsets"]["id"]=input.getValues().getOriginalId(); reject(bad);
    bad=record; bad["ragged_outputs"][0]["values"]["data_type"]=DataType::BF16; reject(bad);
    bad=record; bad["outputs"][0]["dimensions"]=vector<uint64_t>{kM1X,3}; reject(bad);
    bad=record; bad["inputs"][0]["data_type"]=DataType::BF16; reject(bad);
    bad=record; bad["use_residual"]=true; reject(bad);
    bad=record; bad["parameters"]["weights"]["shape"]=vector<uint64_t>{kIn1X,3}; reject(bad);
    bad=record; bad["parameters"]["biases"]["dtype"]=DataType::BF16; reject(bad);

    // A second logical input still writes the same schema irrespective of
    // which runtime batch sizes are later selected for placement.
    Network secondNetwork("fc_1x_second_logical_input");
    const auto second=RaggedNetworkInput::Builder().network(secondNetwork).name("tokens")
        .valuesDataType(DataType::FP32).offsetsDataType(DataType::UINT64)
        .trailingDimensions({kIn1X}).maxValuesPerRow(4).build();
    const auto secondFc=build1X(secondNetwork,second,true);
    const json secondRecord=secondFc.architectureJson();
    EXPECT_EQ(secondRecord.at("version"),"1.3.0");
    EXPECT_FALSE(secondRecord.at("ragged_inputs").at(0).contains("batch_size"));
    EXPECT_FALSE(secondRecord.at("ragged_inputs").at(0).contains("max_total_values"));
    EXPECT_EQ(second.getDescriptorForPlacement(2).getMaxTotalValues(),8u);
    EXPECT_EQ(second.getDescriptorForPlacement(7).getMaxTotalValues(),28u);
}

TEST(RaggedFullyConnectedNeutral1X, SavedInferenceReplacesB2AndB7WithEmptyRows) {
    int devices=0;
    if (cudaGetDeviceCount(&devices)!=cudaSuccess || devices==0) GTEST_SKIP()<<"CUDA device unavailable";
    const auto archive=archive1X("fc_1x_inference");
    {
        Network network("fc_1x_inference");
        const auto input=tokens1X(network);
        const auto fc=build1X(network,input,true);
        (void)RaggedNetworkOutput::Builder().network(network).name("projected")
            .inputTensor(fc.getRaggedFeatureOutput().value()).build();
        network.save(archive.string(),true);
    }
    for (uint32_t b : {2u,7u}) {
        SCOPED_TRACE(::testing::Message()<<"B="<<b);
        Network network("fc_1x_inference");
        ASSERT_NO_THROW(network.load(archive.string()));
        const auto input=network.getExternalRaggedNetworkInputs().front().raggedTensor;
        auto fc=find1X(network);
        ASSERT_NE(fc,nullptr);
        const auto before=network.architectureJson();
        vector<Event> events;
        auto placed=network.place(b,events,true);
        ASSERT_NE(placed,nullptr);
        for (auto& event:events) event.synchronize();
        auto& stamped=placed->getStampedNetwork(0);
        auto physical=dynamic_pointer_cast<Impl::RaggedFullyConnected>(
            stamped.getPhysicalLayerFromApiLayer(fc->getId()));
        ASSERT_NE(physical,nullptr);
        ASSERT_TRUE(physical->getFeatureOutput().has_value());
        EXPECT_EQ(physical->getFeatureOutput()->getDimensions(),(vector<uint64_t>{b*kM1X,kOut1X}));
        const auto weights=download1X(physical->getParameterStorage("weights"));
        const auto biases=download1X(physical->getParameterStorage("biases"));
        auto output=placed->inferLogical(batch1X(input,b));
        ASSERT_TRUE(output.contains("projected"));
        ASSERT_TRUE(holds_alternative<Impl::RaggedTensor>(output.at("projected")));
        const auto result=get<Impl::RaggedTensor>(output.at("projected"));
        const auto ends=result.getHostOffsetsIfAvailable();
        ASSERT_TRUE(ends.has_value());
        EXPECT_EQ(ends.value(),offsets1X(b));
        const float* observed=result.getValues().getMemPtr<float>();
        for (uint64_t t=0; t<ends->back(); ++t)
            for (uint32_t o=0; o<kOut1X; ++o) {
                float expected=biases[o];
                for (uint32_t i=0; i<kIn1X; ++i) expected+=token1X(t,i)*weights[i*kOut1X+o];
                EXPECT_NEAR(observed[t*kOut1X+o],expected,2e-3f);
            }
        EXPECT_EQ(network.architectureJson(),before);
    }
    filesystem::remove_all(archive);
}

TEST(RaggedFullyConnectedNeutral1X, SavedTrainingB2AndB7UpdatesWeightsBiasAndInputGradient) {
    int devices=0;
    if (cudaGetDeviceCount(&devices)!=cudaSuccess || devices==0) GTEST_SKIP()<<"CUDA device unavailable";
    const auto archive=archive1X("fc_1x_training");
    {
        Network network("fc_1x_training");
        (void)Sgd::Builder().network(network).initialLearningRate(.001f).decay(0.f).momentum(0.f).build();
        const auto tokens=tokens1X(network);
        const auto labels=RaggedNetworkInput::Builder().network(network).name("labels")
            .valuesDataType(DataType::FP32).trailingDimensions({kOut1X}).partition(tokens).build();
        const auto inputRivet=GradientRivet::Builder().network(network).tensor(tokens.getValues()).build();
        const auto raggedInput=tokens.withValues(inputRivet.getFeatureOutput().value());
        auto fc=build1X(network,raggedInput,true);
        auto prediction=Impl::Expression::input("predictions",DataType::FP32,DataType::FP32);
        auto label=Impl::Expression::input("labels",DataType::FP32,DataType::FP32);
        auto diff=prediction-label;
        auto loss=Impl::DynamicExpression::fromExpressionDefinition(
            Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({{
                "loss",(diff*diff).withOutputDType(DataType::FP32)}})));
        auto grad=Impl::DynamicExpression::fromExpressionDefinition(
            Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({{
                "predictions_grad",(diff*Impl::Expression(2.0*Impl::Loss::getLossScalingFactor()))
                    .withOutputDType(DataType::FP32)}})));
        auto raw=RaggedCustomLoss::Builder().network(network)
            .predictions(fc.getRaggedFeatureOutput().value()).labels(labels)
            .lossExpression(loss).gradientExpression(grad).build();
        (void)NetworkOutput::Builder().network(network).name("raw_loss")
            .inputTensor(raw.getRaggedRawLoss().getValues()).dataType(DataType::FP32).build();
        network.save(archive.string(),true);
    }
    for (uint32_t b : {2u,7u}) {
        SCOPED_TRACE(::testing::Message()<<"B="<<b);
        Network network("fc_1x_training");
        ASSERT_NO_THROW(network.load(archive.string()));
        const auto allInputs=network.getExternalRaggedNetworkInputs();
        auto tokenIt=find_if(allInputs.begin(),allInputs.end(),[](const auto& x){return x.name=="tokens";});
        ASSERT_NE(tokenIt,allInputs.end());
        const auto input=tokenIt->raggedTensor;
        auto fc=find1X(network);
        ASSERT_NE(fc,nullptr);
        const auto before=network.architectureJson();
        vector<Event> events;
        auto placed=network.place(b,events,false);
        ASSERT_NE(placed,nullptr);
        for (auto& e:events) e.synchronize();
        auto physical=dynamic_pointer_cast<Impl::RaggedFullyConnected>(
            placed->getStampedNetwork(0).getPhysicalLayerFromApiLayer(fc->getId()));
        ASSERT_NE(physical,nullptr);
        const auto oldWeights=download1X(physical->getParameterStorage("weights"));
        const auto oldBias=download1X(physical->getParameterStorage("biases"));
        map<string,Impl::Tensor> outputs;
        map<string,Event> ready;
        auto finished=placed->submitBatch(0,batch1X(input,b,true),outputs,ready,false);
        finished.synchronize();
        placed->synchronize();
        ASSERT_TRUE(outputs.contains("raw_loss"));
        ASSERT_GT(physical->getErrorOutputs().size(),0u);
        ASSERT_TRUE(physical->getErrorOutputs()[0].has_value());
        const auto dx=download1X(physical->getErrorOutputs()[0].value());
        const auto newWeights=download1X(physical->getParameterStorage("weights"));
        const auto newBias=download1X(physical->getParameterStorage("biases"));
        const auto ends=offsets1X(b);
        float gradientWeights[kIn1X*kOut1X]={};
        float gradientBias[kOut1X]={};
        for (uint64_t t=0; t<ends.back(); ++t)
            for (uint32_t o=0; o<kOut1X; ++o) {
                float y=oldBias[o];
                for (uint32_t i=0; i<kIn1X; ++i) y+=token1X(t,i)*oldWeights[i*kOut1X+o];
                gradientBias[o]+=2*y;
                for (uint32_t i=0; i<kIn1X; ++i)
                    gradientWeights[i*kOut1X+o]+=2*y*token1X(t,i);
            }
        const float scale=Impl::Loss::getLossScalingFactor();
        for (uint64_t t=0; t<ends.back(); ++t) {
            for (uint32_t i=0; i<kIn1X; ++i) {
                // Derivative of sum((XW+b)^2), computed from pre-update
                // parameters. This checks the input-gradient path independently
                // of the optimizer updates and excludes inactive packed tails.
                float expected=0.f;
                for (uint32_t o=0; o<kOut1X; ++o) {
                    float y=oldBias[o];
                    for (uint32_t j=0; j<kIn1X; ++j)
                        y+=token1X(t,j)*oldWeights[j*kOut1X+o];
                    expected+=2.f*y*oldWeights[i*kOut1X+o];
                }
                ASSERT_GT(dx.size(),t*kIn1X+i);
                EXPECT_NEAR(dx[t*kIn1X+i]/scale,expected,2e-2f);
            }
        }
        bool weightUpdated=false, biasUpdated=false;
        for (uint32_t i=0; i<kIn1X*kOut1X; ++i) {
            const float delta=newWeights[i]-oldWeights[i];
            weightUpdated|=fabs(delta)>1e-8f;
            if (fabs(gradientWeights[i])>1e-3f) EXPECT_LE(delta*gradientWeights[i],1e-7f);
        }
        for (uint32_t i=0; i<kOut1X; ++i) {
            const float delta=newBias[i]-oldBias[i];
            biasUpdated|=fabs(delta)>1e-8f;
            if (fabs(gradientBias[i])>1e-3f) EXPECT_LE(delta*gradientBias[i],1e-7f);
        }
        EXPECT_TRUE(weightUpdated);
        EXPECT_TRUE(biasUpdated);
        EXPECT_EQ(network.architectureJson(),before);
    }
    filesystem::remove_all(archive);
}


// 1Y-A: only the existing activation post-op on a single neutral ragged FC
// application. Residual, custom epilogue and multi-input variants
// remain deliberately gated until their own user-tested patches.
namespace {
FullyConnected buildActivated1YA(Network& network, const RaggedTensor& input) {
    return FullyConnected::Builder().network(network).featureInput(input)
        .numOutputFeatures(kOut1X).hasBias(true).weightsDataType(DataType::FP32)
        .computeDataType(DataType::FP32).outputDataType(DataType::FP32)
        .activation(Sigmoid::Builder().build()).build();
}
float projected1YA(uint64_t token, uint32_t output,
                   const vector<float>& weights, const vector<float>& bias) {
    float linear = bias.at(output);
    for (uint32_t i=0; i<kIn1X; ++i) linear+=token1X(token,i)*weights.at(i*kOut1X+output);
    return 1.f/(1.f+std::exp(-linear));
}
} // namespace

TEST(RaggedFullyConnectedNeutral1YA, ActivationSchemaIsDistinctAndStrict) {
    Network network("fc_1ya_schema");
    const auto input=tokens1X(network);
    const auto fc=buildActivated1YA(network,input);
    const json record=fc.architectureJson();
    ASSERT_EQ(record.at("version"),"1.4.0");
    ASSERT_TRUE(record.at("activation").is_object());
    EXPECT_FALSE(record.at("ragged_inputs").at(0).contains("geometry_mode"));
    EXPECT_FALSE(record.at("ragged_inputs").at(0).contains("batch_size"));
    EXPECT_FALSE(record.at("ragged_outputs").at(0).contains("max_total_values"));
    EXPECT_TRUE(fc.getRaggedFeatureOutput()->sharesPartitionWith(input));
    for (uint32_t b : {2u,7u})
        EXPECT_EQ(fc.getOutputTensorBytes(b),b*kM1X*kOut1X*sizeof(float));
    shared_ptr<thor_file::TarReader> reader;
    auto reject=[&](json broken){EXPECT_ANY_THROW(FullyConnected::deserialize(reader,broken,&network));};
    json bad=record; bad["version"]="1.3.0"; reject(bad);
    bad=record; bad["activation"]=nullptr; reject(bad);
    bad=record; bad["activation"]="sigmoid"; reject(bad);
    bad=record; bad["use_residual"]=true; reject(bad);
    bad=record; bad["output_dropout_probability"]=0.25f; reject(bad);
    bad=record; bad["ragged_inputs"][0]["batch_size"]=7; reject(bad);
    bad=record; bad["ragged_outputs"][0]["offsets"]["id"]=input.getValues().getOriginalId(); reject(bad);
    bad=record; bad["parameters"]["weights"]["shape"]=vector<uint64_t>{kIn1X,3}; reject(bad);
    // Existing 1.3.0 archives must remain activation-free.
    const json basic=build1X(network,input,true).architectureJson();
    ASSERT_EQ(basic.at("version"),"1.3.0");
    bad=basic; bad["activation"]=record.at("activation"); reject(bad);
}

TEST(RaggedFullyConnectedNeutral1YA, SavedActivatedInferenceAtB2AndB7) {
    int devices=0;
    if (cudaGetDeviceCount(&devices)!=cudaSuccess || devices==0) GTEST_SKIP()<<"CUDA device unavailable";
    const auto archive=archive1X("fc_1ya_inference");
    {
        Network network("fc_1ya_inference");
        auto input=tokens1X(network);
        auto fc=buildActivated1YA(network,input);
        (void)RaggedNetworkOutput::Builder().network(network).name("activated")
            .inputTensor(fc.getRaggedFeatureOutput().value()).build();
        network.save(archive.string(),true);
    }
    for (uint32_t b : {2u,7u}) {
        SCOPED_TRACE(::testing::Message()<<"B="<<b);
        Network network("fc_1ya_inference");
        ASSERT_NO_THROW(network.load(archive.string()));
        const auto input=network.getExternalRaggedNetworkInputs().front().raggedTensor;
        auto fc=find1X(network);
        ASSERT_NE(fc,nullptr);
        EXPECT_EQ(fc->architectureJson().at("version"),"1.4.0");
        const auto before=network.architectureJson();
        vector<Event> events;
        auto placed=network.place(b,events,true);
        ASSERT_NE(placed,nullptr);
        for (auto& event:events) event.synchronize();
        auto& stamped=placed->getStampedNetwork(0);
        auto physical=dynamic_pointer_cast<Impl::RaggedFullyConnected>(
            stamped.getPhysicalLayerFromApiLayer(fc->getId()));
        ASSERT_NE(physical,nullptr);
        ASSERT_TRUE(physical->getFeatureOutput().has_value());
        EXPECT_EQ(physical->getFeatureOutput()->getDimensions(),(vector<uint64_t>{b*kM1X,kOut1X}));
        const auto weights=download1X(physical->getParameterStorage("weights"));
        const auto bias=download1X(physical->getParameterStorage("biases"));
        auto resultMap=placed->inferLogical(batch1X(input,b));
        ASSERT_TRUE(resultMap.contains("activated"));
        ASSERT_TRUE(holds_alternative<Impl::RaggedTensor>(resultMap.at("activated")));
        const auto result=get<Impl::RaggedTensor>(resultMap.at("activated"));
        const auto ends=result.getHostOffsetsIfAvailable();
        ASSERT_TRUE(ends.has_value());
        EXPECT_EQ(ends.value(),offsets1X(b));
        const float* actual=result.getValues().getMemPtr<float>();
        for (uint64_t t=0;t<ends->back();++t)
            for (uint32_t o=0;o<kOut1X;++o)
                EXPECT_NEAR(actual[t*kOut1X+o],projected1YA(t,o,weights,bias),2e-3f);
        EXPECT_EQ(network.architectureJson(),before);
    }
    filesystem::remove_all(archive);
}

TEST(RaggedFullyConnectedNeutral1YA, SavedActivatedTrainingB2AndB7BackpropagatesActiveOnly) {
    int devices=0;
    if (cudaGetDeviceCount(&devices)!=cudaSuccess || devices==0) GTEST_SKIP()<<"CUDA device unavailable";
    const auto archive=archive1X("fc_1ya_training");
    {
        Network network("fc_1ya_training");
        (void)Sgd::Builder().network(network).initialLearningRate(.001f).decay(0.f).momentum(0.f).build();
        const auto input=tokens1X(network);
        const auto labels=RaggedNetworkInput::Builder().network(network).name("labels")
            .valuesDataType(DataType::FP32).trailingDimensions({kOut1X}).partition(input).build();
        const auto rivet=GradientRivet::Builder().network(network).tensor(input.getValues()).build();
        const auto fc=buildActivated1YA(network,input.withValues(rivet.getFeatureOutput().value()));
        auto predictions=Impl::Expression::input("predictions",DataType::FP32,DataType::FP32);
        auto targets=Impl::Expression::input("labels",DataType::FP32,DataType::FP32);
        auto diff=predictions-targets;
        auto loss=Impl::DynamicExpression::fromExpressionDefinition(
            Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({{
                "loss",(diff*diff).withOutputDType(DataType::FP32)}})));
        auto grad=Impl::DynamicExpression::fromExpressionDefinition(
            Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({{
                "predictions_grad",(diff*Impl::Expression(2.0*Impl::Loss::getLossScalingFactor()))
                    .withOutputDType(DataType::FP32)}})));
        auto raw=RaggedCustomLoss::Builder().network(network)
            .predictions(fc.getRaggedFeatureOutput().value()).labels(labels)
            .lossExpression(loss).gradientExpression(grad).build();
        (void)NetworkOutput::Builder().network(network).name("raw_loss")
            .inputTensor(raw.getRaggedRawLoss().getValues()).dataType(DataType::FP32).build();
        network.save(archive.string(),true);
    }
    for (uint32_t b : {2u,7u}) {
        SCOPED_TRACE(::testing::Message()<<"B="<<b);
        Network network("fc_1ya_training");
        ASSERT_NO_THROW(network.load(archive.string()));
        auto fc=find1X(network);
        ASSERT_NE(fc,nullptr);
        const auto before=network.architectureJson();
        const auto inputs=network.getExternalRaggedNetworkInputs();
        auto it=find_if(inputs.begin(),inputs.end(),[](const auto& item){return item.name=="tokens";});
        ASSERT_NE(it,inputs.end());
        vector<Event> events;
        auto placed=network.place(b,events,false);
        ASSERT_NE(placed,nullptr);
        for (auto& event:events) event.synchronize();
        auto physical=dynamic_pointer_cast<Impl::RaggedFullyConnected>(
            placed->getStampedNetwork(0).getPhysicalLayerFromApiLayer(fc->getId()));
        ASSERT_NE(physical,nullptr);
        const auto oldWeights=download1X(physical->getParameterStorage("weights"));
        const auto oldBias=download1X(physical->getParameterStorage("biases"));
        map<string,Impl::Tensor> outputs;
        map<string,Event> ready;
        auto finished=placed->submitBatch(0,batch1X(it->raggedTensor,b,true),outputs,ready,false);
        finished.synchronize();
        placed->synchronize();
        ASSERT_TRUE(outputs.contains("raw_loss"));
        ASSERT_FALSE(physical->getErrorOutputs().empty());
        ASSERT_TRUE(physical->getErrorOutputs()[0].has_value());
        const auto dx=download1X(physical->getErrorOutputs()[0].value());
        const auto newWeights=download1X(physical->getParameterStorage("weights"));
        const auto newBias=download1X(physical->getParameterStorage("biases"));
        float dw[kIn1X*kOut1X]={};
        float db[kOut1X]={};
        const auto ends=offsets1X(b);
        const float scale=Impl::Loss::getLossScalingFactor();
        for (uint64_t t=0;t<ends.back();++t) {
            for (uint32_t o=0;o<kOut1X;++o) {
                const float s=projected1YA(t,o,oldWeights,oldBias);
                const float delta=2.f*s*s*(1.f-s);
                db[o]+=delta;
                for (uint32_t i=0;i<kIn1X;++i) dw[i*kOut1X+o]+=token1X(t,i)*delta;
            }
            for (uint32_t i=0;i<kIn1X;++i) {
                float expected=0.f;
                for (uint32_t o=0;o<kOut1X;++o) {
                    const float s=projected1YA(t,o,oldWeights,oldBias);
                    expected+=2.f*s*s*(1.f-s)*oldWeights[i*kOut1X+o];
                }
                ASSERT_GT(dx.size(),t*kIn1X+i);
                EXPECT_NEAR(dx[t*kIn1X+i]/scale,expected,2e-2f);
            }
        }
        bool weightUpdated=false, biasUpdated=false;
        for (uint32_t i=0;i<kIn1X*kOut1X;++i) {
            const float change=newWeights[i]-oldWeights[i];
            weightUpdated|=fabs(change)>1e-8f;
            if (fabs(dw[i])>1e-3f) EXPECT_LE(change*dw[i],1e-7f);
        }
        for (uint32_t i=0;i<kOut1X;++i) {
            const float change=newBias[i]-oldBias[i];
            biasUpdated|=fabs(change)>1e-8f;
            if (fabs(db[i])>1e-3f) EXPECT_LE(change*db[i],1e-7f);
        }
        EXPECT_TRUE(weightUpdated);
        EXPECT_TRUE(biasUpdated);
        EXPECT_EQ(network.architectureJson(),before);
    }
    filesystem::remove_all(archive);
}


// 1Y-B: first-class residual is a values input sharing the feature tensor's
// canonical row partition, not a second runtime partition.
namespace {
RaggedTensor residual1YB(Network& network, const RaggedTensor& tokens) {
    return RaggedNetworkInput::Builder().network(network).name("residual")
        .valuesDataType(DataType::FP32).trailingDimensions({kOut1X})
        .partition(tokens).build();
}
FullyConnected buildResidual1YB(Network& network, const RaggedTensor& tokens,
                                const RaggedTensor& residual) {
    return FullyConnected::Builder().network(network).featureInput(tokens)
        .numOutputFeatures(kOut1X).hasBias(true).weightsDataType(DataType::FP32)
        .computeDataType(DataType::FP32).outputDataType(DataType::FP32)
        .noActivation().residualInput(residual).build();
}
float residualValue1YB(uint64_t t, uint32_t c) {
    return .35f + .09f*static_cast<float>(t) - .12f*static_cast<float>(c);
}
Batch batch1YB(const RaggedTensor& tokens, const RaggedTensor& residual,
              uint32_t b, bool withLabels=false) {
    Batch batch=batch1X(tokens,b,withLabels);
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    const auto descriptor=residual.getDescriptorForPlacement(b);
    const auto ends=offsets1X(b);
    Impl::Tensor values(cpu,descriptor.getValuesDescriptor());
    Impl::Tensor offsets(cpu,descriptor.getOffsetsDescriptor());
    for (uint64_t t=0;t<b*kM1X;++t)
        for (uint32_t c=0;c<kOut1X;++c)
            values.getMemPtr<float>()[t*kOut1X+c]=
                t<ends.back()?residualValue1YB(t,c):numeric_limits<float>::quiet_NaN();
    for (uint32_t i=0;i<=b;++i) offsets.getMemPtr<uint64_t>()[i]=ends[i];
    Impl::RowPartitionRuntime runtime(offsets,descriptor.getRowPartition());
    runtime.setHostOffsets(ends);
    batch.insert("residual",Impl::RaggedTensor(values,runtime));
    return batch;
}
float projected1YB(uint64_t t, uint32_t o, const vector<float>& weights,
                   const vector<float>& bias) {
    float y=bias[o]+residualValue1YB(t,o);
    for (uint32_t i=0;i<kIn1X;++i) y+=token1X(t,i)*weights[i*kOut1X+o];
    return y;
}
} // namespace

TEST(RaggedFullyConnectedNeutral1YB, ResidualSchemaAndCanonicalPartitionAreStrict) {
    Network network("fc_1yb_schema");
    const auto tokens=tokens1X(network);
    const auto residual=residual1YB(network,tokens);
    const auto fc=buildResidual1YB(network,tokens,residual);
    const auto record=fc.architectureJson();
    ASSERT_EQ(record.at("version"),"1.5.0");
    EXPECT_TRUE(record.at("use_residual").get<bool>());
    ASSERT_TRUE(fc.getRaggedFeatureOutput().has_value());
    EXPECT_TRUE(fc.getRaggedFeatureOutput()->sharesPartitionWith(tokens));
    ASSERT_TRUE(fc.getRaggedResidualInput().has_value());
    EXPECT_TRUE(fc.getRaggedResidualInput()->sharesPartitionWith(tokens));
    EXPECT_NE(residual.getValues().getOriginalId(),tokens.getValues().getOriginalId());
    const json& rr=record.at("ragged_residual_input");
    EXPECT_FALSE(rr.contains("geometry_mode"));
    EXPECT_EQ(rr.at("max_values_per_row"),kM1X);
    EXPECT_FALSE(rr.contains("batch_size"));
    EXPECT_FALSE(rr.contains("max_total_values"));
    EXPECT_EQ(rr.at("offsets").at("dimensions"),(vector<uint64_t>{1}));
    EXPECT_EQ(rr.at("offsets").at("id"),
              record.at("ragged_inputs").at(0).at("offsets").at("id"));
    for (uint32_t b : {2u,7u})
        EXPECT_EQ(fc.getOutputTensorBytes(b),b*kM1X*kOut1X*sizeof(float));
    shared_ptr<thor_file::TarReader> reader;
    auto reject=[&](json broken) { EXPECT_ANY_THROW(FullyConnected::deserialize(reader,broken,&network)); };
    json bad=record; bad["version"]="1.4.0"; reject(bad);
    bad=record; bad["version"]="1.3.0"; reject(bad);
    bad=record; bad["activation"]="sigmoid"; reject(bad);
    bad=record; bad["use_residual"]=false; reject(bad);
    bad=record; bad.erase("ragged_residual_input"); reject(bad);
    bad=record; bad["residual_input"]["dimensions"]=vector<uint64_t>{kM1X,kIn1X}; reject(bad);
    bad=record; bad["residual_input"]["data_type"]=DataType::BF16; reject(bad);
    bad=record; bad["ragged_residual_input"]["values"]["id"]=tokens.getValues().getOriginalId(); reject(bad);
    bad=record; bad["ragged_residual_input"]["values"]["dimensions"]=vector<uint64_t>{kM1X,3}; reject(bad);
    bad=record; bad["ragged_residual_input"]["offsets"]["id"]=tokens.getValues().getOriginalId(); reject(bad);
    bad=record; bad["ragged_residual_input"]["offsets"]["data_type"]=DataType::UINT32; reject(bad);
    bad=record; bad["ragged_residual_input"]["max_values_per_row"]=kM1X+1; reject(bad);
    bad=record; bad["ragged_residual_input"]["batch_size"]=2; reject(bad);
    bad=record; bad["output_dropout_probability"]=1.0f; reject(bad);
    bad=record; bad["parameters"]["weights"]["shape"]=vector<uint64_t>{kOut1X,kOut1X}; reject(bad);
    // A real, but different, row partition must be rejected by the builder.
    const auto wrong=RaggedNetworkInput::Builder().network(network).name("wrong_partition")
        .valuesDataType(DataType::FP32).offsetsDataType(DataType::UINT64)
        .trailingDimensions({kOut1X}).maxValuesPerRow(kM1X).build();
    EXPECT_ANY_THROW((void)buildResidual1YB(network,tokens,wrong));
    // An independently constructed logical model also emits the current
    // residual schema, regardless of old runtime batch/capacity hints.
    Network legacyNetwork("fc_1yb_legacy");
    const auto fixed=RaggedNetworkInput::Builder().network(legacyNetwork).name("fixed")
        .valuesDataType(DataType::FP32).offsetsDataType(DataType::UINT64)
        .trailingDimensions({kIn1X}).maxValuesPerRow(4).build();
    const auto fixedResidual=RaggedNetworkInput::Builder().network(legacyNetwork).name("fixed_residual")
        .valuesDataType(DataType::FP32).trailingDimensions({kOut1X}).partition(fixed).build();
    const json additionalRecord = buildResidual1YB(legacyNetwork,fixed,fixedResidual).architectureJson();
    EXPECT_EQ(additionalRecord.at("version"), "1.5.0");
    EXPECT_FALSE(additionalRecord.at("ragged_residual_input").contains("batch_size"));
}

TEST(RaggedFullyConnectedNeutral1YB, SavedResidualInferenceAtB2AndB7) {
    int devices=0;
    if (cudaGetDeviceCount(&devices)!=cudaSuccess || devices==0) GTEST_SKIP()<<"CUDA device unavailable";
    const auto archive=archive1X("fc_1yb_inference");
    {
        Network network("fc_1yb_inference");
        const auto tokens=tokens1X(network);
        const auto residual=residual1YB(network,tokens);
        const auto fc=buildResidual1YB(network,tokens,residual);
        (void)RaggedNetworkOutput::Builder().network(network).name("projected")
            .inputTensor(fc.getRaggedFeatureOutput().value()).build();
        network.save(archive.string(),true);
    }
    for (uint32_t b : {2u,7u}) {
        SCOPED_TRACE(::testing::Message()<<"B="<<b);
        Network network("fc_1yb_inference");
        ASSERT_NO_THROW(network.load(archive.string()));
        auto fc=find1X(network);
        ASSERT_NE(fc,nullptr);
        ASSERT_TRUE(fc->getRaggedResidualInput().has_value());
        const auto before=network.architectureJson();
        const auto inputs=network.getExternalRaggedNetworkInputs();
        auto tokenIt=find_if(inputs.begin(),inputs.end(),[](const auto& x){return x.name=="tokens";});
        auto residualIt=find_if(inputs.begin(),inputs.end(),[](const auto& x){return x.name=="residual";});
        ASSERT_NE(tokenIt,inputs.end());
        ASSERT_NE(residualIt,inputs.end());
        vector<Event> events;
        auto placed=network.place(b,events,true);
        ASSERT_NE(placed,nullptr);
        for (auto& e:events) e.synchronize();
        auto& stamped=placed->getStampedNetwork(0);
        const auto physical=dynamic_pointer_cast<Impl::RaggedFullyConnected>(
            stamped.getPhysicalLayerFromApiLayer(fc->getId()));
        ASSERT_NE(physical,nullptr);
        ASSERT_TRUE(physical->getFeatureOutput().has_value());
        EXPECT_EQ(physical->getFeatureOutput()->getDimensions(),(vector<uint64_t>{b*kM1X,kOut1X}));
        const auto weights=download1X(physical->getParameterStorage("weights"));
        const auto bias=download1X(physical->getParameterStorage("biases"));
        auto output=placed->inferLogical(batch1YB(tokenIt->raggedTensor,residualIt->raggedTensor,b));
        ASSERT_TRUE(output.contains("projected"));
        ASSERT_TRUE(holds_alternative<Impl::RaggedTensor>(output.at("projected")));
        const auto result=get<Impl::RaggedTensor>(output.at("projected"));
        ASSERT_TRUE(result.getHostOffsetsIfAvailable().has_value());
        EXPECT_EQ(result.getHostOffsetsIfAvailable().value(),offsets1X(b));
        const float* actual=result.getValues().getMemPtr<float>();
        for (uint64_t t=0;t<offsets1X(b).back();++t)
            for (uint32_t o=0;o<kOut1X;++o)
                EXPECT_NEAR(actual[t*kOut1X+o],projected1YB(t,o,weights,bias),2e-3f);
        EXPECT_EQ(network.architectureJson(),before);
    }
    filesystem::remove_all(archive);
}

TEST(RaggedFullyConnectedNeutral1YB, SavedResidualTrainingB2AndB7RoutesBothGradients) {
    int devices=0;
    if (cudaGetDeviceCount(&devices)!=cudaSuccess || devices==0) GTEST_SKIP()<<"CUDA device unavailable";
    const auto archive=archive1X("fc_1yb_training");
    {
        Network network("fc_1yb_training");
        (void)Sgd::Builder().network(network).initialLearningRate(.001f).decay(0.f).momentum(0.f).build();
        const auto tokens=tokens1X(network);
        const auto residual=residual1YB(network,tokens);
        const auto labels=RaggedNetworkInput::Builder().network(network).name("labels")
            .valuesDataType(DataType::FP32).trailingDimensions({kOut1X}).partition(tokens).build();
        const auto inputRivet=GradientRivet::Builder().network(network).tensor(tokens.getValues()).build();
        const auto residualRivet=GradientRivet::Builder().network(network).tensor(residual.getValues()).build();
        auto fc=buildResidual1YB(network,tokens.withValues(inputRivet.getFeatureOutput().value()),
                                residual.withValues(residualRivet.getFeatureOutput().value()));
        auto prediction=Impl::Expression::input("predictions",DataType::FP32,DataType::FP32);
        auto label=Impl::Expression::input("labels",DataType::FP32,DataType::FP32);
        auto diff=prediction-label;
        auto loss=Impl::DynamicExpression::fromExpressionDefinition(
            Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({{
                "loss",(diff*diff).withOutputDType(DataType::FP32)}})));
        auto grad=Impl::DynamicExpression::fromExpressionDefinition(
            Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({{
                "predictions_grad",(diff*Impl::Expression(2.0*Impl::Loss::getLossScalingFactor()))
                    .withOutputDType(DataType::FP32)}})));
        auto raw=RaggedCustomLoss::Builder().network(network)
            .predictions(fc.getRaggedFeatureOutput().value()).labels(labels)
            .lossExpression(loss).gradientExpression(grad).build();
        (void)NetworkOutput::Builder().network(network).name("raw_loss")
            .inputTensor(raw.getRaggedRawLoss().getValues()).dataType(DataType::FP32).build();
        network.save(archive.string(),true);
    }
    for (uint32_t b : {2u,7u}) {
        SCOPED_TRACE(::testing::Message()<<"B="<<b);
        Network network("fc_1yb_training");
        ASSERT_NO_THROW(network.load(archive.string()));
        const auto inputs=network.getExternalRaggedNetworkInputs();
        auto tokens=find_if(inputs.begin(),inputs.end(),[](const auto& x){return x.name=="tokens";});
        auto residual=find_if(inputs.begin(),inputs.end(),[](const auto& x){return x.name=="residual";});
        ASSERT_NE(tokens,inputs.end());
        ASSERT_NE(residual,inputs.end());
        auto fc=find1X(network);
        ASSERT_NE(fc,nullptr);
        const auto before=network.architectureJson();
        vector<Event> events;
        auto placed=network.place(b,events,false);
        ASSERT_NE(placed,nullptr);
        for (auto& e:events) e.synchronize();
        auto physical=dynamic_pointer_cast<Impl::RaggedFullyConnected>(
            placed->getStampedNetwork(0).getPhysicalLayerFromApiLayer(fc->getId()));
        ASSERT_NE(physical,nullptr);
        const auto weights=download1X(physical->getParameterStorage("weights"));
        const auto bias=download1X(physical->getParameterStorage("biases"));
        map<string,Impl::Tensor> outputs;
        map<string,Event> ready;
        auto finished=placed->submitBatch(0,batch1YB(tokens->raggedTensor,residual->raggedTensor,b,true),
                                         outputs,ready,false);
        finished.synchronize();
        placed->synchronize();
        ASSERT_TRUE(outputs.contains("raw_loss"));
        const auto& gradients=physical->getErrorOutputs();
        ASSERT_GE(gradients.size(),3u);
        ASSERT_TRUE(gradients[0].has_value());
        ASSERT_TRUE(gradients[2].has_value());
        const auto dx=download1X(gradients[0].value());
        const auto dr=download1X(gradients[2].value());
        const auto newWeights=download1X(physical->getParameterStorage("weights"));
        const auto newBias=download1X(physical->getParameterStorage("biases"));
        const auto ends=offsets1X(b);
        const float scale=Impl::Loss::getLossScalingFactor();
        float dw[kIn1X*kOut1X]={};
        float db[kOut1X]={};
        for (uint64_t t=0;t<ends.back();++t) {
            for (uint32_t o=0;o<kOut1X;++o) {
                const float delta=2.f*projected1YB(t,o,weights,bias);
                ASSERT_GT(dr.size(),t*kOut1X+o);
                EXPECT_NEAR(dr[t*kOut1X+o]/scale,delta,2e-2f);
                db[o]+=delta;
                for (uint32_t i=0;i<kIn1X;++i) dw[i*kOut1X+o]+=token1X(t,i)*delta;
            }
            for (uint32_t i=0;i<kIn1X;++i) {
                float expected=0.f;
                for (uint32_t o=0;o<kOut1X;++o)
                    expected+=2.f*projected1YB(t,o,weights,bias)*weights[i*kOut1X+o];
                ASSERT_GT(dx.size(),t*kIn1X+i);
                EXPECT_NEAR(dx[t*kIn1X+i]/scale,expected,2e-2f);
            }
        }
        bool updatedW=false, updatedB=false;
        for (uint32_t i=0;i<kIn1X*kOut1X;++i) {
            const float change=newWeights[i]-weights[i];
            updatedW|=fabs(change)>1e-8f;
            if (fabs(dw[i])>1e-3f) EXPECT_LE(change*dw[i],1e-7f);
        }
        for (uint32_t i=0;i<kOut1X;++i) {
            const float change=newBias[i]-bias[i];
            updatedB|=fabs(change)>1e-8f;
            if (fabs(db[i])>1e-3f) EXPECT_LE(change*db[i],1e-7f);
        }
        EXPECT_TRUE(updatedW);
        EXPECT_TRUE(updatedB);
        EXPECT_EQ(network.architectureJson(),before);
    }
    filesystem::remove_all(archive);
}


// 1Y-C: output dropout uses the existing physical FC dropout post-op, with
// placement-local Philox sequence state.  Only the configured probability and
// seed are model reconstruction state.  We intentionally keep combinations
// with residuals/activations and custom epilogues deferred.
namespace {
constexpr float kDropout1YC = 0.25f;
constexpr int64_t kDropoutSeed1YC = 0x12345;
FullyConnected buildDropout1YC(Network& network, const RaggedTensor& input) {
    return FullyConnected::Builder().network(network).featureInput(input)
        .numOutputFeatures(kOut1X).hasBias(true).weightsDataType(DataType::FP32)
        .computeDataType(DataType::FP32).outputDataType(DataType::FP32)
        .noActivation().outputDropoutProbability(kDropout1YC)
        .outputDropoutSeed(kDropoutSeed1YC).build();
}
float projected1YC(uint64_t token, uint32_t output, const vector<float>& weights,
                   const vector<float>& bias) {
    float v=bias.at(output);
    for (uint32_t i=0; i<kIn1X; ++i) v+=token1X(token,i)*weights.at(i*kOut1X+output);
    return v;
}
} // namespace

TEST(RaggedFullyConnectedNeutral1YC, DropoutSchemaIsStrictAndPreservesOldVariants) {
    Network network("fc_1yc_schema");
    const auto tokens=tokens1X(network);
    const auto fc=buildDropout1YC(network,tokens);
    const json record=fc.architectureJson();
    ASSERT_EQ(record.at("version"),"1.6.0");
    EXPECT_FLOAT_EQ(record.at("output_dropout_probability").get<float>(),kDropout1YC);
    EXPECT_EQ(record.at("output_dropout_seed").get<int64_t>(),kDropoutSeed1YC);
    EXPECT_TRUE(fc.getRaggedFeatureOutput()->sharesPartitionWith(tokens));
    for (uint32_t b : {2u,7u})
        EXPECT_EQ(fc.getOutputTensorBytes(b),b*kM1X*kOut1X*sizeof(float));
    for (const char* field : {"ragged_inputs","ragged_outputs"}) {
        const auto& ref=record.at(field).at(0);
        EXPECT_FALSE(ref.contains("geometry_mode"));
        EXPECT_FALSE(ref.contains("batch_size"));
        EXPECT_FALSE(ref.contains("max_total_values"));
    }
    shared_ptr<thor_file::TarReader> reader;
    auto reject=[&](json broken){ EXPECT_ANY_THROW(FullyConnected::deserialize(reader,broken,&network)); };
    json bad=record; bad["version"]="1.3.0"; reject(bad);
    bad=record; bad["version"]="1.4.0"; reject(bad);
    bad=record; bad["output_dropout_probability"]=0.0f; reject(bad);
    bad=record; bad["output_dropout_probability"]=1.0f; reject(bad);
    bad=record; bad["output_dropout_probability"]=-0.1f; reject(bad);
    bad=record; bad["output_dropout_seed"]="random"; reject(bad);
    bad=record; bad["activation"]=Sigmoid::Builder().build()->architectureJson(); reject(bad);
    bad=record; bad["use_residual"]=true; reject(bad);
    bad=record; bad["ragged_inputs"][0]["batch_size"]=2; reject(bad);
    bad=record; bad["ragged_outputs"][0]["offsets"]["id"]=tokens.getValues().getOriginalId(); reject(bad);
    bad=record; bad["parameters"]["weights"]["shape"]=vector<uint64_t>{4,kOut1X}; reject(bad);
    // Existing accepted neutral schema versions keep their original semantics.
    EXPECT_EQ(build1X(network,tokens,true).architectureJson().at("version"),"1.3.0");
    EXPECT_EQ(buildActivated1YA(network,tokens).architectureJson().at("version"),"1.4.0");
}

TEST(RaggedFullyConnectedNeutral1YC, SavedDropoutEvaluationB2B7DoesNotApplyMask) {
    int devices=0;
    if (cudaGetDeviceCount(&devices)!=cudaSuccess || devices==0) GTEST_SKIP()<<"CUDA device unavailable";
    const auto archive=archive1X("fc_1yc_inference");
    {
        Network network("fc_1yc_inference");
        const auto tokens=tokens1X(network);
        const auto fc=buildDropout1YC(network,tokens);
        (void)RaggedNetworkOutput::Builder().network(network).name("projected")
            .inputTensor(fc.getRaggedFeatureOutput().value()).build();
        network.save(archive.string(),true);
    }
    for (uint32_t b : {2u,7u}) {
        SCOPED_TRACE(::testing::Message()<<"B="<<b);
        Network network("fc_1yc_inference");
        ASSERT_NO_THROW(network.load(archive.string()));
        auto fc=find1X(network);
        ASSERT_NE(fc,nullptr);
        EXPECT_EQ(fc->architectureJson().at("version"),"1.6.0");
        EXPECT_FLOAT_EQ(fc->getOutputDropoutProbability(),kDropout1YC);
        EXPECT_EQ(fc->getOutputDropoutSeed(),kDropoutSeed1YC);
        const auto input=network.getExternalRaggedNetworkInputs().front().raggedTensor;
        const auto before=network.architectureJson();
        vector<Event> events;
        auto placed=network.place(b,events,true);
        ASSERT_NE(placed,nullptr);
        for (auto& event:events) event.synchronize();
        auto physical=dynamic_pointer_cast<Impl::RaggedFullyConnected>(
            placed->getStampedNetwork(0).getPhysicalLayerFromApiLayer(fc->getId()));
        ASSERT_NE(physical,nullptr);
        ASSERT_TRUE(physical->getFeatureOutput().has_value());
        EXPECT_EQ(physical->getFeatureOutput()->getDimensions(),(vector<uint64_t>{b*kM1X,kOut1X}));
        const auto weights=download1X(physical->getParameterStorage("weights"));
        const auto bias=download1X(physical->getParameterStorage("biases"));
        const auto outputs=placed->inferLogical(batch1X(input,b));
        ASSERT_TRUE(outputs.contains("projected"));
        const auto result=get<Impl::RaggedTensor>(outputs.at("projected"));
        const auto ends=result.getHostOffsetsIfAvailable();
        ASSERT_TRUE(ends.has_value());
        EXPECT_EQ(ends.value(),offsets1X(b));
        const float* actual=result.getValues().getMemPtr<float>();
        for (uint64_t t=0;t<ends->back();++t)
            for (uint32_t o=0;o<kOut1X;++o)
                EXPECT_NEAR(actual[t*kOut1X+o],projected1YC(t,o,weights,bias),2e-3f);
        EXPECT_EQ(network.architectureJson(),before);
    }
    filesystem::remove_all(archive);
}

TEST(RaggedFullyConnectedNeutral1YC, SavedDropoutTrainingB2B7PropagatesActualMaskGradients) {
    int devices=0;
    if (cudaGetDeviceCount(&devices)!=cudaSuccess || devices==0) GTEST_SKIP()<<"CUDA device unavailable";
    const auto archive=archive1X("fc_1yc_training");
    {
        Network network("fc_1yc_training");
        (void)Sgd::Builder().network(network).initialLearningRate(.001f).decay(0.f).momentum(0.f).build();
        const auto tokens=tokens1X(network);
        const auto labels=RaggedNetworkInput::Builder().network(network).name("labels")
            .valuesDataType(DataType::FP32).trailingDimensions({kOut1X}).partition(tokens).build();
        const auto rivet=GradientRivet::Builder().network(network).tensor(tokens.getValues()).build();
        const auto fc=buildDropout1YC(network,tokens.withValues(rivet.getFeatureOutput().value()));
        auto predictions=Impl::Expression::input("predictions",DataType::FP32,DataType::FP32);
        auto targets=Impl::Expression::input("labels",DataType::FP32,DataType::FP32);
        auto diff=predictions-targets;
        auto loss=Impl::DynamicExpression::fromExpressionDefinition(
            Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({{
                "loss",(diff*diff).withOutputDType(DataType::FP32)}})));
        auto gradient=Impl::DynamicExpression::fromExpressionDefinition(
            Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({{
                "predictions_grad",(diff*Impl::Expression(2.0*Impl::Loss::getLossScalingFactor()))
                    .withOutputDType(DataType::FP32)}})));
        const auto raw=RaggedCustomLoss::Builder().network(network)
            .predictions(fc.getRaggedFeatureOutput().value()).labels(labels)
            .lossExpression(loss).gradientExpression(gradient).build();
        (void)NetworkOutput::Builder().network(network).name("raw_loss")
            .inputTensor(raw.getRaggedRawLoss().getValues()).dataType(DataType::FP32).build();
        network.save(archive.string(),true);
    }
    for (uint32_t b : {2u,7u}) {
        SCOPED_TRACE(::testing::Message()<<"B="<<b);
        Network network("fc_1yc_training");
        ASSERT_NO_THROW(network.load(archive.string()));
        auto fc=find1X(network);
        ASSERT_NE(fc,nullptr);
        const auto before=network.architectureJson();
        const auto inputs=network.getExternalRaggedNetworkInputs();
        auto it=find_if(inputs.begin(),inputs.end(),[](const auto& x){return x.name=="tokens";});
        ASSERT_NE(it,inputs.end());
        vector<Event> events;
        auto placed=network.place(b,events,false);
        ASSERT_NE(placed,nullptr);
        for (auto& event:events) event.synchronize();
        auto physical=dynamic_pointer_cast<Impl::RaggedFullyConnected>(
            placed->getStampedNetwork(0).getPhysicalLayerFromApiLayer(fc->getId()));
        ASSERT_NE(physical,nullptr);
        EXPECT_TRUE(physical->isTrainingDropoutEnabled());
        ASSERT_TRUE(physical->getFeatureOutput().has_value());
        const auto oldWeights=download1X(physical->getParameterStorage("weights"));
        const auto oldBias=download1X(physical->getParameterStorage("biases"));
        map<string,Impl::Tensor> outputs;
        map<string,Event> ready;
        auto finished=placed->submitBatch(0,batch1X(it->raggedTensor,b,true),outputs,ready,false);
        finished.synchronize();
        placed->synchronize();
        ASSERT_TRUE(outputs.contains("raw_loss"));
        const auto forward=download1X(physical->getFeatureOutput().value());
        const auto& errorOutputs=physical->getErrorOutputs();
        ASSERT_FALSE(errorOutputs.empty());
        ASSERT_TRUE(errorOutputs[0].has_value());
        const auto dx=download1X(errorOutputs[0].value());
        const auto newWeights=download1X(physical->getParameterStorage("weights"));
        const auto newBias=download1X(physical->getParameterStorage("biases"));
        const auto ends=offsets1X(b);
        const float lossScale=Impl::Loss::getLossScalingFactor();
        float dw[kIn1X*kOut1X]={};
        float db[kOut1X]={};
        uint64_t kept=0, dropped=0;
        for (uint64_t t=0;t<ends.back();++t) {
            float expectedDx[kIn1X]={};
            for (uint32_t o=0;o<kOut1X;++o) {
                const float projected=projected1YC(t,o,oldWeights,oldBias);
                const float observed=forward.at(t*kOut1X+o);
                const float retained=projected/(1.f-kDropout1YC);
                const bool isDropped=std::abs(observed)<1e-5f;
                if (isDropped) {
                    ++dropped;
                    EXPECT_NEAR(observed,0.f,1e-5f);
                } else {
                    ++kept;
                    EXPECT_NEAR(observed,retained,3e-3f);
                }
                const float gate=isDropped?0.f:1.f/(1.f-kDropout1YC);
                const float upstream=2.f*observed*gate;
                db[o]+=upstream;
                for (uint32_t i=0;i<kIn1X;++i) {
                    dw[i*kOut1X+o]+=token1X(t,i)*upstream;
                    expectedDx[i]+=upstream*oldWeights[i*kOut1X+o];
                }
            }
            for (uint32_t i=0;i<kIn1X;++i)
                EXPECT_NEAR(dx.at(t*kIn1X+i)/lossScale,expectedDx[i],3e-2f);
        }
        // The mask is stochastic; require a valid gated result, not a flaky
        // specific count of retained/dropped elements for small B.
        EXPECT_EQ(kept+dropped,ends.back()*kOut1X);
        bool weightUpdated=false, biasUpdated=false;
        for (uint32_t i=0;i<kIn1X*kOut1X;++i) {
            const float change=newWeights.at(i)-oldWeights.at(i);
            weightUpdated|=std::abs(change)>1e-8f;
            if (std::abs(dw[i])>1e-3f) EXPECT_LE(change*dw[i],1e-6f);
        }
        for (uint32_t i=0;i<kOut1X;++i) {
            const float change=newBias.at(i)-oldBias.at(i);
            biasUpdated|=std::abs(change)>1e-8f;
            if (std::abs(db[i])>1e-3f) EXPECT_LE(change*db[i],1e-6f);
        }
        // Nonzero updates require at least one surviving mask element.  With
        // the fixed seed and several elements that is expected in both B runs.
        if (kept>0) { EXPECT_TRUE(weightUpdated); EXPECT_TRUE(biasUpdated); }
        EXPECT_EQ(network.architectureJson(),before);
    }
    filesystem::remove_all(archive);
}


// 1Y-D: an existing custom FC expression with one same-partition ragged
// auxiliary. The second values tensor is a real data edge, never a new
// partition token. Backward must reach both values edges.
namespace {
RaggedTensor auxiliary1YD(Network& network, const RaggedTensor& feature) {
    return RaggedNetworkInput::Builder().network(network).name("aux")
        .valuesDataType(DataType::FP32).trailingDimensions({kOut1X})
        .partition(feature).build();
}
FullyConnected buildEpilogue1YD(Network& network, const RaggedTensor& feature,
                                const RaggedTensor& auxiliary) {
    auto projection=FullyConnected::epilogueInput(DataType::FP32,DataType::FP32);
    auto aux=FullyConnected::epilogueAuxInput("aux",DataType::FP32,DataType::FP32);
    return FullyConnected::Builder().network(network).featureInput(feature)
        .numOutputFeatures(kOut1X).hasBias(true)
        .weightsDataType(DataType::FP32).computeDataType(DataType::FP32)
        .outputDataType(DataType::FP32).noActivation()
        .epilogueInput("aux",auxiliary).epilogue(projection+aux).build();
}
float auxiliaryValue1YD(uint64_t token,uint32_t channel) {
    return -0.23f+0.09f*static_cast<float>(token)+0.16f*static_cast<float>(channel);
}
Batch batch1YD(const RaggedTensor& input,const RaggedTensor& auxiliary,uint32_t b,
               bool includeLabels=false) {
    Batch result=batch1X(input,b,includeLabels);
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    const auto descriptor=auxiliary.getDescriptorForPlacement(b);
    Impl::Tensor values(cpu,descriptor.getValuesDescriptor());
    Impl::Tensor offsets(cpu,descriptor.getOffsetsDescriptor());
    const auto ends=offsets1X(b);
    for (uint64_t t=0;t<b*kM1X;++t)
        for (uint32_t c=0;c<kOut1X;++c)
            values.getMemPtr<float>()[t*kOut1X+c]=
                t<ends.back()?auxiliaryValue1YD(t,c):numeric_limits<float>::quiet_NaN();
    for (uint32_t r=0;r<=b;++r) offsets.getMemPtr<uint64_t>()[r]=ends[r];
    Impl::RowPartitionRuntime runtime(offsets,descriptor.getRowPartition());
    runtime.setHostOffsets(ends);
    result.insert("aux",Impl::RaggedTensor(values,runtime));
    return result;
}
float projected1YD(uint64_t token,uint32_t channel,const vector<float>& weights,
                   const vector<float>& bias) {
    float y=bias.at(channel)+auxiliaryValue1YD(token,channel);
    for (uint32_t i=0;i<kIn1X;++i)
        y+=token1X(token,i)*weights.at(i*kOut1X+channel);
    return y;
}
} // namespace

TEST(RaggedFullyConnectedNeutral1YD, EpilogueAuxiliarySchemaStrictAndLegacyUnchanged) {
    Network network("fc_1yd_schema");
    const auto feature=tokens1X(network);
    const auto auxiliary=auxiliary1YD(network,feature);
    const auto fc=buildEpilogue1YD(network,feature,auxiliary);
    ASSERT_TRUE(fc.getRaggedFeatureOutput().has_value());
    EXPECT_TRUE(fc.getRaggedFeatureOutput()->sharesPartitionWith(feature));
    const json record=fc.architectureJson();
    ASSERT_EQ(record.at("version"),"1.7.0");
    ASSERT_EQ(record.at("epilogue_inputs").size(),1u);
    const auto& binding=record.at("epilogue_inputs").at(0);
    EXPECT_EQ(binding.at("name"),"aux");
    const json& auxRef=binding.at("ragged_tensor");
    EXPECT_FALSE(auxRef.contains("geometry_mode"));
    EXPECT_EQ(auxRef.size(), 5u);
    EXPECT_TRUE(SegmentedPrimitiveDetail::hasCanonicalLayerRaggedReferenceFields(auxRef));
    json previouslySerializedAuxRef = auxRef;
    previouslySerializedAuxRef["geometry_mode"] = "placement_neutral";
    EXPECT_TRUE(SegmentedPrimitiveDetail::hasCanonicalLayerRaggedReferenceFields(previouslySerializedAuxRef));
    EXPECT_EQ(SegmentedPrimitiveDetail::layerRaggedReferenceGrammar(
                  previouslySerializedAuxRef, "FullyConnected"),
              SegmentedPrimitiveDetail::LayerRaggedReferenceGrammar::CANONICAL_LOGICAL);
    previouslySerializedAuxRef["geometry_mode"] = "fixed";
    EXPECT_ANY_THROW((void)SegmentedPrimitiveDetail::layerRaggedReferenceGrammar(
        previouslySerializedAuxRef, "FullyConnected"));
    EXPECT_FALSE(auxRef.contains("batch_size"));
    EXPECT_FALSE(auxRef.contains("max_total_values"));
    EXPECT_EQ(auxRef.at("max_values_per_row"),kM1X);
    EXPECT_EQ(auxRef.at("offsets").at("dimensions"),(vector<uint64_t>{1}));
    EXPECT_EQ(auxRef.at("offsets").at("id"),record.at("ragged_inputs").at(0).at("offsets").at("id"));
    for (uint32_t b:{2u,7u})
        EXPECT_EQ(fc.getOutputTensorBytes(b),b*kM1X*kOut1X*sizeof(float));
    shared_ptr<thor_file::TarReader> reader;
    auto reject=[&](json broken){ EXPECT_ANY_THROW(FullyConnected::deserialize(reader,broken,&network)); };
    json bad=record; bad["version"]="1.2.0"; reject(bad);
    bad=record; bad["version"]="1.6.0"; reject(bad);
    bad=record; bad["epilogue"]=nullptr; reject(bad);
    bad=record; bad["epilogue_inputs"]=json::array(); reject(bad);
    bad=record; bad["epilogue_inputs"][0]["name"]="__bad"; reject(bad);
    bad=record; bad["epilogue_inputs"][0]["tensor"]["dimensions"]=vector<uint64_t>{kM1X,3}; reject(bad);
    bad=record; bad["epilogue_inputs"][0]["ragged_tensor"]["values"]["id"]=feature.getValues().getOriginalId(); reject(bad);
    bad=record; bad["epilogue_inputs"][0]["ragged_tensor"]["offsets"]["id"]=feature.getValues().getOriginalId(); reject(bad);
    bad=record; bad["epilogue_inputs"][0]["ragged_tensor"]["max_values_per_row"]=kM1X+1; reject(bad);
    bad=record; bad["epilogue_inputs"][0]["ragged_tensor"]["batch_size"]=2; reject(bad);
    bad=record; bad["epilogue_inputs"][0]["ragged_tensor"]["max_total_values"]=8; reject(bad);
    bad=record; bad["output_dropout_probability"]=1.0f; reject(bad);
    bad=record; bad["use_residual"]=true; reject(bad);
    bad=record; bad["activation"]="invalid_activation_object"; reject(bad);
    bad=record; bad["inputs"][0]["data_type"]=DataType::BF16; reject(bad);
    // Genuine different partition: the builder must not accept it as an auxiliary.
    const auto wrong=RaggedNetworkInput::Builder().network(network).name("different_partition")
        .valuesDataType(DataType::FP32).offsetsDataType(DataType::UINT64)
        .trailingDimensions({kOut1X}).maxValuesPerRow(kM1X).build();
    EXPECT_ANY_THROW((void)buildEpilogue1YD(network,feature,wrong));
    // Deprecated builder B/capacity hints cannot select a second model geometry or
    // change the version of the single-auxiliary epilogue feature schema.
    Network legacyNetwork("fc_1yd_legacy");
    const auto fixed=RaggedNetworkInput::Builder().network(legacyNetwork).name("fixed")
        .valuesDataType(DataType::FP32).offsetsDataType(DataType::UINT64)
        .trailingDimensions({kIn1X}).maxValuesPerRow(4).build();
    const auto fixedAux=RaggedNetworkInput::Builder().network(legacyNetwork).name("aux")
        .valuesDataType(DataType::FP32).trailingDimensions({kOut1X}).partition(fixed).build();
    const auto withObsoleteHints = buildEpilogue1YD(legacyNetwork, fixed, fixedAux).architectureJson();
    EXPECT_EQ(withObsoleteHints.at("version"), "1.7.0");
    EXPECT_EQ(withObsoleteHints.at("ragged_inputs").at(0).at("values").at("dimensions"),
              (vector<uint64_t>{4, kIn1X}));
    EXPECT_EQ(withObsoleteHints.at("ragged_inputs").at(0).at("offsets").at("dimensions"),
              (vector<uint64_t>{1}));
}

TEST(RaggedFullyConnectedNeutral1YD, SavedEpilogueInferenceB2B7WithEmptyRows) {
    int devices=0;
    if (cudaGetDeviceCount(&devices)!=cudaSuccess || devices==0) GTEST_SKIP()<<"CUDA device unavailable";
    const auto archive=archive1X("fc_1yd_inference");
    {
        Network network("fc_1yd_inference");
        const auto feature=tokens1X(network);
        const auto auxiliary=auxiliary1YD(network,feature);
        const auto fc=buildEpilogue1YD(network,feature,auxiliary);
        (void)RaggedNetworkOutput::Builder().network(network).name("projected")
            .inputTensor(fc.getRaggedFeatureOutput().value()).build();
        network.save(archive.string(),true);
    }
    for (uint32_t b:{2u,7u}) {
        SCOPED_TRACE(::testing::Message()<<"B="<<b);
        Network network("fc_1yd_inference");
        ASSERT_NO_THROW(network.load(archive.string()));
        auto fc=find1X(network);
        ASSERT_NE(fc,nullptr);
        EXPECT_EQ(fc->architectureJson().at("version"),"1.7.0");
        const auto before=network.architectureJson();
        const auto inputs=network.getExternalRaggedNetworkInputs();
        auto tokens=find_if(inputs.begin(),inputs.end(),[](const auto& x){return x.name=="tokens";});
        auto aux=find_if(inputs.begin(),inputs.end(),[](const auto& x){return x.name=="aux";});
        ASSERT_NE(tokens,inputs.end());
        ASSERT_NE(aux,inputs.end());
        vector<Event> events;
        auto placed=network.place(b,events,true);
        ASSERT_NE(placed,nullptr);
        for (auto& e:events) e.synchronize();
        auto physical=dynamic_pointer_cast<Impl::RaggedFullyConnected>(
            placed->getStampedNetwork(0).getPhysicalLayerFromApiLayer(fc->getId()));
        ASSERT_NE(physical,nullptr);
        ASSERT_TRUE(physical->getFeatureOutput().has_value());
        EXPECT_EQ(physical->getFeatureOutput()->getDimensions(),(vector<uint64_t>{b*kM1X,kOut1X}));
        const auto weights=download1X(physical->getParameterStorage("weights"));
        const auto bias=download1X(physical->getParameterStorage("biases"));
        const auto output=placed->inferLogical(batch1YD(tokens->raggedTensor,aux->raggedTensor,b));
        ASSERT_TRUE(output.contains("projected"));
        ASSERT_TRUE(holds_alternative<Impl::RaggedTensor>(output.at("projected")));
        const auto result=get<Impl::RaggedTensor>(output.at("projected"));
        ASSERT_TRUE(result.getHostOffsetsIfAvailable().has_value());
        EXPECT_EQ(result.getHostOffsetsIfAvailable().value(),offsets1X(b));
        const float* actual=result.getValues().getMemPtr<float>();
        for (uint64_t t=0;t<offsets1X(b).back();++t)
            for (uint32_t o=0;o<kOut1X;++o)
                EXPECT_NEAR(actual[t*kOut1X+o],projected1YD(t,o,weights,bias),2e-3f);
        EXPECT_EQ(network.architectureJson(),before);
    }
    filesystem::remove_all(archive);
}

TEST(RaggedFullyConnectedNeutral1YD, SavedEpilogueTrainingB2B7RoutesAuxiliaryGradient) {
    int devices=0;
    if (cudaGetDeviceCount(&devices)!=cudaSuccess || devices==0) GTEST_SKIP()<<"CUDA device unavailable";
    const auto archive=archive1X("fc_1yd_training");
    {
        Network network("fc_1yd_training");
        (void)Sgd::Builder().network(network).initialLearningRate(.001f).decay(0.f).momentum(0.f).build();
        const auto feature=tokens1X(network);
        const auto auxiliary=auxiliary1YD(network,feature);
        const auto labels=RaggedNetworkInput::Builder().network(network).name("labels")
            .valuesDataType(DataType::FP32).trailingDimensions({kOut1X}).partition(feature).build();
        const auto firstRivet=GradientRivet::Builder().network(network).tensor(feature.getValues()).build();
        const auto auxRivet=GradientRivet::Builder().network(network).tensor(auxiliary.getValues()).build();
        const auto fc=buildEpilogue1YD(network,feature.withValues(firstRivet.getFeatureOutput().value()),
                                      auxiliary.withValues(auxRivet.getFeatureOutput().value()));
        auto prediction=Impl::Expression::input("predictions",DataType::FP32,DataType::FP32);
        auto label=Impl::Expression::input("labels",DataType::FP32,DataType::FP32);
        auto diff=prediction-label;
        auto loss=Impl::DynamicExpression::fromExpressionDefinition(
            Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({{
                "loss",(diff*diff).withOutputDType(DataType::FP32)}})));
        auto grad=Impl::DynamicExpression::fromExpressionDefinition(
            Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({{
                "predictions_grad",(diff*Impl::Expression(2.0*Impl::Loss::getLossScalingFactor()))
                    .withOutputDType(DataType::FP32)}})));
        const auto raw=RaggedCustomLoss::Builder().network(network)
            .predictions(fc.getRaggedFeatureOutput().value()).labels(labels)
            .lossExpression(loss).gradientExpression(grad).build();
        (void)NetworkOutput::Builder().network(network).name("raw_loss")
            .inputTensor(raw.getRaggedRawLoss().getValues()).dataType(DataType::FP32).build();
        network.save(archive.string(),true);
    }
    for (uint32_t b:{2u,7u}) {
        SCOPED_TRACE(::testing::Message()<<"B="<<b);
        Network network("fc_1yd_training");
        ASSERT_NO_THROW(network.load(archive.string()));
        const auto inputs=network.getExternalRaggedNetworkInputs();
        auto tokens=find_if(inputs.begin(),inputs.end(),[](const auto& x){return x.name=="tokens";});
        auto aux=find_if(inputs.begin(),inputs.end(),[](const auto& x){return x.name=="aux";});
        ASSERT_NE(tokens,inputs.end());
        ASSERT_NE(aux,inputs.end());
        auto fc=find1X(network);
        ASSERT_NE(fc,nullptr);
        const auto before=network.architectureJson();
        vector<Event> events;
        auto placed=network.place(b,events,false);
        ASSERT_NE(placed,nullptr);
        for (auto& e:events) e.synchronize();
        auto physical=dynamic_pointer_cast<Impl::RaggedFullyConnected>(
            placed->getStampedNetwork(0).getPhysicalLayerFromApiLayer(fc->getId()));
        ASSERT_NE(physical,nullptr);
        const auto oldW=download1X(physical->getParameterStorage("weights"));
        const auto oldB=download1X(physical->getParameterStorage("biases"));
        map<string,Impl::Tensor> outputs;
        map<string,Event> ready;
        auto finished=placed->submitBatch(0,batch1YD(tokens->raggedTensor,aux->raggedTensor,b,true),
                                         outputs,ready,false);
        finished.synchronize();
        placed->synchronize();
        ASSERT_TRUE(outputs.contains("raw_loss"));
        const auto& error=physical->getErrorOutputs();
        ASSERT_GE(error.size(),3u);
        ASSERT_TRUE(error[0].has_value());
        ASSERT_TRUE(error[2].has_value());
        const auto dx=download1X(error[0].value());
        const auto dAux=download1X(error[2].value());
        const auto newW=download1X(physical->getParameterStorage("weights"));
        const auto newB=download1X(physical->getParameterStorage("biases"));
        const auto ends=offsets1X(b);
        const float scale=Impl::Loss::getLossScalingFactor();
        float dw[kIn1X*kOut1X]={};
        float db[kOut1X]={};
        for (uint64_t t=0;t<ends.back();++t) {
            for (uint32_t o=0;o<kOut1X;++o) {
                const float delta=2.f*projected1YD(t,o,oldW,oldB);
                EXPECT_NEAR(dAux.at(t*kOut1X+o)/scale,delta,2e-2f);
                db[o]+=delta;
                for (uint32_t i=0;i<kIn1X;++i) dw[i*kOut1X+o]+=token1X(t,i)*delta;
            }
            for (uint32_t i=0;i<kIn1X;++i) {
                float expected=0.f;
                for (uint32_t o=0;o<kOut1X;++o)
                    expected+=2.f*projected1YD(t,o,oldW,oldB)*oldW[i*kOut1X+o];
                EXPECT_NEAR(dx.at(t*kIn1X+i)/scale,expected,2e-2f);
            }
        }
        bool weightUpdated=false,biasUpdated=false;
        for (uint32_t i=0;i<kIn1X*kOut1X;++i) {
            const float change=newW[i]-oldW[i];
            weightUpdated|=fabs(change)>1e-8f;
            if (fabs(dw[i])>1e-3f) EXPECT_LE(change*dw[i],1e-7f);
        }
        for (uint32_t o=0;o<kOut1X;++o) {
            const float change=newB[o]-oldB[o];
            biasUpdated|=fabs(change)>1e-8f;
            if (fabs(db[o])>1e-3f) EXPECT_LE(change*db[o],1e-7f);
        }
        EXPECT_TRUE(weightUpdated);
        EXPECT_TRUE(biasUpdated);
        EXPECT_EQ(network.architectureJson(),before);
    }
    filesystem::remove_all(archive);
}

TEST(RaggedFullyConnectedNeutralMultiApplication2B, IndependentPartitionsShareForwardAndBackwardParametersAtB2AndB7) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        GTEST_SKIP() << "CUDA device unavailable";

    constexpr uint64_t m0 = 4, m1 = 6, inFeatures = 3, outFeatures = 2;
    Network network("neutral_multi_fc_shared_training");
    (void)Sgd::Builder().network(network).initialLearningRate(.001f).decay(0.f).momentum(0.f).build();

    auto source = [&](const string& name, uint64_t m) {
        return RaggedNetworkInput::Builder().network(network).name(name)
            .valuesDataType(DataType::FP32).offsetsDataType(DataType::UINT64)
            .trailingDimensions({inFeatures}).maxValuesPerRow(m).build();
    };
    const RaggedTensor tokens0 = source("tokens0", m0);
    const RaggedTensor tokens1 = source("tokens1", m1);
    ASSERT_FALSE(tokens0.sharesPartitionWith(tokens1));
    const RaggedTensor labels0 = RaggedNetworkInput::Builder().network(network).name("labels0")
        .valuesDataType(DataType::FP32).trailingDimensions({outFeatures}).partition(tokens0).build();
    const RaggedTensor labels1 = RaggedNetworkInput::Builder().network(network).name("labels1")
        .valuesDataType(DataType::FP32).trailingDimensions({outFeatures}).partition(tokens1).build();
    auto rivet0 = GradientRivet::Builder().network(network).tensor(tokens0.getValues()).build();
    auto rivet1 = GradientRivet::Builder().network(network).tensor(tokens1.getValues()).build();
    const auto fc = FullyConnected::Builder().network(network)
        .featureInput(tokens0.withValues(rivet0.getFeatureOutput().value()))
        .featureInput(tokens1.withValues(rivet1.getFeatureOutput().value()))
        .numOutputFeatures(outFeatures).hasBias(true)
        .weightsDataType(DataType::FP32).computeDataType(DataType::FP32)
        .outputDataType(DataType::FP32).noActivation().build();
    ASSERT_EQ(fc.getParameters().size(), 2u);
    EXPECT_EQ(fc.architectureJson().at("version"), "1.3.0");
    EXPECT_EQ(fc.getOutputTensorBytes(2), 2 * (m0 + m1) * outFeatures * sizeof(float));
    EXPECT_EQ(fc.getOutputTensorBytes(7), 7 * (m0 + m1) * outFeatures * sizeof(float));

    auto prediction = Impl::Expression::input("predictions", DataType::FP32, DataType::FP32);
    auto label = Impl::Expression::input("labels", DataType::FP32, DataType::FP32);
    auto difference = prediction - label;
    auto loss = Impl::DynamicExpression::fromExpressionDefinition(
        Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({{
            "loss", (difference * difference).withOutputDType(DataType::FP32)}})));
    auto gradient = Impl::DynamicExpression::fromExpressionDefinition(
        Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({{
            "predictions_grad", (difference * Impl::Expression(2.0 * Impl::Loss::getLossScalingFactor()))
                .withOutputDType(DataType::FP32)}})));
    for (uint32_t app = 0; app < 2; ++app) {
        const auto output = fc.getRaggedFeatureOutput(app);
        ASSERT_TRUE(output.has_value());
        EXPECT_TRUE(output->sharesPartitionWith(app == 0 ? tokens0 : tokens1));
        EXPECT_EQ(output->getMaxValuesPerRow(), app == 0 ? m0 : m1);
        const auto raw = RaggedCustomLoss::Builder().network(network)
            .predictions(output.value()).labels(app == 0 ? labels0 : labels1)
            .lossExpression(loss).gradientExpression(gradient).build();
        (void)NetworkOutput::Builder().network(network).name("loss" + to_string(app))
            .inputTensor(raw.getRaggedRawLoss().getValues()).dataType(DataType::FP32).build();
    }
    const json modelBeforePlacement = network.architectureJson();
    // Network::connect() resolves the configured default optimizer into the
    // trainable parameter specifications once, independently of placement B.
    // Validate that this is the only allowed initial normalization.
    std::string fcLayerPath;
    for (size_t i = 0; i < modelBeforePlacement.at("layers").size(); ++i) {
        if (modelBeforePlacement.at("layers").at(i).value("layer_type", std::string{}) == "fully_connected") {
            ASSERT_TRUE(fcLayerPath.empty());
            fcLayerPath = "/layers/" + std::to_string(i);
        }
    }
    ASSERT_FALSE(fcLayerPath.empty());
    const std::string weightsOptimizerPath = fcLayerPath + "/parameters/weights/optimizer_override";
    const std::string biasesOptimizerPath = fcLayerPath + "/parameters/biases/optimizer_override";
    json connectedArchitecture;

    auto offsetsFor = [](uint32_t app, uint32_t b) {
        vector<uint64_t> offsets(b + 1, 0);
        for (uint32_t row = 0; row < b; ++row) {
            const uint64_t length = app == 0 ? (row % 3 == 0 ? 0 : row % 3 == 1 ? 2 : 1)
                                             : (row % 4 == 0 ? 5 : row % 4 == 1 ? 0 : row % 4 == 2 ? 4 : 2);
            offsets[row + 1] = offsets[row] + length;
        }
        return offsets;
    };
    auto tokenValue = [](uint32_t app, uint64_t t, uint32_t c) {
        return app == 0 ? (.2f + .12f * t + .2f * c)
                        : (-.7f + .17f * t - .24f * c);
    };
    for (uint32_t b : {2u, 7u}) {
        SCOPED_TRACE(::testing::Message() << "B=" << b);
        vector<Event> events;
        auto placed = network.place(b, events, false);
        ASSERT_NE(placed, nullptr);
        for (auto& event : events) event.synchronize();
        const json architectureAfterPlacement = network.architectureJson();
        if (b == 2) {
            const json changes = json::diff(modelBeforePlacement, architectureAfterPlacement);
            ASSERT_EQ(changes.size(), 2u) << changes.dump(2);
            bool sawWeightsOptimizer = false, sawBiasesOptimizer = false;
            for (const auto& change : changes) {
                EXPECT_EQ(change.at("op"), "add");
                const auto path = change.at("path").get<std::string>();
                if (path == weightsOptimizerPath) sawWeightsOptimizer = true;
                else if (path == biasesOptimizerPath) sawBiasesOptimizer = true;
                else ADD_FAILURE() << "Unexpected architecture mutation: " << change.dump();
            }
            EXPECT_TRUE(sawWeightsOptimizer);
            EXPECT_TRUE(sawBiasesOptimizer);
            connectedArchitecture = architectureAfterPlacement;
        } else {
            // The same logical model has identical connected architecture for B=2 and B=7.
            EXPECT_EQ(architectureAfterPlacement, connectedArchitecture);
        }
        auto physical = dynamic_pointer_cast<Impl::RaggedFullyConnected>(
            placed->getStampedNetwork(0).getPhysicalLayerFromApiLayer(fc.getId()));
        ASSERT_NE(physical, nullptr);
        EXPECT_EQ(physical->getFeatureOutputs().size(), 2u);
        const auto initialWeights = download1X(physical->getParameterStorage("weights"));
        const auto initialBiases = download1X(physical->getParameterStorage("biases"));
        ASSERT_EQ(initialWeights.size(), inFeatures * outFeatures);
        ASSERT_EQ(initialBiases.size(), outFeatures);

        Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
        Batch batch;
        vector<vector<uint64_t>> offsetsByApp;
        for (uint32_t app = 0; app < 2; ++app) {
            const RaggedTensor& tokens = app == 0 ? tokens0 : tokens1;
            const auto offsets = offsetsFor(app, b);
            offsetsByApp.push_back(offsets);
            const auto descriptor = tokens.getDescriptorForPlacement(b);
            const uint64_t m = app == 0 ? m0 : m1;
            Impl::Tensor values(cpu, descriptor.getValuesDescriptor());
            Impl::Tensor boundaries(cpu, descriptor.getOffsetsDescriptor());
            for (uint64_t t = 0; t < b * m; ++t)
                for (uint32_t c = 0; c < inFeatures; ++c)
                    values.getMemPtr<float>()[t * inFeatures + c] = t < offsets.back()
                        ? tokenValue(app, t, c) : numeric_limits<float>::quiet_NaN();
            for (uint32_t row = 0; row <= b; ++row)
                boundaries.getMemPtr<uint64_t>()[row] = offsets[row];
            Impl::RowPartitionRuntime runtime(boundaries, descriptor.getRowPartition());
            runtime.setHostOffsets(offsets);
            batch.insert("tokens" + to_string(app), Impl::RaggedTensor(values, runtime));
            Impl::Tensor labelValues(cpu, Impl::TensorDescriptor(DataType::FP32, {b * m, outFeatures}));
            std::fill_n(labelValues.getMemPtr<float>(), b * m * outFeatures, 0.0f);
            batch.insert("labels" + to_string(app), Impl::RaggedTensor(labelValues, runtime));
        }
        batch.setValidExampleCount(b);
        map<string, Impl::Tensor> outputs;
        map<string, Event> ready;
        auto finished = placed->submitBatch(0, batch, outputs, ready, false);
        finished.synchronize();
        placed->synchronize();
        ASSERT_TRUE(outputs.contains("loss0"));
        ASSERT_TRUE(outputs.contains("loss1"));
        const auto featureOutputs = physical->getFeatureOutputs();
        const auto inputGradients = physical->getErrorOutputs();
        ASSERT_EQ(featureOutputs.size(), 2u);
        ASSERT_EQ(inputGradients.size(), 4u);
        vector<float> expectedWeightGradient(inFeatures * outFeatures, 0.0f);
        vector<float> expectedBiasGradient(outFeatures, 0.0f);
        const float lossScale = Impl::Loss::getLossScalingFactor();
        for (uint32_t app = 0; app < 2; ++app) {
            ASSERT_TRUE(featureOutputs[app].has_value());
            ASSERT_TRUE(inputGradients[app * 2].has_value());
            const uint64_t m = app == 0 ? m0 : m1;
            EXPECT_EQ(featureOutputs[app]->getDimensions(), (vector<uint64_t>{b * m, outFeatures}));
            const auto result = download1X(featureOutputs[app].value());
            const auto dx = download1X(inputGradients[app * 2].value());
            for (uint64_t t = 0; t < offsetsByApp[app].back(); ++t) {
                for (uint32_t o = 0; o < outFeatures; ++o) {
                    float expected = initialBiases[o];
                    for (uint32_t i = 0; i < inFeatures; ++i)
                        expected += tokenValue(app, t, i) * initialWeights[i * outFeatures + o];
                    EXPECT_NEAR(result[t * outFeatures + o], expected, 3e-3f);
                    expectedBiasGradient[o] += 2.f * expected;
                    for (uint32_t i = 0; i < inFeatures; ++i)
                        expectedWeightGradient[i * outFeatures + o] +=
                            2.f * expected * tokenValue(app, t, i);
                }
                for (uint32_t i = 0; i < inFeatures; ++i) {
                    float expected = 0.f;
                    for (uint32_t o = 0; o < outFeatures; ++o) {
                        float projection = initialBiases[o];
                        for (uint32_t j = 0; j < inFeatures; ++j)
                            projection += tokenValue(app, t, j) * initialWeights[j * outFeatures + o];
                        expected += 2.f * projection * initialWeights[i * outFeatures + o];
                    }
                    EXPECT_NEAR(dx[t * inFeatures + i] / lossScale, expected, 3e-2f);
                }
            }
        }
        const auto updatedWeights = download1X(physical->getParameterStorage("weights"));
        const auto updatedBiases = download1X(physical->getParameterStorage("biases"));
        ASSERT_EQ(updatedWeights.size(), expectedWeightGradient.size());
        ASSERT_EQ(updatedBiases.size(), expectedBiasGradient.size());
        bool changed = false;
        double dot = 0.0, deltaNormSquared = 0.0, referenceNormSquared = 0.0;
        for (size_t i = 0; i < updatedWeights.size(); ++i) {
            const float delta = updatedWeights[i] - initialWeights[i];
            changed |= std::fabs(delta) > 1e-8f;
            if (std::fabs(expectedWeightGradient[i]) > 1e-3f)
                EXPECT_LE(delta * expectedWeightGradient[i], 1e-6f);
            dot += static_cast<double>(delta) * expectedWeightGradient[i];
            deltaNormSquared += static_cast<double>(delta) * delta;
            referenceNormSquared += static_cast<double>(expectedWeightGradient[i]) * expectedWeightGradient[i];
        }
        EXPECT_TRUE(changed);
        ASSERT_GT(deltaNormSquared, 0.0);
        ASSERT_GT(referenceNormSquared, 0.0);
        // A single shared update must follow the SUM of both applications' gradients,
        // not either application's gradient in isolation. The optimizer may scale
        // the combined gradient, so compare its direction rather than its magnitude.
        EXPECT_LT(dot / std::sqrt(deltaNormSquared * referenceNormSquared), -0.95);
        bool biasChanged = false;
        for (size_t i = 0; i < updatedBiases.size(); ++i) {
            const float delta = updatedBiases[i] - initialBiases[i];
            biasChanged |= std::fabs(delta) > 1e-8f;
            if (std::fabs(expectedBiasGradient[i]) > 1e-3f)
                EXPECT_LE(delta * expectedBiasGradient[i], 1e-6f);
        }
        EXPECT_TRUE(biasChanged);
        // Neither numerical training nor optimizer updates may mutate logical topology.
        EXPECT_EQ(network.architectureJson(), architectureAfterPlacement);
    }
}

TEST(RaggedFullyConnectedNeutralMultiApplication2C, SavedSharedProjectionRestoresAtB2AndB7) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        GTEST_SKIP() << "CUDA device unavailable";

    constexpr uint64_t m0 = 4, m1 = 6, inputFeatures = 3, outputFeatures = 2;
    const auto archive = archive1X("fc_multi_2c_saved");
    vector<float> savedWeights, savedBiases;
    {
        Network network("fc_multi_2c_saved");
        auto makeInput = [&](const string& name, uint64_t m) {
            return RaggedNetworkInput::Builder().network(network).name(name)
                .valuesDataType(DataType::FP32).offsetsDataType(DataType::UINT64)
                .trailingDimensions({inputFeatures}).maxValuesPerRow(m).build();
        };
        const RaggedTensor first = makeInput("tokens0", m0);
        const RaggedTensor second = makeInput("tokens1", m1);
        ASSERT_FALSE(first.sharesPartitionWith(second));
        // Different M is legal; different input feature width is not. Keep the
        // rejected input in a separate graph so the executable model is clean.
        {
            Network invalidNetwork("fc_multi_2c_incompatible_width");
            const auto valid = RaggedNetworkInput::Builder().network(invalidNetwork)
                .name("valid").valuesDataType(DataType::FP32).offsetsDataType(DataType::UINT64)
                .trailingDimensions({inputFeatures}).maxValuesPerRow(m0).build();
            const auto incompatibleWidth = RaggedNetworkInput::Builder().network(invalidNetwork)
                .name("incompatible_width").valuesDataType(DataType::FP32).offsetsDataType(DataType::UINT64)
                .trailingDimensions({inputFeatures + 1}).maxValuesPerRow(m1).build();
            EXPECT_ANY_THROW((void)FullyConnected::Builder().network(invalidNetwork)
                .featureInput(valid).featureInput(incompatibleWidth));
        }
        const auto fc = FullyConnected::Builder().network(network)
            .featureInput(first).featureInput(second).numOutputFeatures(outputFeatures)
            .hasBias(true).weightsDataType(DataType::FP32)
            .computeDataType(DataType::FP32).outputDataType(DataType::FP32)
            .noActivation().build();
        for (uint32_t i = 0; i < 2; ++i) {
            auto projected = fc.getRaggedFeatureOutput(i);
            ASSERT_TRUE(projected.has_value());
            (void)RaggedNetworkOutput::Builder().network(network)
                .name("projected" + to_string(i)).inputTensor(projected.value()).build();
        }
        const auto& fcJson = fc.architectureJson();
        ASSERT_EQ(fcJson.at("version"), "1.3.0");
        ASSERT_EQ(fcJson.at("ragged_inputs").size(), 2u);
        ASSERT_EQ(fcJson.at("ragged_outputs").size(), 2u);
        for (uint32_t i = 0; i < 2; ++i) {
            const auto& input = fcJson.at("ragged_inputs").at(i);
            const auto& output = fcJson.at("ragged_outputs").at(i);
            EXPECT_FALSE(input.contains("batch_size"));
            EXPECT_FALSE(input.contains("max_total_values"));
            EXPECT_FALSE(output.contains("batch_size"));
            EXPECT_FALSE(output.contains("max_total_values"));
            EXPECT_EQ(input.at("max_values_per_row"), i == 0 ? m0 : m1);
            EXPECT_EQ(output.at("max_values_per_row"), i == 0 ? m0 : m1);
            EXPECT_EQ(input.at("offsets").at("id"), output.at("offsets").at("id"));
        }
        EXPECT_NE(fcJson.at("ragged_inputs").at(0).at("offsets").at("id"),
                  fcJson.at("ragged_inputs").at(1).at("offsets").at("id"));

        // The layer reader must reject malformed per-application identities and
        // geometry rather than silently merging independent partitions.
        shared_ptr<thor_file::TarReader> reader;
        auto reject = [&](json broken) {
            EXPECT_ANY_THROW(FullyConnected::deserialize(reader, broken, &network));
        };
        json broken = fcJson;
        broken["ragged_inputs"][1]["max_values_per_row"] = 7;
        reject(broken);
        broken = fcJson;
        broken["ragged_outputs"][1]["offsets"]["id"] =
            fcJson.at("ragged_inputs").at(0).at("offsets").at("id");
        reject(broken);
        broken = fcJson;
        broken["ragged_inputs"][1]["values"]["id"] =
            fcJson.at("ragged_inputs").at(0).at("values").at("id");
        reject(broken);
        broken = fcJson;
        broken["outputs"].erase(1);
        reject(broken);
        broken = fcJson;
        broken["parameters"]["weights"]["shape"] = vector<uint64_t>{3, 3};
        reject(broken);

        vector<Event> events;
        auto placed = network.place(2, events, true);
        ASSERT_NE(placed, nullptr);
        for (auto& event : events) event.synchronize();
        auto physical = dynamic_pointer_cast<Impl::RaggedFullyConnected>(
            placed->getStampedNetwork(0).getPhysicalLayerFromApiLayer(fc.getId()));
        ASSERT_NE(physical, nullptr);
        savedWeights = download1X(physical->getParameterStorage("weights"));
        savedBiases = download1X(physical->getParameterStorage("biases"));
        ASSERT_EQ(savedWeights.size(), inputFeatures * outputFeatures);
        ASSERT_EQ(savedBiases.size(), outputFeatures);
        placed->save(archive.string(), true, false);
    }

    auto offsetsFor = [](uint32_t application, uint32_t b) {
        vector<uint64_t> offsets(b + 1, 0);
        for (uint32_t r = 0; r < b; ++r) {
            const uint64_t length = application == 0 ? (r % 2 == 0 ? 0 : 2)
                                                       : (r % 3 == 0 ? 5 : r % 3 == 1 ? 1 : 0);
            offsets[r + 1] = offsets[r] + length;
        }
        return offsets;
    };
    auto inputValue = [](uint32_t application, uint64_t token, uint32_t feature) {
        return application == 0 ? (.25f + .15f * token + .3f * feature)
                                : (-.4f + .22f * token - .2f * feature);
    };
    for (uint32_t b : {2u, 7u}) {
        SCOPED_TRACE(::testing::Message() << "B=" << b);
        Network loaded("fc_multi_2c_saved");
        ASSERT_NO_THROW(loaded.load(archive.string()));
        auto loadedFc = find1X(loaded);
        ASSERT_NE(loadedFc, nullptr);
        const auto fcJson = loadedFc->architectureJson();
        ASSERT_EQ(fcJson.at("inputs").size(), 2u);
        ASSERT_EQ(fcJson.at("ragged_inputs").size(), 2u);
        ASSERT_EQ(fcJson.at("ragged_outputs").size(), 2u);
        ASSERT_TRUE(loadedFc->getRaggedFeatureInput(0).has_value());
        ASSERT_TRUE(loadedFc->getRaggedFeatureInput(1).has_value());
        ASSERT_TRUE(loadedFc->getRaggedFeatureOutput(0).has_value());
        ASSERT_TRUE(loadedFc->getRaggedFeatureOutput(1).has_value());
        const auto first = loadedFc->getRaggedFeatureInput(0).value();
        const auto second = loadedFc->getRaggedFeatureInput(1).value();
        EXPECT_FALSE(first.sharesPartitionWith(second));
        EXPECT_TRUE(loadedFc->getRaggedFeatureOutput(0)->sharesPartitionWith(first));
        EXPECT_TRUE(loadedFc->getRaggedFeatureOutput(1)->sharesPartitionWith(second));
        EXPECT_EQ(first.getValues().getDimensions(), (vector<uint64_t>{m0, inputFeatures}));
        EXPECT_EQ(second.getValues().getDimensions(), (vector<uint64_t>{m1, inputFeatures}));
        vector<Event> events;
        auto placed = loaded.place(b, events, true);
        ASSERT_NE(placed, nullptr);
        for (auto& event : events) event.synchronize();
        auto physical = dynamic_pointer_cast<Impl::RaggedFullyConnected>(
            placed->getStampedNetwork(0).getPhysicalLayerFromApiLayer(loadedFc->getId()));
        ASSERT_NE(physical, nullptr);
        EXPECT_EQ(download1X(physical->getParameterStorage("weights")), savedWeights);
        EXPECT_EQ(download1X(physical->getParameterStorage("biases")), savedBiases);
        EXPECT_EQ(physical->getFeatureOutputs().size(), 2u);

        Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
        Batch batch;
        vector<vector<uint64_t>> expectedOffsets;
        for (uint32_t app = 0; app < 2; ++app) {
            const auto& logical = app == 0 ? first : second;
            const auto descriptor = logical.getDescriptorForPlacement(b);
            const uint64_t m = app == 0 ? m0 : m1;
            const auto offsets = offsetsFor(app, b);
            expectedOffsets.push_back(offsets);
            Impl::Tensor values(cpu, descriptor.getValuesDescriptor());
            Impl::Tensor boundaries(cpu, descriptor.getOffsetsDescriptor());
            for (uint64_t t = 0; t < b * m; ++t)
                for (uint32_t i = 0; i < inputFeatures; ++i)
                    values.getMemPtr<float>()[t * inputFeatures + i] = t < offsets.back()
                        ? inputValue(app, t, i) : numeric_limits<float>::quiet_NaN();
            for (uint32_t i = 0; i <= b; ++i) boundaries.getMemPtr<uint64_t>()[i] = offsets[i];
            Impl::RowPartitionRuntime runtime(boundaries, descriptor.getRowPartition());
            runtime.setHostOffsets(offsets);
            batch.insert("tokens" + to_string(app), Impl::RaggedTensor(values, runtime));
        }
        batch.setValidExampleCount(b);
        const auto results = placed->inferLogical(batch);
        for (uint32_t app = 0; app < 2; ++app) {
            const auto name = "projected" + to_string(app);
            const uint64_t m = app == 0 ? m0 : m1;
            ASSERT_TRUE(results.contains(name));
            ASSERT_TRUE(holds_alternative<Impl::RaggedTensor>(results.at(name)));
            const auto& result = get<Impl::RaggedTensor>(results.at(name));
            ASSERT_EQ(result.getValues().getDimensions(), (vector<uint64_t>{b * m, outputFeatures}));
            const auto offsets = result.getHostOffsetsIfAvailable();
            ASSERT_TRUE(offsets.has_value());
            EXPECT_EQ(offsets.value(), expectedOffsets[app]);
            const float* observed = result.getValues().getMemPtr<float>();
            for (uint64_t t = 0; t < offsets->back(); ++t)
                for (uint32_t o = 0; o < outputFeatures; ++o) {
                    float reference = savedBiases[o];
                    for (uint32_t i = 0; i < inputFeatures; ++i)
                        reference += inputValue(app, t, i) * savedWeights[i * outputFeatures + o];
                    EXPECT_NEAR(observed[t * outputFeatures + o], reference, 3e-3f);
                }
        }
    }
    filesystem::remove_all(archive);
}


// One real archived-network load checks the shared ownership rule for both
// parameterized readers; their existing suites cover numerical execution.
TEST(RaggedFullyConnectedNeutral1X, HistoricalRmsNormAndProjectionRecoverLogicalMAtTwoPlacements) {
    const string name = "rms_fc_legacy_logical_geometry";
    const auto archive = archive1X("rms_fc_legacy_geometry");
    filesystem::remove_all(archive);
    Network network(name);
    const RaggedTensor input = tokens1X(network);
    const RMSNorm norm = RMSNorm::Builder().network(network).featureInput(input)
        .normalizedShape({kIn1X}).epsilon(1e-5).parameterDataType(DataType::FP32).build();
    const FullyConnected projection = build1X(network, norm.getRaggedFeatureOutput().value(), true);
    (void)RaggedNetworkOutput::Builder().network(network).name("projected")
        .inputTensor(projection.getRaggedFeatureOutput().value()).build();

    json record = network.architectureJson();
    uint32_t migratedReaders = 0;
    // Simulate an archived 1.0.0 layer with physical dimensions while retaining
    // authoritative semantic M in the network's ragged boundary declaration.
    auto legacyReference = [](json& ref) {
        ref.erase("layer_ragged_reference_version");
        ref.erase("geometry_mode");
        ref.erase("max_values_per_row");
        ref["batch_size"] = 31;             // Deliberately inconsistent hints
        ref["max_total_values"] = 929;
        ref["values"]["dimensions"][0] = 3 * kM1X;
        ref["offsets"]["dimensions"][0] = 4;
    };
    for (json& layer : record.at("layers")) {
        if (layer.at("layer_type") == "rms_norm" ||
            layer.at("layer_type") == "fully_connected") {
            layer["version"] = "1.0.0";
            for (json& ref : layer.at("ragged_inputs")) legacyReference(ref);
            for (json& ref : layer.at("ragged_outputs")) legacyReference(ref);
            for (json& ref : layer.at("inputs")) ref["dimensions"][0] = 3 * kM1X;
            for (json& ref : layer.at("outputs")) ref["dimensions"][0] = 3 * kM1X;
            ++migratedReaders;
        } else if (layer.at("layer_type") == "network_input") {
            const string inputName = layer.at("name").get<string>();
            if (inputName != "tokens.values" && inputName != "__thor_row_partition.tokens") continue;
            const uint64_t physical = inputName == "tokens.values" ? 3 * kM1X : 4;
            layer["dimensions"][0] = physical;
            for (const char* key : {"feature_input", "feature_output"})
                layer[key]["dimensions"][0] = physical;
        }
    }
    ASSERT_EQ(migratedReaders, 2u);
    const string encoded = record.dump(4);
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor bytes(cpu, Impl::TensorDescriptor(DataType::UINT8, {encoded.size()}));
    std::memcpy(bytes.getMemPtr<void>(), encoded.data(), encoded.size());
    thor_file::TarWriter writer(name);
    writer.addArchiveFile(name + ".thor.json", bytes);
    writer.createArchive(archive, true);

    for (uint32_t b : {2u, 7u}) {
        SCOPED_TRACE(::testing::Message() << "B=" << b);
        Network loaded(name);
        ASSERT_NO_THROW(loaded.load(archive.string()));
        shared_ptr<RMSNorm> loadedNorm;
        shared_ptr<FullyConnected> loadedFc;
        for (uint32_t i = 0; i < loaded.getNumLayers(); ++i) {
            const auto layer = loaded.getLayer(i);
            if (auto rms = dynamic_pointer_cast<RMSNorm>(layer)) loadedNorm = rms;
            if (auto fc = dynamic_pointer_cast<FullyConnected>(layer)) loadedFc = fc;
        }
        ASSERT_NE(loadedNorm, nullptr);
        ASSERT_NE(loadedFc, nullptr);
        const RaggedTensor normOutput = loadedNorm->getRaggedFeatureOutput().value();
        const RaggedTensor fcInput = loadedFc->getRaggedFeatureInput().value();
        const RaggedTensor fcOutput = loadedFc->getRaggedFeatureOutput().value();
        EXPECT_EQ(normOutput.getValuesDimensions(), (vector<uint64_t>{kM1X, kIn1X}));
        EXPECT_EQ(normOutput.getOffsetsDimensions(), (vector<uint64_t>{1}));
        EXPECT_EQ(fcInput.getValues().getOriginalId(), normOutput.getValues().getOriginalId());
        EXPECT_TRUE(fcInput.sharesPartitionWith(normOutput));
        EXPECT_EQ(fcOutput.getValuesDimensions(), (vector<uint64_t>{kM1X, kOut1X}));
        EXPECT_EQ(fcOutput.getDescriptorForPlacement(b).getValuesDimensions(),
                  (vector<uint64_t>{b * kM1X, kOut1X}));
        vector<Event> events;
        auto placed = loaded.place(b, events, true);
        ASSERT_NE(placed, nullptr);
        for (auto& event : events) event.synchronize();
        EXPECT_FALSE(loadedNorm->architectureJson().at("ragged_inputs").at(0).contains("batch_size"));
        EXPECT_FALSE(loadedFc->architectureJson().at("ragged_outputs").at(0).contains("max_total_values"));
    }
    filesystem::remove_all(archive);
}

// FC-2A extends already-serialized optional post-ops rather than inventing
// new architecture versions. Keep one positive acceptance gate for each newly
// supported composition and for arbitrary epilogue auxiliary arity.
TEST(RaggedFullyConnectedNeutralFC2A, ActivationDropoutResidualReuseExistingSchemasAndPlaceAtB2B7) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        GTEST_SKIP() << "CUDA device unavailable";
    for (bool withResidual : {false, true}) {
        for (float dropoutProbability : {0.0f, 0.2f}) {
            SCOPED_TRACE(::testing::Message() << "withResidual=" << withResidual
                         << ", dropoutProbability=" << dropoutProbability);
            const string name = "fc2a_activation_" + to_string(withResidual) + "_" +
                                to_string(static_cast<int>(dropoutProbability > 0.0f));
            const auto archive = archive1X(name);
            filesystem::remove_all(archive);
            const string expectedVersion = withResidual ? "1.5.0" :
                                           dropoutProbability > 0.0f ? "1.6.0" : "1.4.0";
            {
                Network network(name);
                const auto feature = tokens1X(network);
                FullyConnected::Builder builder;
                builder.network(network).featureInput(feature)
                    .numOutputFeatures(kOut1X).hasBias(true)
                    .weightsDataType(DataType::FP32).computeDataType(DataType::FP32)
                    .outputDataType(DataType::FP32).activation(Sigmoid::Builder().build())
                    .outputDropoutProbability(dropoutProbability).outputDropoutSeed(1234);
                if (withResidual) builder.residualInput(residual1YB(network, feature));
                const auto fc = builder.build();
                const json record = fc.architectureJson();
                EXPECT_EQ(record.at("version"), expectedVersion);
                EXPECT_TRUE(record.at("activation").is_object());
                EXPECT_FLOAT_EQ(record.at("output_dropout_probability").get<float>(), dropoutProbability);
                (void)RaggedNetworkOutput::Builder().network(network).name("projected")
                    .inputTensor(fc.getRaggedFeatureOutput().value()).build();
                network.save(archive.string(), true);
            }
            for (uint32_t b : {2u, 7u}) {
                SCOPED_TRACE(::testing::Message() << "B=" << b);
                Network loaded(name);
                ASSERT_NO_THROW(loaded.load(archive.string()));
                const auto fc = find1X(loaded);
                ASSERT_NE(fc, nullptr);
                EXPECT_EQ(fc->architectureJson().at("version"), expectedVersion);
                EXPECT_TRUE(fc->architectureJson().at("activation").is_object());
                vector<Event> events;
                auto placed = loaded.place(b, events, true);
                ASSERT_NE(placed, nullptr);
                for (auto& event : events) event.synchronize();
            }
            filesystem::remove_all(archive);
        }
    }
}

TEST(RaggedFullyConnectedNeutralFC2A, EpilogueZeroAndTwoAuxiliariesRemainVersion17AtB2B7) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        GTEST_SKIP() << "CUDA device unavailable";
    for (uint32_t auxCount : {0u, 2u}) {
        SCOPED_TRACE(::testing::Message() << "auxCount=" << auxCount);
        const string name = "fc2a_epilogue_" + to_string(auxCount);
        const auto archive = archive1X(name);
        filesystem::remove_all(archive);
        {
            Network network(name);
            const auto feature = tokens1X(network);
            FullyConnected::Builder builder;
            builder.network(network).featureInput(feature)
                .numOutputFeatures(kOut1X).hasBias(true)
                .weightsDataType(DataType::FP32).computeDataType(DataType::FP32)
                .outputDataType(DataType::FP32).noActivation();
            auto projected = FullyConnected::epilogueInput(DataType::FP32, DataType::FP32);
            auto expression = projected * 2.0f;
            if (auxCount == 2u) {
                const auto first = auxiliary1YD(network, feature);
                const auto second = RaggedNetworkInput::Builder().network(network).name("other_aux")
                    .valuesDataType(DataType::FP32).trailingDimensions({kOut1X})
                    .partition(feature).build();
                auto firstExpr = FullyConnected::epilogueAuxInput("aux", DataType::FP32, DataType::FP32);
                auto secondExpr = FullyConnected::epilogueAuxInput("other", DataType::FP32, DataType::FP32);
                builder.epilogueInput("aux", first).epilogueInput("other", second);
                expression = expression + firstExpr + secondExpr;
            }
            const auto fc = builder.epilogue(expression).build();
            const json record = fc.architectureJson();
            EXPECT_EQ(record.at("version"), "1.7.0");
            EXPECT_EQ(record.at("epilogue_inputs").size(), auxCount);
            (void)RaggedNetworkOutput::Builder().network(network).name("projected")
                .inputTensor(fc.getRaggedFeatureOutput().value()).build();
            network.save(archive.string(), true);
        }
        for (uint32_t b : {2u, 7u}) {
            SCOPED_TRACE(::testing::Message() << "B=" << b);
            Network loaded(name);
            ASSERT_NO_THROW(loaded.load(archive.string()));
            const auto fc = find1X(loaded);
            ASSERT_NE(fc, nullptr);
            EXPECT_EQ(fc->architectureJson().at("version"), "1.7.0");
            EXPECT_EQ(fc->architectureJson().at("epilogue_inputs").size(), auxCount);
            vector<Event> events;
            auto placed = loaded.place(b, events, true);
            ASSERT_NE(placed, nullptr);
            for (auto& event : events) event.synchronize();
        }
        filesystem::remove_all(archive);
    }
}

// FC-2B composes the already-existing 1.7.0 epilogue grammar with its
// optional activation, dropout, and residual fields. Inference must execute
// affine -> activation -> epilogue -> (dropout bypassed) -> residual.
TEST(RaggedFullyConnectedNeutralFC2B, ComposedEpilogueInferenceOrderAndCanonicalSchemaAtB2B7) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        GTEST_SKIP() << "CUDA device unavailable";

    struct Configuration { bool activation; bool dropout; bool residual; };
    for (const Configuration configuration : {
             Configuration{true, false, false},
             Configuration{false, true, true},
             Configuration{true, true, true}}) {
        SCOPED_TRACE(::testing::Message() << "activation=" << configuration.activation
                     << ", dropout=" << configuration.dropout
                     << ", residual=" << configuration.residual);
        const string name = "fc2b_composed_" + to_string(configuration.activation) +
                            to_string(configuration.dropout) + to_string(configuration.residual);
        const auto archive = archive1X(name);
        filesystem::remove_all(archive);
        {
            Network network(name);
            const auto feature = tokens1X(network);
            const auto auxiliary = auxiliary1YD(network, feature);
            FullyConnected::Builder builder;
            builder.network(network).featureInput(feature)
                .numOutputFeatures(kOut1X).hasBias(true)
                .weightsDataType(DataType::FP32).computeDataType(DataType::FP32)
                .outputDataType(DataType::FP32)
                .outputDropoutProbability(configuration.dropout ? 0.25f : 0.0f)
                .outputDropoutSeed(2026);
            if (configuration.activation) builder.activation(Sigmoid::Builder().build());
            else builder.noActivation();
            if (configuration.residual) builder.residualInput(residual1YB(network, feature));
            auto projected = FullyConnected::epilogueInput(DataType::FP32, DataType::FP32);
            auto aux = FullyConnected::epilogueAuxInput("aux", DataType::FP32, DataType::FP32);
            const auto fc = builder.epilogueInput("aux", auxiliary).epilogue(projected + aux).build();
            const json record = fc.architectureJson();
            EXPECT_EQ(record.at("version"), "1.7.0");
            EXPECT_EQ(record.at("use_residual").get<bool>(), configuration.residual);
            EXPECT_EQ(record.contains("ragged_residual_input"), configuration.residual);
            EXPECT_EQ(record.at("activation").is_object(), configuration.activation);
            EXPECT_FLOAT_EQ(record.at("output_dropout_probability").get<float>(),
                            configuration.dropout ? 0.25f : 0.0f);
            EXPECT_FALSE(record.at("ragged_inputs").at(0).contains("batch_size"));
            // An extended 1.7.0 record still rejects malformed semantic
            // references and schema fields instead of silently dropping them.
            shared_ptr<thor_file::TarReader> reader;
            json bad = record;
            bad["output_dropout_probability"] = 1.0f;
            EXPECT_ANY_THROW(FullyConnected::deserialize(reader, bad, &network));
            bad = record;
            bad["activation"] = "invalid_activation_object";
            EXPECT_ANY_THROW(FullyConnected::deserialize(reader, bad, &network));
            if (configuration.residual) {
                bad = record;
                bad.erase("ragged_residual_input");
                EXPECT_ANY_THROW(FullyConnected::deserialize(reader, bad, &network));
                bad = record;
                bad["ragged_residual_input"]["offsets"]["id"] = feature.getValues().getOriginalId();
                EXPECT_ANY_THROW(FullyConnected::deserialize(reader, bad, &network));
            }
            (void)RaggedNetworkOutput::Builder().network(network).name("projected")
                .inputTensor(fc.getRaggedFeatureOutput().value()).build();
            network.save(archive.string(), true);
        }
        for (uint32_t b : {2u, 7u}) {
            SCOPED_TRACE(::testing::Message() << "B=" << b);
            Network network(name);
            ASSERT_NO_THROW(network.load(archive.string()));
            const auto fc = find1X(network);
            ASSERT_NE(fc, nullptr);
            const json before = network.architectureJson();
            const auto inputs = network.getExternalRaggedNetworkInputs();
            auto tokens = find_if(inputs.begin(), inputs.end(), [](const auto& x) { return x.name == "tokens"; });
            auto aux = find_if(inputs.begin(), inputs.end(), [](const auto& x) { return x.name == "aux"; });
            auto residual = find_if(inputs.begin(), inputs.end(), [](const auto& x) { return x.name == "residual"; });
            ASSERT_NE(tokens, inputs.end());
            ASSERT_NE(aux, inputs.end());
            if (configuration.residual) ASSERT_NE(residual, inputs.end());
            vector<Event> events;
            auto placed = network.place(b, events, true);
            ASSERT_NE(placed, nullptr);
            for (auto& event : events) event.synchronize();
            auto physical = dynamic_pointer_cast<Impl::RaggedFullyConnected>(
                placed->getStampedNetwork(0).getPhysicalLayerFromApiLayer(fc->getId()));
            ASSERT_NE(physical, nullptr);
            const auto weights = download1X(physical->getParameterStorage("weights"));
            const auto biases = download1X(physical->getParameterStorage("biases"));
            Batch batch = batch1YD(tokens->raggedTensor, aux->raggedTensor, b);
            if (configuration.residual) {
                Batch residualBatch = batch1YB(tokens->raggedTensor, residual->raggedTensor, b);
                batch.insert("residual", residualBatch.getRaggedTensor("residual"));
            }
            const auto output = placed->inferLogical(std::move(batch));
            ASSERT_TRUE(output.contains("projected"));
            ASSERT_TRUE(holds_alternative<Impl::RaggedTensor>(output.at("projected")));
            const auto result = get<Impl::RaggedTensor>(output.at("projected"));
            ASSERT_TRUE(result.getHostOffsetsIfAvailable().has_value());
            EXPECT_EQ(result.getHostOffsetsIfAvailable().value(), offsets1X(b));
            const float* actual = result.getValues().getMemPtr<float>();
            for (uint64_t t = 0; t < offsets1X(b).back(); ++t) {
                for (uint32_t o = 0; o < kOut1X; ++o) {
                    float projectedValue = biases[o];
                    for (uint32_t i = 0; i < kIn1X; ++i)
                        projectedValue += token1X(t, i) * weights[i*kOut1X+o];
                    if (configuration.activation) projectedValue = 1.f / (1.f + expf(-projectedValue));
                    // Evaluation skips dropout, but not the epilogue or residual.
                    const float expected = projectedValue + auxiliaryValue1YD(t, o) +
                        (configuration.residual ? residualValue1YB(t, o) : 0.f);
                    EXPECT_NEAR(actual[t*kOut1X+o], expected, 3e-3f);
                }
            }
            EXPECT_EQ(network.architectureJson(), before);
        }
        filesystem::remove_all(archive);
    }
}

TEST(RaggedFullyConnectedNeutralFC2B, EpilogueResidualDropoutTrainsAndRoutesBothGradientsB2B7) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        GTEST_SKIP() << "CUDA device unavailable";
    const string name = "fc2b_train_composition";
    const auto archive = archive1X(name);
    filesystem::remove_all(archive);
    {
        Network network(name);
        (void)Sgd::Builder().network(network).initialLearningRate(.001f).decay(0.f).momentum(0.f).build();
        const auto feature = tokens1X(network);
        const auto auxiliary = auxiliary1YD(network, feature);
        const auto residual = residual1YB(network, feature);
        const auto labels = RaggedNetworkInput::Builder().network(network).name("labels")
            .valuesDataType(DataType::FP32).trailingDimensions({kOut1X}).partition(feature).build();
        const auto auxRivet = GradientRivet::Builder().network(network).tensor(auxiliary.getValues()).build();
        const auto residualRivet = GradientRivet::Builder().network(network).tensor(residual.getValues()).build();
        auto projected = FullyConnected::epilogueInput(DataType::FP32, DataType::FP32);
        auto aux = FullyConnected::epilogueAuxInput("aux", DataType::FP32, DataType::FP32);
        const auto fc = FullyConnected::Builder().network(network).featureInput(feature)
            .numOutputFeatures(kOut1X).hasBias(true)
            .weightsDataType(DataType::FP32).computeDataType(DataType::FP32)
            .outputDataType(DataType::FP32).noActivation()
            .epilogueInput("aux", auxiliary.withValues(auxRivet.getFeatureOutput().value()))
            .epilogue(projected + aux)
            .outputDropoutProbability(0.25f).outputDropoutSeed(2026)
            .residualInput(residual.withValues(residualRivet.getFeatureOutput().value())).build();
        EXPECT_EQ(fc.architectureJson().at("version"), "1.7.0");
        auto predictions = Impl::Expression::input("predictions", DataType::FP32, DataType::FP32);
        auto target = Impl::Expression::input("labels", DataType::FP32, DataType::FP32);
        auto diff = predictions-target;
        auto loss = Impl::DynamicExpression::fromExpressionDefinition(
            Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({{
                "loss", (diff*diff).withOutputDType(DataType::FP32)}})));
        auto gradient = Impl::DynamicExpression::fromExpressionDefinition(
            Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({{
                "predictions_grad", (diff*Impl::Expression(2.0*Impl::Loss::getLossScalingFactor()))
                    .withOutputDType(DataType::FP32)}})));
        const auto raw = RaggedCustomLoss::Builder().network(network)
            .predictions(fc.getRaggedFeatureOutput().value()).labels(labels)
            .lossExpression(loss).gradientExpression(gradient).build();
        (void)NetworkOutput::Builder().network(network).name("raw_loss")
            .inputTensor(raw.getRaggedRawLoss().getValues()).dataType(DataType::FP32).build();
        network.save(archive.string(), true);
    }
    for (uint32_t b : {2u, 7u}) {
        SCOPED_TRACE(::testing::Message() << "B=" << b);
        Network network(name);
        ASSERT_NO_THROW(network.load(archive.string()));
        const auto inputs = network.getExternalRaggedNetworkInputs();
        auto tokens = find_if(inputs.begin(), inputs.end(), [](const auto& x) { return x.name == "tokens"; });
        auto aux = find_if(inputs.begin(), inputs.end(), [](const auto& x) { return x.name == "aux"; });
        auto residual = find_if(inputs.begin(), inputs.end(), [](const auto& x) { return x.name == "residual"; });
        ASSERT_NE(tokens, inputs.end());
        ASSERT_NE(aux, inputs.end());
        ASSERT_NE(residual, inputs.end());
        vector<Event> events;
        auto placed = network.place(b, events, false);
        ASSERT_NE(placed, nullptr);
        for (auto& event : events) event.synchronize();
        const auto fc = find1X(network);
        ASSERT_NE(fc, nullptr);
        auto physical = dynamic_pointer_cast<Impl::RaggedFullyConnected>(
            placed->getStampedNetwork(0).getPhysicalLayerFromApiLayer(fc->getId()));
        ASSERT_NE(physical, nullptr);
        const auto initialWeights = download1X(physical->getParameterStorage("weights"));
        Batch batch = batch1YD(tokens->raggedTensor, aux->raggedTensor, b, true);
        Batch residualBatch = batch1YB(tokens->raggedTensor, residual->raggedTensor, b);
        batch.insert("residual", residualBatch.getRaggedTensor("residual"));
        map<string, Impl::Tensor> outputs;
        map<string, Event> ready;
        auto finished = placed->submitBatch(0, std::move(batch), outputs, ready, false);
        finished.synchronize();
        placed->synchronize();
        EXPECT_TRUE(outputs.contains("raw_loss"));
        const auto errors = physical->getErrorOutputs();
        ASSERT_GE(errors.size(), 4u);
        ASSERT_TRUE(errors[2].has_value()); // residual gradient
        ASSERT_TRUE(errors[3].has_value()); // epilogue auxiliary gradient
        const auto dResidual = download1X(errors[2].value());
        const auto dAux = download1X(errors[3].value());
        for (uint64_t i = 0; i < offsets1X(b).back()*kOut1X; ++i) {
            EXPECT_TRUE(isfinite(dResidual.at(i)));
            EXPECT_TRUE(isfinite(dAux.at(i)));
        }
        // The auxiliary gradient is either masked out or scaled by 1/(1-p);
        // the skip/residual gradient bypasses dropout. Check both branches
        // without depending on a particular Philox mask realization.
        const float scale = Impl::Loss::getLossScalingFactor();
        for (uint64_t i = 0; i < offsets1X(b).back()*kOut1X; ++i) {
            const float dAuxUnscaled = dAux.at(i)/scale;
            const float dResidualUnscaled = dResidual.at(i)/scale;
            EXPECT_TRUE(fabs(dAuxUnscaled) < 1e-2f ||
                        fabs(dAuxUnscaled - dResidualUnscaled/0.75f) < 2e-2f);
        }
    }
    filesystem::remove_all(archive);
}

// FC-3A: per-application activation/dropout has no auxiliary binding schema.
// Reuse the canonical 1.6.0 grammar with independent M and row partitions;
// both the training kernel and dropout-free evaluation variant must stamp.
TEST(RaggedFullyConnectedNeutralFC3A, MultiApplicationActivationDropoutUsesExistingSchemaAtB2B7) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        GTEST_SKIP() << "CUDA device unavailable";

    constexpr uint64_t m0 = 4, m1 = 6, inFeatures = 3, outFeatures = 2;
    const string name = "fc3a_multi_activation_dropout";
    const auto archive = archive1X(name);
    filesystem::remove_all(archive);
    {
        Network network(name);
        (void)Sgd::Builder().network(network).initialLearningRate(.001f)
            .decay(0.f).momentum(0.f).build();
        auto input = [&](const string& sourceName, uint64_t m) {
            return RaggedNetworkInput::Builder().network(network).name(sourceName)
                .valuesDataType(DataType::FP32).offsetsDataType(DataType::UINT64)
                .trailingDimensions({inFeatures}).maxValuesPerRow(m).build();
        };
        const RaggedTensor first = input("first", m0);
        const RaggedTensor second = input("second", m1);
        ASSERT_FALSE(first.sharesPartitionWith(second));
        const auto fc = FullyConnected::Builder().network(network)
            .featureInput(first).featureInput(second).numOutputFeatures(outFeatures)
            .hasBias(true).weightsDataType(DataType::FP32)
            .computeDataType(DataType::FP32).outputDataType(DataType::FP32)
            .activation(Sigmoid::Builder().build())
            .outputDropoutProbability(0.25f).outputDropoutSeed(2026).build();
        EXPECT_EQ(fc.architectureJson().at("version"), "1.6.0");
        EXPECT_EQ(fc.architectureJson().at("ragged_inputs").size(), 2u);
        const auto predictions = Impl::Expression::input("predictions", DataType::FP32, DataType::FP32);
        const auto targets = Impl::Expression::input("labels", DataType::FP32, DataType::FP32);
        const auto difference = predictions - targets;
        const auto loss = Impl::DynamicExpression::fromExpressionDefinition(
            Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({{
                "loss", (difference * difference).withOutputDType(DataType::FP32)}})));
        const auto gradient = Impl::DynamicExpression::fromExpressionDefinition(
            Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({{
                "predictions_grad", (difference * Impl::Expression(2.0 * Impl::Loss::getLossScalingFactor()))
                    .withOutputDType(DataType::FP32)}})));
        for (uint32_t app = 0; app < 2; ++app) {
            const auto projected = fc.getRaggedFeatureOutput(app);
            ASSERT_TRUE(projected.has_value());
            (void)RaggedNetworkOutput::Builder().network(network)
                .name("projected" + to_string(app)).inputTensor(projected.value()).build();
            const RaggedTensor& feature = app == 0 ? first : second;
            const auto labels = RaggedNetworkInput::Builder().network(network)
                .name("target" + to_string(app)).valuesDataType(DataType::FP32)
                .trailingDimensions({outFeatures}).partition(feature).build();
            const auto raw = RaggedCustomLoss::Builder().network(network)
                .predictions(projected.value()).labels(labels)
                .lossExpression(loss).gradientExpression(gradient).build();
            (void)NetworkOutput::Builder().network(network).name("loss" + to_string(app))
                .inputTensor(raw.getRaggedRawLoss().getValues()).dataType(DataType::FP32).build();
        }
        network.save(archive.string(), true);
    }

    for (uint32_t b : {2u, 7u}) {
        for (bool inferenceOnly : {true, false}) {
            SCOPED_TRACE(::testing::Message() << "B=" << b << ", inferenceOnly=" << inferenceOnly);
            Network network(name);
            ASSERT_NO_THROW(network.load(archive.string()));
            const auto fc = find1X(network);
            ASSERT_NE(fc, nullptr);
            const json authored = fc->architectureJson();
            EXPECT_EQ(authored.at("version"), "1.6.0");
            ASSERT_EQ(fc->architectureJson().at("inputs").size(), 2u);
            const auto first = fc->getRaggedFeatureInput(0).value();
            const auto second = fc->getRaggedFeatureInput(1).value();
            EXPECT_EQ(first.getMaxValuesPerRow(), m0);
            EXPECT_EQ(second.getMaxValuesPerRow(), m1);
            EXPECT_FALSE(first.sharesPartitionWith(second));
            vector<Event> events;
            auto placed = network.place(b, events, inferenceOnly);
            ASSERT_NE(placed, nullptr);
            for (auto& event : events) event.synchronize();
            const auto physical = dynamic_pointer_cast<Impl::RaggedFullyConnected>(
                placed->getStampedNetwork(0).getPhysicalLayerFromApiLayer(fc->getId()));
            ASSERT_NE(physical, nullptr);
            EXPECT_EQ(physical->getFeatureOutputs().size(), 2u);
            EXPECT_EQ(fc->architectureJson().at("ragged_inputs"), authored.at("ragged_inputs"));
            EXPECT_EQ(fc->architectureJson().at("ragged_outputs"), authored.at("ragged_outputs"));
            EXPECT_EQ(fc->architectureJson().at("version"), authored.at("version"));

            const auto weights = download1X(physical->getParameterStorage("weights"));
            const auto biases = download1X(physical->getParameterStorage("biases"));
            Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
            Batch batch;
            vector<vector<uint64_t>> allOffsets;
            for (uint32_t app = 0; app < 2; ++app) {
                const RaggedTensor& logical = app == 0 ? first : second;
                const uint64_t m = app == 0 ? m0 : m1;
                auto descriptor = logical.getDescriptorForPlacement(b);
                Impl::Tensor values(cpu, descriptor.getValuesDescriptor());
                Impl::Tensor offsets(cpu, descriptor.getOffsetsDescriptor());
                vector<uint64_t> ends(b + 1, 0);
                for (uint32_t r = 0; r < b; ++r)
                    ends[r + 1] = ends[r] + (app == 0 ? r % 3 : (r + 1) % 5);
                allOffsets.push_back(ends);
                for (uint64_t t = 0; t < b * m; ++t)
                    for (uint32_t i = 0; i < inFeatures; ++i)
                        values.getMemPtr<float>()[t * inFeatures + i] =
                            t < ends.back() ? (.2f + .07f * t + .15f * i) :
                                              numeric_limits<float>::quiet_NaN();
                for (uint32_t r = 0; r <= b; ++r) offsets.getMemPtr<uint64_t>()[r] = ends[r];
                Impl::RowPartitionRuntime partition(offsets, descriptor.getRowPartition());
                partition.setHostOffsets(ends);
                batch.insert(app == 0 ? "first" : "second", Impl::RaggedTensor(values, partition));
                // Inference placement prunes the training-only loss and its labels.
                // Supply target inputs only when those loss branches are stamped.
                if (!inferenceOnly) {
                    Impl::Tensor labels(cpu, Impl::TensorDescriptor(DataType::FP32, {b * m, outFeatures}));
                    std::fill_n(labels.getMemPtr<float>(), b * m * outFeatures, 0.f);
                    batch.insert("target" + to_string(app), Impl::RaggedTensor(labels, partition));
                }
            }
            batch.setValidExampleCount(b);
            if (!inferenceOnly) {
                map<string, Impl::Tensor> outputs;
                map<string, Event> ready;
                auto finished = placed->submitBatch(0, batch, outputs, ready, false);
                finished.synchronize();
                placed->synchronize();
                ASSERT_TRUE(outputs.contains("loss0"));
                ASSERT_TRUE(outputs.contains("loss1"));
                vector<vector<bool>> masks;
                for (uint32_t app = 0; app < 2; ++app) {
                    const auto featureOutput = physical->getFeatureOutputs().at(app);
                    ASSERT_TRUE(featureOutput.has_value());
                    const auto observed = download1X(featureOutput.value());
                    vector<bool> mask;
                    for (uint64_t t = 0; t < allOffsets[app].back(); ++t) {
                        for (uint32_t o = 0; o < outFeatures; ++o) {
                            float affine = biases[o];
                            for (uint32_t i = 0; i < inFeatures; ++i)
                                affine += (.2f + .07f * t + .15f * i) * weights[i * outFeatures + o];
                            const float activated = 1.f / (1.f + std::exp(-affine));
                            const float value = observed.at(t * outFeatures + o);
                            const bool kept = std::abs(value) > 1e-6f;
                            mask.push_back(kept);
                            EXPECT_NEAR(value, kept ? activated / .75f : 0.f, 3e-3f);
                        }
                    }
                    masks.push_back(std::move(mask));
                }
                // The same input prefix and shared weights must not receive an
                // identical dropout stream just because applications share a layer.
                // Check the longer B=7 run to avoid a tiny stochastic sample.
                if (b == 7) {
                    const auto common = std::min(masks[0].size(), masks[1].size());
                    ASSERT_GT(common, 0u);
                    EXPECT_FALSE(std::equal(masks[0].begin(), masks[0].begin() + common,
                                            masks[1].begin()));
                }
                continue;
            }
            const auto results = placed->inferLogical(batch);
            for (uint32_t app = 0; app < 2; ++app) {
                const string outputName = "projected" + to_string(app);
                ASSERT_TRUE(results.contains(outputName));
                ASSERT_TRUE(holds_alternative<Impl::RaggedTensor>(results.at(outputName)));
                const auto& projected = get<Impl::RaggedTensor>(results.at(outputName));
                ASSERT_TRUE(projected.getHostOffsetsIfAvailable().has_value());
                EXPECT_EQ(projected.getHostOffsetsIfAvailable().value(), allOffsets[app]);
                const float* observed = projected.getValues().getMemPtr<float>();
                for (uint64_t t = 0; t < allOffsets[app].back(); ++t) {
                    for (uint32_t o = 0; o < outFeatures; ++o) {
                        float affine = biases[o];
                        for (uint32_t i = 0; i < inFeatures; ++i)
                            affine += (.2f + .07f * t + .15f * i) * weights[i * outFeatures + o];
                        const float expected = 1.f / (1.f + std::exp(-affine));
                        EXPECT_NEAR(observed[t * outFeatures + o], expected, 3e-3f);
                    }
                }
            }
        }
    }
    filesystem::remove_all(archive);
}

// FC-3B: each application owns an exact-partition residual, with independent
// M/offsets but one shared weight/bias parameter set. Existing 1.5.0 writes
// singleton residual references as objects and multi-app references as arrays.
TEST(RaggedFullyConnectedNeutralFC3B, IndependentResidualsTrainAndRestoreAtB2B7) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        GTEST_SKIP() << "CUDA device unavailable";

    constexpr uint64_t m0 = 4, m1 = 6, inFeatures = 3, outFeatures = 2;
    const string name = "fc3b_multi_residual";
    const auto archive = archive1X(name);
    filesystem::remove_all(archive);
    {
        Network network(name);
        (void)Sgd::Builder().network(network).initialLearningRate(.001f)
            .decay(0.f).momentum(0.f).build();
        auto featureInput = [&](const string& source, uint64_t m) {
            return RaggedNetworkInput::Builder().network(network).name(source)
                .valuesDataType(DataType::FP32).offsetsDataType(DataType::UINT64)
                .trailingDimensions({inFeatures}).maxValuesPerRow(m).build();
        };
        const RaggedTensor first = featureInput("first", m0);
        const RaggedTensor second = featureInput("second", m1);
        const RaggedTensor residual0 = RaggedNetworkInput::Builder().network(network)
            .name("residual0").valuesDataType(DataType::FP32)
            .trailingDimensions({outFeatures}).partition(first).build();
        const RaggedTensor residual1 = RaggedNetworkInput::Builder().network(network)
            .name("residual1").valuesDataType(DataType::FP32)
            .trailingDimensions({outFeatures}).partition(second).build();
        const auto fc = FullyConnected::Builder().network(network)
            .featureInput(first).featureInput(second).numOutputFeatures(outFeatures)
            .hasBias(true).weightsDataType(DataType::FP32)
            .computeDataType(DataType::FP32).outputDataType(DataType::FP32)
            .noActivation().residualInput(residual0).residualInput(residual1).build();
        const json authored = fc.architectureJson();
        EXPECT_EQ(authored.at("version"), "1.5.0");
        ASSERT_TRUE(authored.at("residual_input").is_array());
        ASSERT_EQ(authored.at("residual_input").size(), 2u);
        ASSERT_EQ(authored.at("ragged_residual_input").size(), 2u);
        EXPECT_NE(authored.at("ragged_residual_input").at(0).at("max_values_per_row"),
                  authored.at("ragged_residual_input").at(1).at("max_values_per_row"));
        shared_ptr<thor_file::TarReader> reader;
        auto reject = [&](json broken) {
            EXPECT_ANY_THROW(FullyConnected::deserialize(reader, broken, &network));
        };
        json broken = authored;
        broken["residual_input"] = authored.at("residual_input").at(0);
        reject(broken); // Multi-application 1.5.0 needs per-application references.
        broken = authored;
        broken["residual_input"][1] = broken["residual_input"][0];
        reject(broken); // No two applications may alias a residual values input.
        broken = authored;
        broken["ragged_residual_input"][1] = broken["ragged_residual_input"][0];
        reject(broken); // Exact row partition and M belong to each application.
        EXPECT_THROW((void)FullyConnected::Builder().network(network)
            .featureInput(first).featureInput(second).numOutputFeatures(outFeatures)
            .hasBias(true).noActivation().residualInput(residual0).build(),
            std::invalid_argument);

        auto predictions = Impl::Expression::input("predictions", DataType::FP32, DataType::FP32);
        auto targets = Impl::Expression::input("labels", DataType::FP32, DataType::FP32);
        auto difference = predictions - targets;
        const auto loss = Impl::DynamicExpression::fromExpressionDefinition(
            Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({{
                "loss", (difference * difference).withOutputDType(DataType::FP32)}})));
        const auto gradient = Impl::DynamicExpression::fromExpressionDefinition(
            Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({{
                "predictions_grad", (difference * Impl::Expression(2.0 * Impl::Loss::getLossScalingFactor()))
                    .withOutputDType(DataType::FP32)}})));
        for (uint32_t app = 0; app < 2; ++app) {
            const auto projected = fc.getRaggedFeatureOutput(app).value();
            (void)RaggedNetworkOutput::Builder().network(network)
                .name("projected" + to_string(app)).inputTensor(projected).build();
            const RaggedTensor feature = app == 0 ? first : second;
            const auto labels = RaggedNetworkInput::Builder().network(network)
                .name("target" + to_string(app)).valuesDataType(DataType::FP32)
                .trailingDimensions({outFeatures}).partition(feature).build();
            const auto raw = RaggedCustomLoss::Builder().network(network)
                .predictions(projected).labels(labels)
                .lossExpression(loss).gradientExpression(gradient).build();
            (void)NetworkOutput::Builder().network(network).name("loss" + to_string(app))
                .inputTensor(raw.getRaggedRawLoss().getValues()).dataType(DataType::FP32).build();
        }
        network.save(archive.string(), true);
    }

    for (uint32_t b : {2u, 7u}) {
        for (bool inferenceOnly : {true, false}) {
            SCOPED_TRACE(::testing::Message() << "B=" << b << ", inferenceOnly=" << inferenceOnly);
            Network network(name);
            ASSERT_NO_THROW(network.load(archive.string()));
            const auto fc = find1X(network);
            ASSERT_NE(fc, nullptr);
            const json authored = fc->architectureJson();
            EXPECT_EQ(authored.at("version"), "1.5.0");
            ASSERT_EQ(authored.at("ragged_inputs").size(), 2u);
            ASSERT_TRUE(fc->getRaggedResidualInput(0).has_value());
            ASSERT_TRUE(fc->getRaggedResidualInput(1).has_value());
            EXPECT_EQ(fc->getRaggedResidualInput(0)->getMaxValuesPerRow(), m0);
            EXPECT_EQ(fc->getRaggedResidualInput(1)->getMaxValuesPerRow(), m1);
            EXPECT_TRUE(fc->getRaggedResidualInput(0)->sharesPartitionWith(fc->getRaggedFeatureInput(0).value()));
            EXPECT_TRUE(fc->getRaggedResidualInput(1)->sharesPartitionWith(fc->getRaggedFeatureInput(1).value()));
            vector<Event> events;
            auto placed = network.place(b, events, inferenceOnly);
            ASSERT_NE(placed, nullptr);
            for (auto& event : events) event.synchronize();
            const auto physical = dynamic_pointer_cast<Impl::RaggedFullyConnected>(
                placed->getStampedNetwork(0).getPhysicalLayerFromApiLayer(fc->getId()));
            ASSERT_NE(physical, nullptr);
            EXPECT_EQ(physical->getFeatureOutputs().size(), 2u);
            const auto weights = download1X(physical->getParameterStorage("weights"));
            const auto biases = download1X(physical->getParameterStorage("biases"));
            Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
            Batch batch;
            vector<vector<uint64_t>> endsByApplication;
            for (uint32_t app = 0; app < 2; ++app) {
                const uint64_t m = app == 0 ? m0 : m1;
                const RaggedTensor feature = fc->getRaggedFeatureInput(app).value();
                const auto descriptor = feature.getDescriptorForPlacement(b);
                Impl::Tensor values(cpu, descriptor.getValuesDescriptor());
                Impl::Tensor offsets(cpu, descriptor.getOffsetsDescriptor());
                vector<uint64_t> ends(b + 1, 0);
                for (uint32_t row = 0; row < b; ++row)
                    ends[row + 1] = ends[row] + (app == 0 ? 1 + row % 3 : 1 + row % 5);
                endsByApplication.push_back(ends);
                for (uint64_t t = 0; t < b * m; ++t)
                    for (uint32_t i = 0; i < inFeatures; ++i)
                        values.getMemPtr<float>()[t * inFeatures + i] =
                            t < ends.back() ? (.1f + .03f * t + .07f * i + .1f * app) :
                                              numeric_limits<float>::quiet_NaN();
                for (uint32_t row = 0; row <= b; ++row) offsets.getMemPtr<uint64_t>()[row] = ends[row];
                Impl::RowPartitionRuntime partition(offsets, descriptor.getRowPartition());
                partition.setHostOffsets(ends);
                batch.insert(app == 0 ? "first" : "second", Impl::RaggedTensor(values, partition));
                Impl::Tensor residualValues(cpu, Impl::TensorDescriptor(DataType::FP32, {b * m, outFeatures}));
                for (uint64_t t = 0; t < b * m; ++t)
                    for (uint32_t i = 0; i < outFeatures; ++i)
                        residualValues.getMemPtr<float>()[t * outFeatures + i] =
                            t < ends.back() ? (.45f + .12f * app + .02f * t + .03f * i) :
                                              numeric_limits<float>::quiet_NaN();
                batch.insert("residual" + to_string(app), Impl::RaggedTensor(residualValues, partition));
                if (!inferenceOnly) {
                    Impl::Tensor labels(cpu, Impl::TensorDescriptor(DataType::FP32, {b * m, outFeatures}));
                    std::fill_n(labels.getMemPtr<float>(), b * m * outFeatures, 0.0f);
                    batch.insert("target" + to_string(app), Impl::RaggedTensor(labels, partition));
                }
            }
            batch.setValidExampleCount(b);
            map<string, Impl::Tensor> outputs;
            auto expectedOutput = [&](uint32_t app, uint64_t t, uint32_t o) {
                float expected = biases[o] + .45f + .12f * app + .02f * t + .03f * o;
                for (uint32_t i = 0; i < inFeatures; ++i)
                    expected += (.1f + .03f * t + .07f * i + .1f * app) * weights[i * outFeatures + o];
                return expected;
            };
            if (inferenceOnly) {
                const auto results = placed->inferLogical(batch);
                for (uint32_t app = 0; app < 2; ++app) {
                    const string outputName = "projected" + to_string(app);
                    ASSERT_TRUE(results.contains(outputName));
                    ASSERT_TRUE(holds_alternative<Impl::RaggedTensor>(results.at(outputName)));
                    const auto& result = get<Impl::RaggedTensor>(results.at(outputName));
                    ASSERT_TRUE(result.getHostOffsetsIfAvailable().has_value());
                    EXPECT_EQ(result.getHostOffsetsIfAvailable().value(), endsByApplication[app]);
                    const float* actual = result.getValues().getMemPtr<float>();
                    for (uint64_t t = 0; t < endsByApplication[app].back(); ++t)
                        for (uint32_t o = 0; o < outFeatures; ++o)
                            EXPECT_NEAR(actual[t * outFeatures + o], expectedOutput(app, t, o), 4e-3f);
                }
            } else {
                map<string, Event> ready;
                auto finished = placed->submitBatch(0, batch, outputs, ready, false);
                finished.synchronize();
                placed->synchronize();
                EXPECT_TRUE(outputs.contains("loss0"));
                EXPECT_TRUE(outputs.contains("loss1"));
                const auto errors = physical->getErrorOutputs();
                ASSERT_EQ(errors.size(), 6u);
                for (uint32_t app = 0; app < 2; ++app) {
                    ASSERT_TRUE(errors[app * 3 + 2].has_value());
                    ASSERT_TRUE(physical->getFeatureOutputs().at(app).has_value());
                    const auto residualGrad = download1X(errors[app * 3 + 2].value());
                    const auto forward = download1X(physical->getFeatureOutputs().at(app).value());
                    for (uint64_t t = 0; t < endsByApplication[app].back(); ++t)
                        for (uint32_t o = 0; o < outFeatures; ++o) {
                            const uint64_t k = t * outFeatures + o;
                            EXPECT_NEAR(forward.at(k), expectedOutput(app, t, o), 4e-3f);
                            EXPECT_NEAR(residualGrad.at(k) / Impl::Loss::getLossScalingFactor(),
                                        2.0f * expectedOutput(app, t, o), 4e-2f);
                        }
                }
            }
            EXPECT_EQ(fc->architectureJson().at("version"), authored.at("version"));
            EXPECT_EQ(fc->architectureJson().at("ragged_inputs"), authored.at("ragged_inputs"));
            EXPECT_EQ(fc->architectureJson().at("ragged_outputs"), authored.at("ragged_outputs"));
            EXPECT_EQ(fc->architectureJson().at("residual_input"), authored.at("residual_input"));
            EXPECT_EQ(fc->architectureJson().at("ragged_residual_input"), authored.at("ragged_residual_input"));
        }
    }
    filesystem::remove_all(archive);
}
