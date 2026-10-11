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
constexpr uint64_t M1Z=4, H1Z=3;
constexpr float EPS1Z=1e-5f;
RaggedTensor input1Z(Network& network) {
    return RaggedNetworkInput::Builder().network(network).name("tokens")
        .valuesDataType(DataType::FP32).offsetsDataType(DataType::UINT64)
        .trailingDimensions({H1Z}).maxValuesPerRow(M1Z).build();
}
RMSNorm layer1Z(Network& network, const RaggedTensor& input) {
    return RMSNorm::Builder().network(network).featureInput(input)
        .normalizedShape({H1Z}).epsilon(EPS1Z).parameterDataType(DataType::FP32).build();
}
shared_ptr<RMSNorm> find1Z(Network& network) {
    for (uint32_t i=0; i<network.getNumLayers(); ++i) {
        auto rms=dynamic_pointer_cast<RMSNorm>(network.getLayer(i));
        if (rms) return rms;
    }
    return {};
}
filesystem::path archive1Z(const string& prefix) {
    return filesystem::temp_directory_path() /
        (prefix+"_"+to_string(chrono::steady_clock::now().time_since_epoch().count()));
}
vector<uint64_t> offsets1Z(uint32_t b) {
    vector<uint64_t> result(b+1,0);
    for (uint32_t r=0; r<b; ++r) result[r+1]=result[r]+(r%3==0?0u:r%3==1?2u:1u);
    return result;
}
float x1Z(uint64_t t, uint32_t c) {
    return .25f+.13f*static_cast<float>(t)+.15f*static_cast<float>(c);
}
Batch batch1Z(const RaggedTensor& input, uint32_t b, bool labels=false) {
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    const auto descriptor=input.getDescriptorForPlacement(b);
    const auto ends=offsets1Z(b);
    Impl::Tensor values(cpu,descriptor.getValuesDescriptor());
    Impl::Tensor boundaries(cpu,descriptor.getOffsetsDescriptor());
    for (uint64_t t=0;t<b*M1Z;++t)
        for (uint32_t c=0;c<H1Z;++c)
            values.getMemPtr<float>()[t*H1Z+c] =
                t<ends.back()?x1Z(t,c):numeric_limits<float>::quiet_NaN();
    for (uint32_t r=0;r<=b;++r) boundaries.getMemPtr<uint64_t>()[r]=ends[r];
    Impl::RowPartitionRuntime partition(boundaries,descriptor.getRowPartition());
    partition.setHostOffsets(ends);
    Batch result;
    result.insert("tokens",Impl::RaggedTensor(values,partition));
    if (labels) {
        Impl::Tensor targets(cpu,Impl::TensorDescriptor(DataType::FP32,{b*M1Z,H1Z}));
        for (uint64_t i=0;i<b*M1Z*H1Z;++i) targets.getMemPtr<float>()[i]=0.f;
        result.insert("labels",Impl::RaggedTensor(targets,partition));
    }
    result.setValidExampleCount(b);
    return result;
}
vector<float> download1Z(Impl::Tensor value) {
    const auto bytes=value.getArraySizeInBytes();
    EXPECT_EQ(bytes%sizeof(float),0u);
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor host=value.clone(cpu);
    Stream stream=Stream::getNextDownloadStream(value.getPlacement().getDeviceNum());
    host.downloadSection(value,stream,0,0,bytes);
    stream.synchronize();
    return vector<float>(host.getMemPtr<float>(),host.getMemPtr<float>()+bytes/sizeof(float));
}
float norm1Z(const vector<float>& x) {
    float sum=0;
    for (float v:x) sum+=v*v;
    return sqrtf(sum/static_cast<float>(H1Z)+EPS1Z);
}
float tokenLoss1Z(const vector<float>& x,const vector<float>& weights) {
    float result=0;
    const float rms=norm1Z(x);
    for (uint32_t i=0;i<H1Z;++i) {
        const float y=x[i]*weights[i]/rms;
        result+=y*y;
    }
    return result;
}
} // namespace

TEST(RaggedRMSNormNeutral1Z, SemanticSchemaMemoryAndFixedCompatibility) {
    Network network("rms_1z_semantics");
    const auto input=input1Z(network);
    const auto rms=layer1Z(network,input);
    ASSERT_TRUE(rms.getRaggedFeatureOutput().has_value());
    const auto output=rms.getRaggedFeatureOutput().value();
    EXPECT_TRUE(output.sharesPartitionWith(input));
    EXPECT_NE(output.getValues().getOriginalId(),input.getValues().getOriginalId());
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    for (uint32_t b : {2u,7u}) {
        EXPECT_EQ(rms.getOutputTensorBytes(b),b*M1Z*H1Z*sizeof(float));
        EXPECT_EQ(rms.getNonFirstInstanceMemRequirementInBytes(b,cpu),rms.getOutputTensorBytes(b));
        EXPECT_EQ(rms.getFirstInstanceMemRequirementInBytes(b,cpu),
                  rms.getOutputTensorBytes(b)+H1Z*sizeof(float));
    }
    EXPECT_ANY_THROW((void)rms.getOutputTensorBytes(0));
    const json record=rms.architectureJson();
    EXPECT_EQ(record.at("version"),"1.0.0");
    for (const char* name : {"ragged_inputs","ragged_outputs"}) {
        const json& r=record.at(name).at(0);
        EXPECT_FALSE(r.contains("geometry_mode"));
        EXPECT_FALSE(r.contains("batch_size"));
        EXPECT_FALSE(r.contains("max_total_values"));
        EXPECT_EQ(r.at("offsets").at("dimensions"),(vector<uint64_t>{1}));
    }
    shared_ptr<thor_file::TarReader> reader;
    auto reject=[&](json broken) { EXPECT_ANY_THROW(RMSNorm::deserialize(reader,broken,&network)); };
    json bad=record; bad["version"]="1.2.0";
    EXPECT_NO_THROW(RMSNorm::deserialize(reader,bad,&network)); // Archived canonical grammar remains readable.
    for (const char* key : {"ragged_inputs", "ragged_outputs"}) {
        for (auto& reference : bad[key]) {
            reference["geometry_mode"] = "placement_neutral";
        }
    }
    EXPECT_NO_THROW(RMSNorm::deserialize(reader,bad,&network)); // Prior marker does not choose model geometry.
    bad=record; bad["batch_size"]=7; reject(bad);
    bad=record; bad["ragged_inputs"][0]["batch_size"]=7; reject(bad);
    bad=record; bad["ragged_outputs"][0]["max_total_values"]=28; reject(bad);
    bad=record; bad["ragged_inputs"][0]["max_values_per_row"]=5; reject(bad);
    bad=record; bad["ragged_inputs"][0]["values"]["dimensions"]=vector<uint64_t>{7,H1Z}; reject(bad);
    bad=record; bad["ragged_outputs"][0]["offsets"]["id"]=input.getValues().getOriginalId(); reject(bad);
    bad=record; bad["inputs"][0]["data_type"]=DataType::BF16; reject(bad);
    bad=record; bad["outputs"][0]["dimensions"]=vector<uint64_t>{M1Z,4}; reject(bad);
    bad=record; bad["parameters"]["weights"]["shape"]=vector<uint64_t>{4}; reject(bad);
    bad=record; bad["epilogue_inputs"].push_back({{"name","aux"}}); reject(bad);
    Network fixedNetwork("rms_1z_explicit_batch_hints");
    auto fixed=RaggedNetworkInput::Builder().network(fixedNetwork).name("fixed")
        .valuesDataType(DataType::FP32).trailingDimensions({H1Z})
        .maxValuesPerRow(M1Z).build();
    const auto legacy=layer1Z(fixedNetwork,fixed);
    EXPECT_EQ(legacy.architectureJson().at("version"),"1.0.0");
    EXPECT_EQ(legacy.getOutputTensorBytes(7),7u*M1Z*H1Z*sizeof(float));
}

TEST(RaggedRMSNormNeutral1Z, SavedInferenceB2AndB7ActiveOnly) {
    int devices=0;
    if (cudaGetDeviceCount(&devices)!=cudaSuccess || devices==0) GTEST_SKIP()<<"CUDA device unavailable";
    const auto archive=archive1Z("rms_1z_inference");
    {
        Network network("rms_1z_inference");
        const auto input=input1Z(network);
        const auto rms=layer1Z(network,input);
        (void)RaggedNetworkOutput::Builder().network(network).name("normalized")
            .inputTensor(rms.getRaggedFeatureOutput().value()).build();
        network.save(archive.string(),true);
    }
    for (uint32_t b : {2u,7u}) {
        SCOPED_TRACE(::testing::Message()<<"B="<<b);
        Network network("rms_1z_inference");
        ASSERT_NO_THROW(network.load(archive.string()));
        auto rms=find1Z(network);
        ASSERT_NE(rms,nullptr);
        const auto before=network.architectureJson();
        const auto input=network.getExternalRaggedNetworkInputs().front().raggedTensor;
        vector<Event> events;
        auto placed=network.place(b,events,true);
        ASSERT_NE(placed,nullptr);
        for (auto& e:events) e.synchronize();
        auto& stamped=placed->getStampedNetwork(0);
        auto physical=dynamic_pointer_cast<Impl::RaggedRMSNorm>(
            stamped.getPhysicalLayerFromApiLayer(rms->getId()));
        ASSERT_NE(physical,nullptr);
        ASSERT_TRUE(physical->getFeatureOutput().has_value());
        EXPECT_EQ(physical->getFeatureOutput()->getDimensions(),(vector<uint64_t>{b*M1Z,H1Z}));
        const auto weights=download1Z(physical->getParameterStorage("weights"));
        auto outputs=placed->inferLogical(batch1Z(input,b));
        ASSERT_TRUE(outputs.contains("normalized"));
        ASSERT_TRUE(holds_alternative<Impl::RaggedTensor>(outputs.at("normalized")));
        const auto result=get<Impl::RaggedTensor>(outputs.at("normalized"));
        const auto ends=result.getHostOffsetsIfAvailable();
        ASSERT_TRUE(ends.has_value());
        EXPECT_EQ(ends.value(),offsets1Z(b));
        const float* actual=result.getValues().getMemPtr<float>();
        for (uint64_t t=0;t<ends->back();++t) {
            vector<float> x(H1Z);
            for (uint32_t c=0;c<H1Z;++c) x[c]=x1Z(t,c);
            const float denom=norm1Z(x);
            for (uint32_t c=0;c<H1Z;++c)
                EXPECT_NEAR(actual[t*H1Z+c],x[c]*weights[c]/denom,2e-3f);
        }
        EXPECT_EQ(network.architectureJson(),before);
    }
    filesystem::remove_all(archive);
}

TEST(RaggedRMSNormNeutral1Z, SavedTrainingB2AndB7FiniteDifferenceAndWeightUpdate) {
    int devices=0;
    if (cudaGetDeviceCount(&devices)!=cudaSuccess || devices==0) GTEST_SKIP()<<"CUDA device unavailable";
    const auto archive=archive1Z("rms_1z_training");
    {
        Network network("rms_1z_training");
        (void)Sgd::Builder().network(network).initialLearningRate(.001f).decay(0.f).momentum(0.f).build();
        const auto input=input1Z(network);
        const auto labels=RaggedNetworkInput::Builder().network(network).name("labels")
            .valuesDataType(DataType::FP32).trailingDimensions({H1Z}).partition(input).build();
        const auto rivet=GradientRivet::Builder().network(network).tensor(input.getValues()).build();
        const auto rms=layer1Z(network,input.withValues(rivet.getFeatureOutput().value()));
        const auto prediction=Impl::Expression::input("predictions",DataType::FP32,DataType::FP32);
        const auto truth=Impl::Expression::input("labels",DataType::FP32,DataType::FP32);
        const auto diff=prediction-truth;
        auto loss=Impl::DynamicExpression::fromExpressionDefinition(
            Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({{
                "loss",(diff*diff).withOutputDType(DataType::FP32)}})));
        auto grad=Impl::DynamicExpression::fromExpressionDefinition(
            Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({{
                "predictions_grad",(diff*Impl::Expression(2.0*Impl::Loss::getLossScalingFactor()))
                    .withOutputDType(DataType::FP32)}})));
        auto raw=RaggedCustomLoss::Builder().network(network)
            .predictions(rms.getRaggedFeatureOutput().value()).labels(labels)
            .lossExpression(loss).gradientExpression(grad).build();
        (void)NetworkOutput::Builder().network(network).name("raw_loss")
            .inputTensor(raw.getRaggedRawLoss().getValues()).dataType(DataType::FP32).build();
        network.save(archive.string(),true);
    }
    for (uint32_t b : {2u,7u}) {
        SCOPED_TRACE(::testing::Message()<<"B="<<b);
        Network network("rms_1z_training");
        ASSERT_NO_THROW(network.load(archive.string()));
        auto rms=find1Z(network);
        ASSERT_NE(rms,nullptr);
        const auto before=network.architectureJson();
        const auto inputs=network.getExternalRaggedNetworkInputs();
        auto it=find_if(inputs.begin(),inputs.end(),[](const auto& x){return x.name=="tokens";});
        ASSERT_NE(it,inputs.end());
        vector<Event> events;
        auto placed=network.place(b,events,false);
        ASSERT_NE(placed,nullptr);
        for (auto& e:events) e.synchronize();
        auto physical=dynamic_pointer_cast<Impl::RaggedRMSNorm>(
            placed->getStampedNetwork(0).getPhysicalLayerFromApiLayer(rms->getId()));
        ASSERT_NE(physical,nullptr);
        const auto oldWeights=download1Z(physical->getParameterStorage("weights"));
        map<string,Impl::Tensor> outputs;
        map<string,Event> ready;
        auto finished=placed->submitBatch(0,batch1Z(it->raggedTensor,b,true),outputs,ready,false);
        finished.synchronize();
        placed->synchronize();
        ASSERT_TRUE(outputs.contains("raw_loss"));
        ASSERT_FALSE(physical->getErrorOutputs().empty());
        ASSERT_TRUE(physical->getErrorOutputs()[0].has_value());
        const auto dx=download1Z(physical->getErrorOutputs()[0].value());
        const auto newWeights=download1Z(physical->getParameterStorage("weights"));
        const float lossScale=Impl::Loss::getLossScalingFactor();
        vector<float> dw(H1Z,0.f);
        for (uint64_t t=0;t<offsets1Z(b).back();++t) {
            vector<float> x(H1Z);
            for (uint32_t c=0;c<H1Z;++c) x[c]=x1Z(t,c);
            const float denom=norm1Z(x);
            for (uint32_t c=0;c<H1Z;++c) {
                dw[c]+=2.f*x[c]*x[c]*oldWeights[c]/(denom*denom);
                const float step=1e-3f;
                auto plus=x,minus=x;
                plus[c]+=step; minus[c]-=step;
                const float expected=(tokenLoss1Z(plus,oldWeights)-tokenLoss1Z(minus,oldWeights))/(2.f*step);
                ASSERT_GT(dx.size(),t*H1Z+c);
                EXPECT_NEAR(dx[t*H1Z+c]/lossScale,expected,3e-2f);
            }
        }
        bool updated=false;
        for (uint32_t c=0;c<H1Z;++c) {
            const float delta=newWeights[c]-oldWeights[c];
            updated|=fabs(delta)>1e-8f;
            if (fabs(dw[c])>1e-4f) EXPECT_LE(delta*dw[c],1e-7f);
        }
        EXPECT_TRUE(updated);
        EXPECT_EQ(network.architectureJson(),before);
    }
    filesystem::remove_all(archive);
}
