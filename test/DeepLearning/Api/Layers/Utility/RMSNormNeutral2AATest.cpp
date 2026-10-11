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
constexpr uint64_t M2AA=4, H2AA=3;
constexpr float EPS2AA=1e-5f;
RaggedTensor input2AA(Network& network) {
    return RaggedNetworkInput::Builder().network(network).name("tokens")
        .valuesDataType(DataType::FP32).offsetsDataType(DataType::UINT64)
        .trailingDimensions({H2AA}).maxValuesPerRow(M2AA).build();
}
RMSNorm layer2AA(Network& network, const RaggedTensor& input) {
    return RMSNorm::Builder().network(network).featureInput(input)
        .normalizedShape({H2AA}).epsilon(EPS2AA).parameterDataType(DataType::FP32)
        .epilogue(RMSNorm::epilogueInput() + Impl::Expression(0.25)).build();
}
shared_ptr<RMSNorm> find2AA(Network& network) {
    for (uint32_t i=0; i<network.getNumLayers(); ++i) {
        auto rms=dynamic_pointer_cast<RMSNorm>(network.getLayer(i));
        if (rms) return rms;
    }
    return {};
}
filesystem::path archive2AA(const string& prefix) {
    return filesystem::temp_directory_path() /
        (prefix+"_"+to_string(chrono::steady_clock::now().time_since_epoch().count()));
}
vector<uint64_t> offsets2AA(uint32_t b) {
    vector<uint64_t> result(b+1,0);
    for (uint32_t r=0; r<b; ++r) result[r+1]=result[r]+(r%3==0?0u:r%3==1?2u:1u);
    return result;
}
float x2AA(uint64_t t, uint32_t c) {
    return .25f+.13f*static_cast<float>(t)+.15f*static_cast<float>(c);
}
Batch batch2AA(const RaggedTensor& input, uint32_t b, bool labels=false) {
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    const auto descriptor=input.getDescriptorForPlacement(b);
    const auto ends=offsets2AA(b);
    Impl::Tensor values(cpu,descriptor.getValuesDescriptor());
    Impl::Tensor boundaries(cpu,descriptor.getOffsetsDescriptor());
    for (uint64_t t=0;t<b*M2AA;++t)
        for (uint32_t c=0;c<H2AA;++c)
            values.getMemPtr<float>()[t*H2AA+c] =
                t<ends.back()?x2AA(t,c):numeric_limits<float>::quiet_NaN();
    for (uint32_t r=0;r<=b;++r) boundaries.getMemPtr<uint64_t>()[r]=ends[r];
    Impl::RowPartitionRuntime partition(boundaries,descriptor.getRowPartition());
    partition.setHostOffsets(ends);
    Batch result;
    result.insert("tokens",Impl::RaggedTensor(values,partition));
    if (labels) {
        Impl::Tensor targets(cpu,Impl::TensorDescriptor(DataType::FP32,{b*M2AA,H2AA}));
        for (uint64_t i=0;i<b*M2AA*H2AA;++i) targets.getMemPtr<float>()[i]=0.f;
        result.insert("labels",Impl::RaggedTensor(targets,partition));
    }
    result.setValidExampleCount(b);
    return result;
}
vector<float> download2AA(Impl::Tensor value) {
    const auto bytes=value.getArraySizeInBytes();
    EXPECT_EQ(bytes%sizeof(float),0u);
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor host=value.clone(cpu);
    Stream stream=Stream::getNextDownloadStream(value.getPlacement().getDeviceNum());
    host.downloadSection(value,stream,0,0,bytes);
    stream.synchronize();
    return vector<float>(host.getMemPtr<float>(),host.getMemPtr<float>()+bytes/sizeof(float));
}
float norm2AA(const vector<float>& x) {
    float sum=0;
    for (float v:x) sum+=v*v;
    return sqrtf(sum/static_cast<float>(H2AA)+EPS2AA);
}
float tokenLoss2AA(const vector<float>& x,const vector<float>& weights) {
    float result=0;
    const float rms=norm2AA(x);
    for (uint32_t i=0;i<H2AA;++i) {
        const float y=x[i]*weights[i]/rms+0.25f;
        result+=y*y;
    }
    return result;
}
} // namespace

TEST(RaggedRMSNormNeutral2AA, SemanticSchemaMemoryAndFixedCompatibility) {
    Network network("rms_2aa_semantics");
    const auto input=input2AA(network);
    const auto rms=layer2AA(network,input);
    ASSERT_TRUE(rms.getRaggedFeatureOutput().has_value());
    const auto output=rms.getRaggedFeatureOutput().value();
    EXPECT_TRUE(output.sharesPartitionWith(input));
    EXPECT_NE(output.getValues().getOriginalId(),input.getValues().getOriginalId());
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    for (uint32_t b : {2u,7u}) {
        EXPECT_EQ(rms.getOutputTensorBytes(b),b*M2AA*H2AA*sizeof(float));
        EXPECT_EQ(rms.getNonFirstInstanceMemRequirementInBytes(b,cpu),rms.getOutputTensorBytes(b));
        EXPECT_EQ(rms.getFirstInstanceMemRequirementInBytes(b,cpu),
                  rms.getOutputTensorBytes(b)+H2AA*sizeof(float));
    }
    EXPECT_ANY_THROW((void)rms.getOutputTensorBytes(0));
    const json record=rms.architectureJson();
    EXPECT_EQ(record.at("version"),"1.0.0");
    EXPECT_TRUE(record.at("epilogue").is_object());
    EXPECT_TRUE(record.at("epilogue_inputs").empty());
    for (const char* name : {"ragged_inputs","ragged_outputs"}) {
        const json& r=record.at(name).at(0);
        EXPECT_FALSE(r.contains("geometry_mode"));
        EXPECT_FALSE(r.contains("batch_size"));
        EXPECT_FALSE(r.contains("max_total_values"));
        EXPECT_EQ(r.at("offsets").at("dimensions"),(vector<uint64_t>{1}));
    }
    shared_ptr<thor_file::TarReader> reader;
    auto reject=[&](json broken) { EXPECT_ANY_THROW(RMSNorm::deserialize(reader,broken,&network)); };
    json bad=record; bad["version"]="1.3.0";
    EXPECT_NO_THROW(RMSNorm::deserialize(reader,bad,&network)); // Superseded canonical writer grammar.
    bad=record; bad["version"]="1.2.0"; reject(bad);
    bad=record; bad["epilogue"]={{"bogus",true}}; reject(bad);
    bad=record; bad["batch_size"]=7; reject(bad);
    bad=record; bad["ragged_inputs"][0]["batch_size"]=7; reject(bad);
    bad=record; bad["ragged_outputs"][0]["max_total_values"]=28; reject(bad);
    bad=record; bad["ragged_inputs"][0]["max_values_per_row"]=5; reject(bad);
    bad=record; bad["ragged_inputs"][0]["values"]["dimensions"]=vector<uint64_t>{7,H2AA}; reject(bad);
    bad=record; bad["ragged_outputs"][0]["offsets"]["id"]=input.getValues().getOriginalId(); reject(bad);
    bad=record; bad["inputs"][0]["data_type"]=DataType::BF16; reject(bad);
    bad=record; bad["outputs"][0]["dimensions"]=vector<uint64_t>{M2AA,4}; reject(bad);
    bad=record; bad["parameters"]["weights"]["shape"]=vector<uint64_t>{4}; reject(bad);
    bad=record; bad["epilogue_inputs"].push_back({{"name","aux"}}); reject(bad);
    bad=record; bad["ragged_inputs"][0]["geometry_mode"]="fixed"; reject(bad);
    Network fixedNetwork("rms_2aa_fixed");
    auto fixed=RaggedNetworkInput::Builder().network(fixedNetwork).name("fixed")
        .valuesDataType(DataType::FP32).trailingDimensions({H2AA})
        .maxValuesPerRow(M2AA).build();
    const auto legacy=layer2AA(fixedNetwork,fixed);
    EXPECT_EQ(legacy.architectureJson().at("version"),"1.0.0");
    EXPECT_EQ(legacy.getOutputTensorBytes(7),7u*M2AA*H2AA*sizeof(float));
    // Ragged auxiliary inputs use the established 1.1.0 feature schema,
    // not a new version for canonical logical geometry.
    Network auxiliaryNetwork("rms_2aa_aux_version_boundary");
    const auto primary=input2AA(auxiliaryNetwork);
    const auto residual=RaggedNetworkInput::Builder().network(auxiliaryNetwork).name("residual")
        .valuesDataType(DataType::FP32).trailingDimensions({H2AA}).partition(primary).build();
    const auto withAux=RMSNorm::Builder().network(auxiliaryNetwork).featureInput(primary)
        .normalizedShape({H2AA}).epilogueInput("residual",residual)
        .epilogue(RMSNorm::epilogueInput()+RMSNorm::epilogueAuxInput("residual")).build();
    EXPECT_EQ(withAux.architectureJson().at("version"), "1.1.0");
    ASSERT_TRUE(withAux.getRaggedFeatureOutput().has_value());
    EXPECT_TRUE(withAux.getRaggedFeatureOutput()->sharesPartitionWith(primary));

    // An auxiliary on an independent canonical partition remains invalid.
    const auto wrongPartition=RaggedNetworkInput::Builder().network(auxiliaryNetwork)
        .name("wrong_partition").valuesDataType(DataType::FP32)
        .offsetsDataType(DataType::UINT64).trailingDimensions({H2AA})
        .maxValuesPerRow(M2AA).build();
    EXPECT_ANY_THROW((void)RMSNorm::Builder().network(auxiliaryNetwork).featureInput(primary)
        .normalizedShape({H2AA}).epilogueInput("wrong_partition",wrongPartition)
        .epilogue(RMSNorm::epilogueInput()+RMSNorm::epilogueAuxInput("wrong_partition")).build());
}

TEST(RaggedRMSNormNeutral2AA, SavedInferenceB2AndB7ActiveOnly) {
    int devices=0;
    if (cudaGetDeviceCount(&devices)!=cudaSuccess || devices==0) GTEST_SKIP()<<"CUDA device unavailable";
    const auto archive=archive2AA("rms_2aa_inference");
    {
        Network network("rms_2aa_inference");
        const auto input=input2AA(network);
        const auto rms=layer2AA(network,input);
        (void)RaggedNetworkOutput::Builder().network(network).name("normalized")
            .inputTensor(rms.getRaggedFeatureOutput().value()).build();
        network.save(archive.string(),true);
    }
    for (uint32_t b : {2u,7u}) {
        SCOPED_TRACE(::testing::Message()<<"B="<<b);
        Network network("rms_2aa_inference");
        ASSERT_NO_THROW(network.load(archive.string()));
        auto rms=find2AA(network);
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
        EXPECT_EQ(physical->getFeatureOutput()->getDimensions(),(vector<uint64_t>{b*M2AA,H2AA}));
        const auto weights=download2AA(physical->getParameterStorage("weights"));
        auto outputs=placed->inferLogical(batch2AA(input,b));
        ASSERT_TRUE(outputs.contains("normalized"));
        ASSERT_TRUE(holds_alternative<Impl::RaggedTensor>(outputs.at("normalized")));
        const auto result=get<Impl::RaggedTensor>(outputs.at("normalized"));
        const auto ends=result.getHostOffsetsIfAvailable();
        ASSERT_TRUE(ends.has_value());
        EXPECT_EQ(ends.value(),offsets2AA(b));
        const float* actual=result.getValues().getMemPtr<float>();
        for (uint64_t t=0;t<ends->back();++t) {
            vector<float> x(H2AA);
            for (uint32_t c=0;c<H2AA;++c) x[c]=x2AA(t,c);
            const float denom=norm2AA(x);
            for (uint32_t c=0;c<H2AA;++c)
                EXPECT_NEAR(actual[t*H2AA+c],x[c]*weights[c]/denom+0.25f,2e-3f);
        }
        EXPECT_EQ(network.architectureJson(),before);
    }
    filesystem::remove_all(archive);
}

TEST(RaggedRMSNormNeutral2AA, SavedTrainingB2AndB7FiniteDifferenceAndWeightUpdate) {
    int devices=0;
    if (cudaGetDeviceCount(&devices)!=cudaSuccess || devices==0) GTEST_SKIP()<<"CUDA device unavailable";
    const auto archive=archive2AA("rms_2aa_training");
    {
        Network network("rms_2aa_training");
        (void)Sgd::Builder().network(network).initialLearningRate(.001f).decay(0.f).momentum(0.f).build();
        const auto input=input2AA(network);
        const auto labels=RaggedNetworkInput::Builder().network(network).name("labels")
            .valuesDataType(DataType::FP32).trailingDimensions({H2AA}).partition(input).build();
        const auto rivet=GradientRivet::Builder().network(network).tensor(input.getValues()).build();
        const auto rms=layer2AA(network,input.withValues(rivet.getFeatureOutput().value()));
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
        Network network("rms_2aa_training");
        ASSERT_NO_THROW(network.load(archive.string()));
        auto rms=find2AA(network);
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
        const auto oldWeights=download2AA(physical->getParameterStorage("weights"));
        map<string,Impl::Tensor> outputs;
        map<string,Event> ready;
        auto finished=placed->submitBatch(0,batch2AA(it->raggedTensor,b,true),outputs,ready,false);
        finished.synchronize();
        placed->synchronize();
        ASSERT_TRUE(outputs.contains("raw_loss"));
        ASSERT_FALSE(physical->getErrorOutputs().empty());
        ASSERT_TRUE(physical->getErrorOutputs()[0].has_value());
        const auto dx=download2AA(physical->getErrorOutputs()[0].value());
        const auto newWeights=download2AA(physical->getParameterStorage("weights"));
        const float lossScale=Impl::Loss::getLossScalingFactor();
        vector<float> dw(H2AA,0.f);
        for (uint64_t t=0;t<offsets2AA(b).back();++t) {
            vector<float> x(H2AA);
            for (uint32_t c=0;c<H2AA;++c) x[c]=x2AA(t,c);
            const float denom=norm2AA(x);
            for (uint32_t c=0;c<H2AA;++c) {
                dw[c]+=2.f*(x[c]*oldWeights[c]/denom+0.25f)*x[c]/denom;
                const float step=1e-3f;
                auto plus=x,minus=x;
                plus[c]+=step; minus[c]-=step;
                const float expected=(tokenLoss2AA(plus,oldWeights)-tokenLoss2AA(minus,oldWeights))/(2.f*step);
                ASSERT_GT(dx.size(),t*H2AA+c);
                EXPECT_NEAR(dx[t*H2AA+c]/lossScale,expected,3e-2f);
            }
        }
        bool updated=false;
        for (uint32_t c=0;c<H2AA;++c) {
            const float delta=newWeights[c]-oldWeights[c];
            updated|=fabs(delta)>1e-8f;
            if (fabs(dw[c])>1e-4f) EXPECT_LE(delta*dw[c],1e-7f);
        }
        EXPECT_TRUE(updated);
        EXPECT_EQ(network.architectureJson(),before);
    }
    filesystem::remove_all(archive);
}
