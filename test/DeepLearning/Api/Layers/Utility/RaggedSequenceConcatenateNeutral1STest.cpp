#include "DeepLearning/Api/Layers/Utility/RaggedSequenceConcatenate.h"
#include "DeepLearning/Api/Layers/Utility/RaggedToPaddedDense.h"
#include "DeepLearning/Api/Layers/Utility/Slice.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Layers/Loss/CustomLoss.h"
#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Utility/RaggedSequenceConcatenate.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Common/Event.h"
#include "test/DeepLearning/Api/Helpers/GradientRivet.h"
#include "gtest/gtest.h"
#include "cuda_runtime.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace Thor;
using namespace std;
using json = nlohmann::json;
namespace Impl = ThorImplementation;
namespace {
RaggedTensor input1S(Network& n, const string& name, uint64_t m) {
    return RaggedNetworkInput::Builder().network(n).name(name)
        .valuesDataType(DataType::FP32).offsetsDataType(DataType::UINT32)
        .trailingDimensions({2}).maxValuesPerRow(m).build();
}
Impl::DynamicExpression loss1S(bool derivative) {
    auto p = Impl::Expression::input("predictions", DataType::FP32, DataType::FP32);
    auto y = Impl::Expression::input("labels", DataType::FP32, DataType::FP32);
    auto d = p - y;
    auto v = derivative ? d * Impl::Expression(2.0 * Impl::Loss::getLossScalingFactor()) : d * d;
    return Impl::DynamicExpression::fromExpressionDefinition(
        Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({{
            derivative ? "predictions_grad" : "loss", v.withOutputDType(DataType::FP32)}})));
}
float value1S(uint32_t source, uint64_t token, uint32_t f) {
    return 0.23f * static_cast<float>(source + 1) + 0.071f * static_cast<float>(token) - 0.11f * static_cast<float>(f);
}
float delta1S(uint32_t row, uint32_t column, uint32_t f) {
    return ((row + column + f) % 2 ? -1.f : 1.f) *
        (0.06f + 0.03f * static_cast<float>((row + 2 * column + f) % 7));
}
vector<float> download1S(const Impl::Tensor& gpu, uint64_t n) {
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor host = gpu.clone(cpu);
    Impl::Tensor source = gpu;
    Stream stream = Stream::getNextDownloadStream(gpu.getPlacement().getDeviceNum());
    host.downloadSection(source, stream, 0, 0, n * sizeof(float));
    stream.synchronize();
    return vector<float>(host.getMemPtr<float>(), host.getMemPtr<float>() + n);
}
class ConcatenateStampProbe1S : public RaggedSequenceConcatenate {
public:
    explicit ConcatenateStampProbe1S(const RaggedSequenceConcatenate& layer) : RaggedSequenceConcatenate(layer) {}
    using RaggedSequenceConcatenate::stamp;
};
void reject1S(Network& n, const json& record) {
    EXPECT_ANY_THROW(RaggedSequenceConcatenate::deserialize(record, &n));
}
}

// Simulate historical packed dimensions and deliberately wrong obsolete
// physical capacity metadata without changing semantic tensor identities.
static void toHistoricalPackedReference(nlohmann::json& ref) {
    const uint64_t m = ref.at("max_values_per_row").get<uint64_t>();
    ref.erase("layer_ragged_reference_version");
    ref.erase("geometry_mode");
    ref.erase("max_values_per_row");
    ref["batch_size"] = 999U;
    ref["max_total_values"] = 123456U;
    ref["values"]["dimensions"][0] = m * 3U;
    ref["offsets"]["dimensions"][0] = 4U;
}

TEST(RaggedSequenceConcatenateNeutral1S, SemanticGeometryAndLegacy) {
    Network n("concat_1s_geometry");
    auto a = input1S(n, "a", 3);
    auto b = input1S(n, "b", 2);
    auto c = input1S(n, "c", 4);
    auto op = RaggedSequenceConcatenate::Builder().network(n)
        .featureInput(a).featureInput(b).featureInput(c).build();
    auto out = op.getRaggedFeatureOutput();
    EXPECT_EQ(out.getValuesDimensions(), (vector<uint64_t>{9,2}));
    EXPECT_EQ(out.getOffsetsDimensions(), (vector<uint64_t>{1}));
    EXPECT_NE(out.getRowPartitionToken(), a.getRowPartitionToken());
    EXPECT_NE(out.getRowPartitionToken(), b.getRowPartitionToken());
    EXPECT_NE(out.getRowPartitionToken(), c.getRowPartitionToken());
    EXPECT_EQ(op.getFeatureInputs().size(), 6U);
    auto record = op.architectureJson();
    EXPECT_EQ(record.at("version"), "1.1.0");
    for (auto& ref : record.at("ragged_inputs")) {
        EXPECT_FALSE(ref.contains("batch_size"));
        EXPECT_FALSE(ref.contains("max_total_values"));
    }
    EXPECT_FALSE(record.at("ragged_output").contains("batch_size"));
    EXPECT_NO_THROW(RaggedSequenceConcatenate::deserialize(record, &n));
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    for (uint32_t B : {2U,7U}) {
        EXPECT_EQ(op.getOutputTensorBytes(B), B * 9U * 2U * sizeof(float));
        EXPECT_GT(op.getFirstInstanceMemRequirementInBytes(B,cpu), op.getOutputTensorBytes(B));
        EXPECT_EQ(out.getDescriptorForPlacement(B).getOffsetsDescriptor().getDimensions(), (vector<uint64_t>{B+1U}));
    }
    EXPECT_ANY_THROW((void)op.getOutputTensorBytes(0));
    auto shared = RaggedSequenceConcatenate::Builder().network(n).featureInput(a)
        .featureInput(a.withValues(Tensor(DataType::FP32,{3,2}))).build();
    EXPECT_EQ(shared.getFeatureInputs().size(), 3U);
    EXPECT_EQ(shared.getRaggedFeatureOutput().getMaxValuesPerRow(), 6U);
    EXPECT_THROW((void)RaggedSequenceConcatenate::Builder().network(n)
        .featureInput(a).featureInput(a).build(), invalid_argument);
    Network fixed("concat_1s_fixed");
    auto fixedA = RaggedNetworkInput::Builder().network(fixed).name("a")
        .valuesDataType(DataType::FP32).offsetsDataType(DataType::UINT32)
        .trailingDimensions({2}).maxValuesPerRow(3).build();
    auto fixedB = RaggedNetworkInput::Builder().network(fixed).name("b")
        .valuesDataType(DataType::FP32).offsetsDataType(DataType::UINT32)
        .trailingDimensions({2}).maxValuesPerRow(2).build();
    auto old = RaggedSequenceConcatenate::Builder().network(fixed)
        .featureInput(fixedA).featureInput(fixedB).build();
    EXPECT_EQ(old.architectureJson().at("version"), "1.1.0");
    EXPECT_NO_THROW(RaggedSequenceConcatenate::deserialize(old.architectureJson(), &fixed));
    // The historical packed grammar is covered in ConcatHistoricalPackedGrammarSumsSemanticM.
    // Distinct values with the same logical partition still require matching trailing dimensions.
    EXPECT_THROW((void)RaggedSequenceConcatenate::Builder().network(n)
        .featureInput(a).featureInput(a.withValues(Tensor(DataType::FP32, {3, 3}))).build(),
        invalid_argument);
}

TEST(RaggedSequenceConcatenateNeutral1S, StrictReferences) {
    Network n("concat_1s_bad");
    auto a = input1S(n, "a", 3);
    auto b = input1S(n, "b", 2);
    auto op = RaggedSequenceConcatenate::Builder().network(n).featureInput(a).featureInput(b).build();
    json j = op.architectureJson(), bad;
    bad = j; bad["version"] = "2.0.0"; reject1S(n,bad);
    bad = j; bad["batch_size"] = 7; reject1S(n,bad);
    bad = j; bad["ragged_inputs"][0]["max_total_values"] = 6; reject1S(n,bad);
    bad = j; bad["ragged_inputs"][1]["values"]["id"] = a.getValues().getOriginalId(); reject1S(n,bad);
    bad = j; bad["ragged_inputs"][0]["values"]["data_type"] = DataType::FP16; reject1S(n,bad);
    bad = j; bad["ragged_inputs"][0]["offsets"]["dimensions"] = vector<uint64_t>{3}; reject1S(n,bad);
    bad = j; bad["ragged_output"]["batch_size"] = 2; reject1S(n,bad);
    bad = j; bad["ragged_output"]["values"]["dimensions"] = vector<uint64_t>{6,2}; reject1S(n,bad);
    bad = j; bad["ragged_output"]["offsets"]["id"] = b.getOffsets().getOriginalId(); reject1S(n,bad);
    bad = j; bad["ragged_output"]["values"]["id"] = a.getValues().getOriginalId(); reject1S(n,bad);
    bad = j; bad["ragged_output"]["max_values_per_row"] = 4; reject1S(n,bad);
    bad = j; bad["ragged_output"]["extra"] = 9; reject1S(n,bad);
    bad = j; bad["ragged_inputs"] = json::array({j["ragged_inputs"][0]}); reject1S(n,bad);
}

TEST(RaggedSequenceConcatenateNeutral1S, EachPortCanDriveStamp) {
    Network n("concat_1s_stamps");
    auto a = input1S(n, "a", 3);
    auto b = input1S(n, "b", 2);
    auto op = RaggedSequenceConcatenate::Builder().network(n).featureInput(a).featureInput(b).build();
    ConcatenateStampProbe1S probe(op);
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    for (uint32_t B : {2U,7U}) {
        auto da = a.getDescriptorForPlacement(B), db = b.getDescriptorForPlacement(B);
        auto va = make_shared<Impl::NetworkInput>(cpu,DataType::FP32,da.getValuesDimensions());
        auto vb = make_shared<Impl::NetworkInput>(cpu,DataType::FP32,db.getValuesDimensions());
        auto oa = make_shared<Impl::NetworkInput>(cpu,DataType::UINT32,da.getOffsetsDescriptor().getDimensions());
        auto ob = make_shared<Impl::NetworkInput>(cpu,DataType::UINT32,db.getOffsetsDescriptor().getDimensions());
        EXPECT_NE(probe.stamp(cpu,va,nullptr,a.getValues(),true),nullptr);
        EXPECT_NE(probe.stamp(cpu,vb,nullptr,b.getValues(),true),nullptr);
        EXPECT_NE(probe.stamp(cpu,va,nullptr,a.getOffsets(),true),nullptr); // HOST_EXTENT values carrier
        EXPECT_NE(probe.stamp(cpu,vb,nullptr,b.getOffsets(),true),nullptr);
        EXPECT_NE(probe.stamp(cpu,oa,nullptr,a.getOffsets(),true),nullptr); // legacy real offsets carrier
        EXPECT_NE(probe.stamp(cpu,ob,nullptr,b.getOffsets(),true),nullptr);
        auto bad = make_shared<Impl::NetworkInput>(cpu,DataType::FP32,vector<uint64_t>{B*3U,3});
        EXPECT_ANY_THROW((void)probe.stamp(cpu,bad,nullptr,a.getValues(),true));
    }
}

TEST(RaggedSequenceConcatenateNeutral1S, SavedThreeInputForwardBackwardAtTwoAndSeven) {
    int count=0;
    if (cudaGetDeviceCount(&count)!=cudaSuccess || count==0) GTEST_SKIP() << "CUDA device unavailable";
    const filesystem::path path = filesystem::temp_directory_path() /
        ("concat_1s_"+to_string(chrono::steady_clock::now().time_since_epoch().count()));
    filesystem::remove_all(path);
    const vector<string> names{"a","b","c"};
    const vector<uint32_t> maxima{3,2,4};
    uint64_t qId=0;
    {
        Network n("concat_1s_training");
        vector<RaggedTensor> sources;
        RaggedSequenceConcatenate::Builder builder;
        builder.network(n);
        for (uint32_t i=0;i<3;++i) {
            auto source=input1S(n,names[i],maxima[i]);
            sources.push_back(source);
            auto rivet=GradientRivet::Builder().network(n).tensor(source.getValues()).build();
            builder.featureInput(source.withValues(rivet.getFeatureOutput().value()));
        }
        auto op=builder.build();
        auto out=op.getRaggedFeatureOutput();
        qId=out.getRowPartitionToken().getOriginalId();
        auto pad=RaggedToPaddedDense::Builder().network(n).featureInput(out).paddingValue(-8.0).build();
        auto labels=NetworkInput::Builder().network(n).name("targets")
            .dataType(DataType::FP32).dimensions({9,2}).build();
        auto loss=CustomLoss::Builder().network(n)
            .predictions(pad.getPaddedFeatureOutput()).labels(labels.getFeatureOutput().value())
            .lossExpression(loss1S(false)).gradientExpression(loss1S(true)).reportsRawLoss().build();
        (void)NetworkOutput::Builder().network(n).name("joined")
            .inputTensor(pad.getPaddedFeatureOutput()).dataType(DataType::FP32).build();
        (void)NetworkOutput::Builder().network(n).name("raw_loss")
            .inputTensor(loss.getRawLoss()).dataType(DataType::FP32).build();
        n.save(path.string(),true);
    }
    for (uint32_t B : {2U,7U}) {
        SCOPED_TRACE(::testing::Message()<<"B="<<B);
        Network n("concat_1s_training");
        ASSERT_NO_THROW(n.load(path.string()));
        shared_ptr<RaggedSequenceConcatenate> op;
        for (uint32_t i=0;i<n.getNumLayers();++i)
            if (auto c=dynamic_pointer_cast<RaggedSequenceConcatenate>(n.getLayer(i))) op=c;
        ASSERT_NE(op,nullptr);
        const auto out=op->getRaggedFeatureOutput();
        EXPECT_EQ(out.getRowPartitionToken().getOriginalId(),qId);
        EXPECT_EQ(out.getMaxValuesPerRow(),9U);
        const json before=n.architectureJson();
        vector<Event> init;
        auto placed=n.place(B,init,false);
        ASSERT_NE(placed,nullptr);
        for (auto& event:init) event.synchronize();
        auto& stamped=placed->getStampedNetwork(0);
        auto physical=dynamic_pointer_cast<Impl::RaggedSequenceConcatenate>(
            stamped.getPhysicalLayerFromApiLayer(op->getId()));
        ASSERT_NE(physical,nullptr);
        auto ports=physical->getFeatureInputs();
        ASSERT_EQ(ports.size(),6U);
        for (uint32_t i=0;i<3;++i) {
            ASSERT_TRUE(ports[i].has_value());
            EXPECT_EQ(ports[i]->getDimensions(),(vector<uint64_t>{B*maxima[i],2}));
        }
        ASSERT_TRUE(physical->getFeatureOutput().has_value());
        EXPECT_EQ(physical->getFeatureOutput()->getDimensions(),(vector<uint64_t>{B*9U,2}));
        EXPECT_EQ(n.architectureJson(),before);
        vector<vector<uint64_t>> offsets(3,vector<uint64_t>(B+1,0));
        vector<uint64_t> qOffsets(B+1,0);
        for (uint32_t row=0;row<B;++row) {
            for (uint32_t i=0;i<3;++i) {
                const uint32_t length=(row+i)%4==0?0U:(row+i)%4==1?maxima[i]:
                    (row+i)%4==2?1U:min<uint32_t>(2U,maxima[i]);
                offsets[i][row+1]=offsets[i][row]+length;
                qOffsets[row+1]+=length;
            }
            qOffsets[row+1]+=qOffsets[row];
        }
        Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
        Batch batch;
        for (uint32_t i=0;i<3;++i) {
            auto source=op->getRaggedFeatureInputs()[i];
            auto desc=source.getDescriptorForPlacement(B);
            Impl::Tensor off(cpu,desc.getOffsetsDescriptor());
            for (uint32_t row=0;row<=B;++row)
                off.getMemPtr<uint32_t>()[row]=static_cast<uint32_t>(offsets[i][row]);
            Impl::RowPartitionRuntime part(off,desc.getRowPartition());
            part.setHostOffsets(offsets[i]);
            Impl::Tensor vals(cpu,desc.getValuesDescriptor());
            for (uint64_t t=0;t<B*maxima[i];++t)
                for (uint32_t f=0;f<2;++f)
                    vals.getMemPtr<float>()[t*2+f]=t<offsets[i].back()?value1S(i,t,f):
                        numeric_limits<float>::quiet_NaN();
            batch.insert(names[i],Impl::RaggedTensor(vals,part));
        }
        Impl::Tensor targets(cpu,Impl::TensorDescriptor(DataType::FP32,{B,9,2}));
        for (uint32_t row=0;row<B;++row) {
            uint32_t col=0;
            for (uint32_t i=0;i<3;++i) {
                const uint32_t len=static_cast<uint32_t>(offsets[i][row+1]-offsets[i][row]);
                for (uint32_t j=0;j<len;++j,++col)
                    for (uint32_t f=0;f<2;++f)
                        targets.getMemPtr<float>()[(row*9+col)*2+f]=
                            value1S(i,offsets[i][row]+j,f)-delta1S(row,col,f);
            }
            for (;col<9;++col)
                for (uint32_t f=0;f<2;++f)
                    targets.getMemPtr<float>()[(row*9+col)*2+f]=-8.f-delta1S(row,col,f);
        }
        batch.insert("targets",targets);
        batch.setValidExampleCount(B);
        map<string,Impl::Tensor> results;
        map<string,Event> ready;
        auto done=placed->submitBatch(0,batch,results,ready,false);
        done.synchronize(); placed->synchronize();
        ASSERT_TRUE(results.contains("joined"));
        auto forward=download1S(results.at("joined"),static_cast<uint64_t>(B)*9*2);
        const auto grads=physical->getErrorOutputs();
        ASSERT_EQ(grads.size(),6U);
        for (uint32_t i=0;i<3;++i) ASSERT_TRUE(grads[i].has_value()) << i;
        for (uint32_t i=3;i<6;++i) EXPECT_FALSE(grads[i].has_value());
        for (uint32_t row=0;row<B;++row) {
            uint32_t col=0;
            for (uint32_t i=0;i<3;++i) {
                const uint32_t len=static_cast<uint32_t>(offsets[i][row+1]-offsets[i][row]);
                for (uint32_t j=0;j<len;++j,++col)
                    for (uint32_t f=0;f<2;++f)
                        EXPECT_NEAR(forward[(row*9+col)*2+f],value1S(i,offsets[i][row]+j,f),3e-4f);
            }
            for (;col<9;++col)
                for (uint32_t f=0;f<2;++f)
                    EXPECT_NEAR(forward[(row*9+col)*2+f],-8.f,3e-4f);
        }
        for (uint32_t i=0;i<3;++i) {
            auto grad=download1S(grads[i].value(),offsets[i].back()*2);
            for (uint32_t row=0;row<B;++row) {
                uint32_t baseCol=0;
                for (uint32_t prior=0;prior<i;++prior)
                    baseCol+=static_cast<uint32_t>(offsets[prior][row+1]-offsets[prior][row]);
                for (uint64_t j=0;j<offsets[i][row+1]-offsets[i][row];++j)
                    for (uint32_t f=0;f<2;++f) {
                        const float expected=2.f*delta1S(row,baseCol+j,f)*Impl::Loss::getLossScalingFactor();
                        EXPECT_NEAR(grad[(offsets[i][row]+j)*2+f],expected,7e-3f)
                            << "gradient source=" << i << " row=" << row << " token=" << j;
                    }
            }
        }
    }
    filesystem::remove_all(path);
}

TEST(RaggedSequenceConcatenateNeutral1S, NeutralCloneRetainsNewPartition) {
    Network source("concat_1s_clone_source");
    const auto a = input1S(source, "a", 3);
    const auto b = input1S(source, "b", 2);
    const auto concat = RaggedSequenceConcatenate::Builder().network(source)
        .featureInput(a).featureInput(b).build();
    (void)RaggedNetworkOutput::Builder().network(source).name("joined")
        .inputTensor(concat.getRaggedFeatureOutput()).build();
    Network destination("concat_1s_clone_destination");
    const auto da = input1S(destination, "a", 3);
    const auto db = input1S(destination, "b", 2);
    ApiTensorRemap remap;
    remap.map(a.getValues(),da.getValues());
    remap.map(a.getRowPartitionToken(),da.getRowPartitionToken());
    remap.map(b.getValues(),db.getValues());
    remap.map(b.getRowPartitionToken(),db.getRowPartitionToken());
    ApiSubgraphCloneOptions options;
    options.inferenceOnly=true;
    auto cloned=destination.cloneSubgraphInto(source,{"__thor_ragged_output.joined.values"},remap,options);
    ASSERT_EQ(cloned.outputTensorsByName.size(),1U);
    EXPECT_EQ(cloned.outputTensorsByName.at("__thor_ragged_output.joined.values").getDimensions(),
        (vector<uint64_t>{5,2}));
    auto q=cloned.clonedTensorBySourceOriginalId.at(
        concat.getRaggedFeatureOutput().getRowPartitionToken().getOriginalId());
    EXPECT_NE(q,da.getRowPartitionToken());
    EXPECT_NE(q,db.getRowPartitionToken());
    EXPECT_EQ(q.getDimensions(),(vector<uint64_t>{1}));
}

TEST(RaggedSequenceConcatenateNeutral1S, SavedSharedPartitionForwardAtTwoAndSeven) {
    int deviceCount=0;
    if (cudaGetDeviceCount(&deviceCount)!=cudaSuccess || deviceCount==0)
        GTEST_SKIP() << "CUDA device unavailable";
    const filesystem::path path=filesystem::temp_directory_path() /
        ("concat_1s_shared_"+to_string(chrono::steady_clock::now().time_since_epoch().count()));
    filesystem::remove_all(path);
    uint64_t qId=0;
    {
        Network n("concat_1s_shared");
        auto source=RaggedNetworkInput::Builder().network(n).name("source")
            .valuesDataType(DataType::FP32).offsetsDataType(DataType::UINT32)
            .trailingDimensions({4}).maxValuesPerRow(3).build();
        auto left=Slice::Builder().network(n).featureInput(source).axis(0).start(0).length(2).build();
        auto right=Slice::Builder().network(n).featureInput(source).axis(0).start(2).length(2).build();
        ASSERT_TRUE(left.getRaggedFeatureOutput().has_value());
        ASSERT_TRUE(right.getRaggedFeatureOutput().has_value());
        auto joined=RaggedSequenceConcatenate::Builder().network(n)
            .featureInput(left.getRaggedFeatureOutput().value())
            .featureInput(right.getRaggedFeatureOutput().value()).build();
        EXPECT_EQ(joined.getFeatureInputs().size(),3U);
        qId=joined.getRaggedFeatureOutput().getRowPartitionToken().getOriginalId();
        auto padded=RaggedToPaddedDense::Builder().network(n)
            .featureInput(joined.getRaggedFeatureOutput()).paddingValue(-6.f).build();
        (void)NetworkOutput::Builder().network(n).name("joined")
            .inputTensor(padded.getPaddedFeatureOutput()).dataType(DataType::FP32).build();
        n.save(path.string(),true);
    }
    for (uint32_t B : {2U,7U}) {
        SCOPED_TRACE(::testing::Message()<<"B="<<B);
        Network n("concat_1s_shared");
        ASSERT_NO_THROW(n.load(path.string()));
        shared_ptr<RaggedSequenceConcatenate> joined;
        for (uint32_t i=0;i<n.getNumLayers();++i)
            if (auto op=dynamic_pointer_cast<RaggedSequenceConcatenate>(n.getLayer(i))) joined=op;
        ASSERT_NE(joined,nullptr);
        EXPECT_EQ(joined->getRaggedFeatureOutput().getRowPartitionToken().getOriginalId(),qId);
        const auto inputs=n.getExternalRaggedNetworkInputs();
        ASSERT_EQ(inputs.size(),1U);
        const auto source=inputs.front().raggedTensor;
        const auto jsonBefore=n.architectureJson();
        vector<Event> init;
        auto placed=n.place(B,init,true);
        ASSERT_NE(placed,nullptr);
        for (auto& event:init) event.synchronize();
        auto& stamped=placed->getStampedNetwork(0);
        auto physical=dynamic_pointer_cast<Impl::RaggedSequenceConcatenate>(
            stamped.getPhysicalLayerFromApiLayer(joined->getId()));
        ASSERT_NE(physical,nullptr);
        const auto ports=physical->getFeatureInputs();
        ASSERT_EQ(ports.size(),3U);
        EXPECT_EQ(ports[0]->getDimensions(),(vector<uint64_t>{B*3U,2}));
        EXPECT_EQ(ports[1]->getDimensions(),(vector<uint64_t>{B*3U,2}));
        ASSERT_TRUE(physical->getFeatureOutput().has_value());
        EXPECT_EQ(physical->getFeatureOutput()->getDimensions(),(vector<uint64_t>{B*6U,2}));
        EXPECT_EQ(n.architectureJson(),jsonBefore);
        vector<uint64_t> offsets(B+1,0);
        for (uint32_t row=0;row<B;++row)
            offsets[row+1]=offsets[row]+(row%3==0?0U:row%3==1?3U:1U);
        Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
        auto desc=source.getDescriptorForPlacement(B);
        Impl::Tensor off(cpu,desc.getOffsetsDescriptor());
        for (uint32_t row=0;row<=B;++row)
            off.getMemPtr<uint32_t>()[row]=static_cast<uint32_t>(offsets[row]);
        Impl::RowPartitionRuntime partition(off,desc.getRowPartition());
        partition.setHostOffsets(offsets);
        Impl::Tensor vals(cpu,desc.getValuesDescriptor());
        for (uint64_t token=0;token<B*3ULL;++token)
            for (uint32_t f=0;f<4;++f)
                vals.getMemPtr<float>()[token*4+f]=token<offsets.back()?
                    0.15f*static_cast<float>(token)+0.08f*static_cast<float>(f):
                    numeric_limits<float>::quiet_NaN();
        Batch batch;
        batch.insert("source",Impl::RaggedTensor(vals,partition));
        batch.setValidExampleCount(B);
        map<string,Impl::Tensor> outputs;
        map<string,Event> ready;
        auto done=placed->submitBatch(0,batch,outputs,ready,false);
        done.synchronize(); placed->synchronize();
        ASSERT_TRUE(outputs.contains("joined"));
        auto actual=download1S(outputs.at("joined"),static_cast<uint64_t>(B)*6*2);
        for (uint32_t row=0;row<B;++row) {
            const uint64_t len=offsets[row+1]-offsets[row];
            for (uint32_t col=0;col<6;++col)
                for (uint32_t f=0;f<2;++f) {
                    const bool active=col<2*len;
                    const uint64_t sourceFeature=col<len?f:f+2;
                    const float expected=active?
                        0.15f*static_cast<float>(offsets[row]+(col%len))+
                            0.08f*static_cast<float>(sourceFeature):-6.f;
                    EXPECT_NEAR(actual[(row*6+col)*2+f],expected,3e-4f);
                }
        }
    }
    filesystem::remove_all(path);
}

TEST(RaggedSequenceConcatenateNeutral1S, ConcatHistoricalPackedGrammarSumsSemanticM) {
    Network n("concat_old_reader");
    const auto a = input1S(n, "a", 3);
    const auto b = input1S(n, "b", 2);
    const auto op = RaggedSequenceConcatenate::Builder().network(n).featureInput(a).featureInput(b).build();
    json old = op.architectureJson();
    old["version"] = "1.0.0";
    for (auto& ref : old["ragged_inputs"]) toHistoricalPackedReference(ref);
    toHistoricalPackedReference(old["ragged_output"]);
    ASSERT_NO_THROW(RaggedSequenceConcatenate::deserialize(old, &n));
    for (uint64_t b : {2U, 7U}) {
        const auto descriptor = op.getRaggedFeatureOutput().getDescriptorForPlacement(b);
        EXPECT_EQ(descriptor.getValuesDimensions(), (vector<uint64_t>{b * 5U, 2U}));
        EXPECT_EQ(descriptor.getOffsetsDescriptor().getDimensions(), (vector<uint64_t>{b + 1U}));
    }
    old["ragged_output"]["offsets"]["data_type"] = DataType::UINT64;
    EXPECT_ANY_THROW(RaggedSequenceConcatenate::deserialize(old, &n));
}
