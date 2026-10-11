// 2C-G2: saved neutral SDPA dropout backward, active-value numerical reference,
// and deterministic per-instance replay.  Test-only; physical cuDNN SDPA stays unchanged.
#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Layers/Learning/ScaledDotProductAttention.h"
#include "DeepLearning/Api/Layers/Loss/RaggedCustomLoss.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Network/StampedNetwork.h"
#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"
#include "test/DeepLearning/Api/Helpers/GradientRivet.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "gtest/gtest.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace Api = Thor;
namespace Impl = ThorImplementation;
using DataType = Impl::DataType;
constexpr uint64_t MQG2 = 4, MKG2 = 5, HQG2 = 4, HKG2 = 2, DG2 = 32;
constexpr double DropScaleG2 = 2.0; // probability=0.5, inverted dropout
constexpr int64_t SeedG2 = 371223, OffsetG2 = 29;
const double ScoreScaleG2 = 1.0 / std::sqrt(static_cast<double>(DG2));

std::vector<uint64_t> offsetsG2(uint32_t b, bool query) {
    std::vector<uint64_t> result(b + 1, 0);
    for (uint32_t row = 0; row < b; ++row) {
        const uint64_t n = query ? (row % 4 == 0 ? 0u : 2u + row % 2u)
                                 : (2u + row % 3u);
        result[row+1] = result[row] + n;
    }
    return result;
}

float qkValueG2(int kind, uint64_t token, uint64_t head, uint64_t dim) {
    const float sample = .07f * (kind+1) + .013f * static_cast<float>(
        (token*11 + head*7 + dim*5 + kind*3) % 31) - .16f
        + .009f * static_cast<float>(dim%7);
    return __half2float(__float2half(sample));
}

Impl::Tensor hostOffsetsG2(const std::vector<uint64_t>& offsets, DataType dtype) {
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor tensor(cpu, Impl::TensorDescriptor(dtype, {offsets.size()}));
    for (size_t i=0; i<offsets.size(); ++i) {
        if (dtype == DataType::UINT32) tensor.getMemPtr<uint32_t>()[i] = static_cast<uint32_t>(offsets[i]);
        else tensor.getMemPtr<uint64_t>()[i] = offsets[i];
    }
    return tensor;
}

Impl::Tensor hostValuesG2(uint32_t b, uint64_t m, uint64_t heads,
                          const std::vector<uint64_t>& offsets, int kind) {
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor tensor(cpu, Impl::TensorDescriptor(DataType::FP16, {b*m, heads, DG2}));
    auto* p = tensor.getMemPtr<half>();
    uint64_t row = 0;
    for (uint64_t token=0; token<b*m; ++token) {
        while (row+1<offsets.size() && token>=offsets[row+1]) ++row;
        for (uint64_t h=0; h<heads; ++h)
            for (uint64_t d=0; d<DG2; ++d) {
                // Each live V token is a *different basis vector within its row*.
                // This makes the binary dropout decisions directly observable in Y:
                // Y[local_key] == softmax[local_key] * inverted_dropout_mask.
                const float value = kind == 2 ? (d == token-offsets[row] ? 1.f : 0.f)
                                               : qkValueG2(kind, token, h, d);
                p[(token*heads+h)*DG2+d] = token<offsets.back()
                    ? __float2half(value)
                    : __float2half(std::numeric_limits<float>::quiet_NaN());
            }
    }
    return tensor;
}

Batch makeBatchG2(uint32_t b, const Api::RaggedTensor& q, const Api::RaggedTensor& k,
                  const Api::RaggedTensor& v, const Api::RaggedTensor& labels) {
    const auto qo=offsetsG2(b,true), ko=offsetsG2(b,false);
    Impl::RowPartitionRuntime qp(hostOffsetsG2(qo, DataType::UINT32),
        q.getDescriptorForPlacement(b).getRowPartition());
    Impl::RowPartitionRuntime kp(hostOffsetsG2(ko, DataType::UINT64),
        k.getDescriptorForPlacement(b).getRowPartition());
    qp.setHostOffsets(qo);
    kp.setHostOffsets(ko);
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor zeros(cpu,labels.getDescriptorForPlacement(b).getValuesDescriptor());
    for (uint64_t i=0; i<b*MQG2*HQG2*DG2; ++i)
        zeros.getMemPtr<half>()[i] = i<qo.back()*HQG2*DG2
            ? __float2half(0.f) : __float2half(std::numeric_limits<float>::quiet_NaN());
    Batch batch;
    batch.insert("q", Impl::RaggedTensor(hostValuesG2(b,MQG2,HQG2,qo,0), qp));
    batch.insert("k", Impl::RaggedTensor(hostValuesG2(b,MKG2,HKG2,ko,1), kp));
    batch.insert("v", Impl::RaggedTensor(hostValuesG2(b,MKG2,HKG2,ko,2), kp));
    batch.insert("labels", Impl::RaggedTensor(zeros, qp));
    batch.setValidExampleCount(b);
    return batch;
}

std::vector<float> readHalfG2(const Impl::Tensor& src) {
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor dst = src.clone(cpu);
    Impl::Tensor copy = src;
    Stream stream=Stream::getNextDownloadStream(src.getPlacement().getDeviceNum());
    dst.copyFromAsync(copy,stream);
    stream.synchronize();
    const half* data=dst.getMemPtr<half>();
    std::vector<float> result(dst.getTotalNumElements());
    for (size_t i=0; i<result.size(); ++i) result[i]=__half2float(data[i]);
    return result;
}

Impl::DynamicExpression lossG2(bool gradient) {
    auto p=Impl::Expression::input("predictions",DataType::FP16,DataType::FP32);
    auto y=Impl::Expression::input("labels",DataType::FP16,DataType::FP32);
    auto diff=p-y;
    auto result=gradient ? diff*Impl::Expression(2.*Impl::Loss::getLossScalingFactor()) : diff*diff;
    return Impl::DynamicExpression::fromExpressionDefinition(
        Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({{
            gradient?"predictions_grad":"loss",result.withOutputDType(DataType::FP16)}})));
}

std::shared_ptr<Api::ScaledDotProductAttention> findG2(Api::Network& net) {
    for (uint32_t i=0; i<net.getNumLayers(); ++i)
        if (auto found=std::dynamic_pointer_cast<Api::ScaledDotProductAttention>(net.getLayer(i)))
            return found;
    return {};
}

std::filesystem::path archiveG2() {
    return std::filesystem::temp_directory_path()/(
        "thor_sdpa_dropout_backward_g2_"+
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
}

void saveG2(const std::filesystem::path& path) {
    Api::Network net("sdpa_dropout_backward_2cg2");
    const auto q=Api::RaggedNetworkInput::Builder().network(net).name("q")
        .valuesDataType(DataType::FP16).offsetsDataType(DataType::UINT32)
        .trailingDimensions({HQG2,DG2}).maxValuesPerRow(MQG2).build();
    const auto k=Api::RaggedNetworkInput::Builder().network(net).name("k")
        .valuesDataType(DataType::FP16).offsetsDataType(DataType::UINT64)
        .trailingDimensions({HKG2,DG2}).maxValuesPerRow(MKG2).build();
    const auto v=Api::RaggedNetworkInput::Builder().network(net).name("v")
        .valuesDataType(DataType::FP16).trailingDimensions({HKG2,DG2}).partition(k).build();
    const auto labels=Api::RaggedNetworkInput::Builder().network(net).name("labels")
        .valuesDataType(DataType::FP16).trailingDimensions({HQG2,DG2}).partition(q).build();
    const auto qr=Api::GradientRivet::Builder().network(net).tensor(q.getValues()).build();
    const auto kr=Api::GradientRivet::Builder().network(net).tensor(k.getValues()).build();
    const auto vr=Api::GradientRivet::Builder().network(net).tensor(v.getValues()).build();
    const auto attention=Api::ScaledDotProductAttention::Builder().network(net)
        .queryInput(q.withValues(*qr.getFeatureOutput()))
        .keyInput(k.withValues(*kr.getFeatureOutput()))
        .valueInput(v.withValues(*vr.getFeatureOutput()))
        .computeDataType(DataType::FP32).dropout(0.5f,SeedG2,OffsetG2).build();
    if (!attention.getRaggedFeatureOutput().has_value())
        throw std::runtime_error("Expected ragged SDPA output");
    (void)Api::RaggedCustomLoss::Builder().network(net)
        .predictions(*attention.getRaggedFeatureOutput()).labels(labels)
        .lossExpression(lossG2(false)).gradientExpression(lossG2(true)).build();
    // The RAW loss also has a reporting consumer; it must remain the training root.
    for (uint32_t i=0; i<net.getNumLayers(); ++i)
        if (auto loss=std::dynamic_pointer_cast<Api::RaggedCustomLoss>(net.getLayer(i))) {
            (void)Api::NetworkOutput::Builder().network(net).name("raw_loss")
                .inputTensor(loss->getRaggedRawLoss().getValues()).dataType(DataType::FP16).build();
            break;
        }
    net.save(path.string(),true);
}

struct ReferenceG2 {
    std::vector<double> dq,dk,dv;
    size_t kept=0, dropped=0;
};

ReferenceG2 referenceG2(uint32_t b, const std::vector<float>& output) {
    const auto qo=offsetsG2(b,true), ko=offsetsG2(b,false);
    ReferenceG2 result;
    result.dq.resize(b*MQG2*HQG2*DG2,0.0);
    result.dk.resize(b*MKG2*HKG2*DG2,0.0);
    result.dv.resize(b*MKG2*HKG2*DG2,0.0);
    for (uint32_t row=0; row<b; ++row)
        for (uint64_t qi=qo[row]; qi<qo[row+1]; ++qi)
            for (uint64_t h=0; h<HQG2; ++h) {
                const uint64_t kh=h/(HQG2/HKG2);
                const size_t n=static_cast<size_t>(ko[row+1]-ko[row]);
                std::vector<double> probs(n), weight(n);
                double maxScore=-std::numeric_limits<double>::infinity();
                for (size_t j=0; j<n; ++j) {
                    double score=0;
                    for (uint64_t d=0; d<DG2; ++d)
                        score+=qkValueG2(0,qi,h,d)*qkValueG2(1,ko[row]+j,kh,d);
                    probs[j]=score*ScoreScaleG2;
                    maxScore=std::max(maxScore,probs[j]);
                }
                double denom=0;
                for (double& p:probs) {p=std::exp(p-maxScore);denom+=p;}
                for (double& p:probs) p/=denom;
                const uint64_t base=(qi*HQG2+h)*DG2;
                for (size_t j=0; j<n; ++j) {
                    const float observed=output.at(base+j);
                    if (!std::isfinite(observed))
                        throw std::runtime_error("Nonfinite active attention dropout output");
                    const bool kept=observed>0.05f;
                    weight[j]=kept ? DropScaleG2 : 0.0;
                    kept ? ++result.kept : ++result.dropped;
                    // The one-hot V basis makes this an independent *numerical*
                    // test of the dropout-scaled softmax probability, not merely
                    // a self-consistency check between forward and backward.
                    EXPECT_NEAR(observed, probs[j]*weight[j], .012)
                        << "row="<<row<<" head="<<h<<" key="<<j;
                }
                for (uint64_t d=n; d<DG2; ++d)
                    EXPECT_NEAR(output.at(base+d),0.f,.004)
                        << "Unexpected non-basis output at row="<<row<<" d="<<d;
                // RAW MSE dY=2*FP16(Y) (the loss layer scales gradients separately).
                std::vector<double> dp(n);
                double mean=0;
                for (size_t j=0; j<n; ++j) {
                    const double dy=2.0*output[base+j];
                    dp[j]=weight[j]*dy;
                    mean+=probs[j]*dp[j];
                    for (uint64_t d=0; d<DG2; ++d)
                        result.dv[((ko[row]+j)*HKG2+kh)*DG2+d] +=
                            probs[j]*weight[j]*2.0*output[base+d];
                }
                for (size_t j=0; j<n; ++j) {
                    const double ds=probs[j]*(dp[j]-mean)*ScoreScaleG2;
                    for (uint64_t d=0; d<DG2; ++d) {
                        result.dq[(qi*HQG2+h)*DG2+d] += ds*qkValueG2(1,ko[row]+j,kh,d);
                        result.dk[((ko[row]+j)*HKG2+kh)*DG2+d] += ds*qkValueG2(0,qi,h,d);
                    }
                }
            }
    return result;
}

void compareG2(const std::vector<float>& actual,const std::vector<double>& expected,
               uint64_t active,const char* component) {
    ASSERT_GE(actual.size(),active);
    const double scale=Impl::Loss::getLossScalingFactor();
    bool nonzero=false;
    for (uint64_t i=0; i<active; ++i) {
        SCOPED_TRACE(::testing::Message()<<component<<" i="<<i);
        ASSERT_TRUE(std::isfinite(actual[i]));
        EXPECT_NEAR(actual[i]/scale,expected[i],.012+std::abs(expected[i])*.20);
        nonzero |= std::abs(expected[i])>1e-5;
    }
    EXPECT_TRUE(nonzero)<<component<<" must exercise a nonzero reference gradient";
}

// Physical ragged buffers have B*M capacity, but only the compact active
// prefix is defined.  Padded tails can contain NaNs or uninitialized data;
// neither may determine dropout replay or mask-advancement assertions.
std::vector<float> activePrefixG2(const std::vector<float>& packed, uint64_t active) {
    if (packed.size() < active)
        throw std::runtime_error("Physical ragged buffer is shorter than its active prefix");
    return std::vector<float>(packed.begin(), packed.begin() + active);
}

struct StepG2 { std::vector<float> output,dq,dk,dv; };
struct InstanceG2 {
    std::unique_ptr<Api::Network> net;
    std::shared_ptr<Api::PlacedNetwork> placed;
    std::shared_ptr<Impl::CustomLayer> physical;
    Api::RaggedTensor q,k,v,labels;
    nlohmann::json architecture;
};

InstanceG2 loadG2(const std::filesystem::path& archive,uint32_t b) {
    InstanceG2 instance;
    instance.net=std::make_unique<Api::Network>("sdpa_dropout_backward_2cg2");
    instance.net->load(archive.string());
    const auto attention=findG2(*instance.net);
    if (!attention) throw std::runtime_error("Missing reloaded SDPA");
    if (attention->architectureJson().at("version")!="2.2.0")
        throw std::runtime_error("Expected neutral 2.2.0 SDPA schema");
    instance.architecture=instance.net->architectureJson();
    auto findInput=[&](const char* name)->Api::RaggedTensor {
        for (const auto& item:instance.net->getExternalRaggedNetworkInputs())
            if (item.name==name) return item.raggedTensor;
        throw std::runtime_error(std::string("Missing input ")+name);
    };
    instance.q=findInput("q"); instance.k=findInput("k");
    instance.v=findInput("v"); instance.labels=findInput("labels");
    if (instance.q.sharesPartitionWith(instance.k) || !instance.k.sharesPartitionWith(instance.v))
        throw std::runtime_error("Incorrect independent Q/KV partition ownership");
    std::vector<Event> events;
    instance.placed=instance.net->place(b,events,false);
    for (auto& e:events) e.synchronize();
    instance.physical=std::dynamic_pointer_cast<Impl::CustomLayer>(
        instance.placed->getStampedNetwork(0).getPhysicalLayerFromApiLayer(attention->getId()));
    if (!instance.physical || !instance.physical->getFeatureOutput().has_value())
        throw std::runtime_error("Missing physical attention output");
    if (instance.physical->getFeatureOutput()->getDimensions()!=
        std::vector<uint64_t>{b*MQG2,HQG2,DG2})
        throw std::runtime_error("Unexpected packed attention output capacity");
    return instance;
}

StepG2 trainStepG2(InstanceG2& instance,uint32_t b) {
    std::map<std::string,Impl::Tensor> outputs;
    std::map<std::string,Event> ready;
    auto done=instance.placed->submitBatch(0,makeBatchG2(b,instance.q,instance.k,instance.v,instance.labels),
                                           outputs,ready,false);
    done.synchronize();
    instance.placed->synchronize();
    if (!outputs.contains("raw_loss")) throw std::runtime_error("Missing RAW reporting output");
    const auto& gradients=instance.physical->getErrorOutputs();
    if (gradients.size()<3 || !gradients[0] || !gradients[1] || !gradients[2])
        throw std::runtime_error("Missing physical SDPA Q/K/V gradients");
    StepG2 result;
    result.output=readHalfG2(*instance.physical->getFeatureOutput());
    result.dq=readHalfG2(*gradients[0]);
    result.dk=readHalfG2(*gradients[1]);
    result.dv=readHalfG2(*gradients[2]);
    return result;
}
} // namespace

TEST(SdpaNeutralDropoutBackward2CG2, SavedB2B7NumericalGradientsAndIndependentTrainingReplay) {
    int devices=0;
    if (cudaGetDeviceCount(&devices)!=cudaSuccess || devices==0)
        GTEST_SKIP()<<"CUDA device required for real SDPA dropout backward";
    const auto archive=archiveG2();
    std::filesystem::remove_all(archive);
    saveG2(archive);
    for (uint32_t b:{2u,7u}) {
        SCOPED_TRACE(::testing::Message()<<"B="<<b);
        auto first=loadG2(archive,b);
        auto second=loadG2(archive,b);
        const auto a0=trainStepG2(first,b), a1=trainStepG2(first,b);
        const auto b0=trainStepG2(second,b), b1=trainStepG2(second,b);
        const auto activeQ=offsetsG2(b,true).back()*HQG2*DG2;
        const auto activeKV=offsetsG2(b,false).back()*HKG2*DG2;
        // Compare precisely the same active values from both stamped instances.
        // Full B*M buffers contain undefined inactive capacity and must not be
        // used for deterministic replay or dropout-mask advancement checks.
        EXPECT_EQ(activePrefixG2(a0.output,activeQ),activePrefixG2(b0.output,activeQ))
            << "first dropout training forward must replay";
        EXPECT_EQ(activePrefixG2(a1.output,activeQ),activePrefixG2(b1.output,activeQ))
            << "second dropout training forward must replay";
        EXPECT_EQ(activePrefixG2(a0.dq,activeQ),activePrefixG2(b0.dq,activeQ)) << "dQ replay";
        EXPECT_EQ(activePrefixG2(a1.dq,activeQ),activePrefixG2(b1.dq,activeQ))
            << "dQ second-step replay";
        EXPECT_EQ(activePrefixG2(a0.dk,activeKV),activePrefixG2(b0.dk,activeKV)) << "dK replay";
        EXPECT_EQ(activePrefixG2(a1.dk,activeKV),activePrefixG2(b1.dk,activeKV))
            << "dK second-step replay";
        EXPECT_EQ(activePrefixG2(a0.dv,activeKV),activePrefixG2(b0.dv,activeKV)) << "dV replay";
        EXPECT_EQ(activePrefixG2(a1.dv,activeKV),activePrefixG2(b1.dv,activeKV))
            << "dV second-step replay";
        EXPECT_NE(activePrefixG2(a0.output,activeQ),activePrefixG2(a1.output,activeQ))
            << "dropout offset must advance during training";
        for (const auto& step:{a0,a1}) {
            const auto reference=referenceG2(b,step.output);
            EXPECT_GT(reference.kept,0u);
            EXPECT_GT(reference.dropped,0u);
            compareG2(step.dq,reference.dq,offsetsG2(b,true).back()*HQG2*DG2,"dQ");
            compareG2(step.dk,reference.dk,offsetsG2(b,false).back()*HKG2*DG2,"dK");
            compareG2(step.dv,reference.dv,offsetsG2(b,false).back()*HKG2*DG2,"dV");
        }
        EXPECT_EQ(first.net->architectureJson(),first.architecture)
            <<"runtime dropout offsets must not be rewritten into the saved graph";
        EXPECT_EQ(second.net->architectureJson(),second.architecture);
    }
    std::filesystem::remove_all(archive);
}
