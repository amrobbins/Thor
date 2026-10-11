// 2C-F4: saved neutral SDPA with one shared K/V values producer.
// Check independent SDPA port gradients and the actual upstream fanout sum.
// No production changes, masking, or dropout in this bounded gate.
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
#include "test/DeepLearning/Implementation/Layers/Helpers/GradientRivet.h"
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
#include <string>
#include <variant>
#include <vector>

namespace {
namespace Api = Thor;
namespace Impl = ThorImplementation;
using DataType = Impl::DataType;
constexpr uint64_t MQ = 4, MK = 5, HQ = 4, HK = 2, D = 32;
const double SCALE = 1.0 / std::sqrt(static_cast<double>(D));

std::vector<uint64_t> offsetsF4(uint32_t b, bool query) {
    std::vector<uint64_t> v(b+1, 0);
    for (uint32_t i=0; i<b; ++i) {
        // Includes empty Q rows and nonuniform lengths; any active query
        // always has at least two keys (so dQ and dK are nontrivial).
        const uint64_t length = query ? (i%4==0 ? 0u : 1u+i%3)
                                      : (2u+(i*3u)%4u);
        v[i+1] = v[i]+length;
    }
    return v;
}

float sourceF4(int kind, uint64_t token, uint64_t head, uint64_t dim) {
    // The key and value ports are the exact same tensor and have identical data.
    if (kind == 2) kind = 1;
    const float x = .07f*(kind+1) + .013f*static_cast<float>(
        (token*11+head*7+dim*5+kind*3)%31) - .16f
        + .009f*static_cast<float>(dim%7);
    return __half2float(__float2half(x));
}

Impl::Tensor hostValuesF4(uint32_t b, uint64_t m, uint64_t h,
                          const std::vector<uint64_t>& off, int kind) {
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor value(cpu, Impl::TensorDescriptor(DataType::FP16,{b*m,h,D}));
    auto *p = value.getMemPtr<half>();
    for (uint64_t t=0; t<b*m; ++t)
        for (uint64_t head=0; head<h; ++head)
            for (uint64_t d=0; d<D; ++d)
                p[(t*h+head)*D+d] = t<off.back()
                    ? __float2half(sourceF4(kind,t,head,d))
                    : __float2half(std::numeric_limits<float>::quiet_NaN());
    return value;
}

Impl::Tensor hostOffsetsF4(const std::vector<uint64_t>& off, DataType dtype) {
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor result(cpu,Impl::TensorDescriptor(dtype,{off.size()}));
    for (size_t i=0;i<off.size();++i) {
        if (dtype==DataType::UINT32) result.getMemPtr<uint32_t>()[i]=static_cast<uint32_t>(off[i]);
        else result.getMemPtr<uint64_t>()[i]=off[i];
    }
    return result;
}

Batch makeBatchF4(uint32_t b, const Api::RaggedTensor& q,
                  const Api::RaggedTensor& kv,
                  const Api::RaggedTensor& labels) {
    auto qo=offsetsF4(b,true), ko=offsetsF4(b,false);
    auto qp=Impl::RowPartitionRuntime(hostOffsetsF4(qo,DataType::UINT32),
                                       q.getDescriptorForPlacement(b).getRowPartition());
    auto kp=Impl::RowPartitionRuntime(hostOffsetsF4(ko,DataType::UINT64),
                                       kv.getDescriptorForPlacement(b).getRowPartition());
    qp.setHostOffsets(qo);
    kp.setHostOffsets(ko);
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor zeros(cpu,labels.getDescriptorForPlacement(b).getValuesDescriptor());
    for (uint64_t i=0;i<b*MQ*HQ*D;++i)
        zeros.getMemPtr<half>()[i]=i<qo.back()*HQ*D
            ? __float2half(0.f) : __float2half(std::numeric_limits<float>::quiet_NaN());
    Batch batch;
    batch.insert("q", Impl::RaggedTensor(hostValuesF4(b,MQ,HQ,qo,0),qp));
    batch.insert("kv", Impl::RaggedTensor(hostValuesF4(b,MK,HK,ko,1),kp));
    batch.insert("labels", Impl::RaggedTensor(zeros,qp));
    batch.setValidExampleCount(b);
    return batch;
}

std::vector<float> readHalfF4(const Impl::Tensor& src) {
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor dst=src.clone(cpu);
    Impl::Tensor source=src;
    Stream stream=Stream::getNextDownloadStream(src.getPlacement().getDeviceNum());
    dst.copyFromAsync(source,stream);
    stream.synchronize();
    const half *values=dst.getMemPtr<half>();
    std::vector<float> result(dst.getTotalNumElements());
    for (size_t i=0;i<result.size();++i) result[i]=__half2float(values[i]);
    return result;
}

struct ReferenceF4 {
    std::vector<double> dq,dk,dv;
};

ReferenceF4 referenceF4(uint32_t b) {
    const auto qo=offsetsF4(b,true), ko=offsetsF4(b,false);
    ReferenceF4 r;
    r.dq.resize(b*MQ*HQ*D,0);
    r.dk.resize(b*MK*HK*D,0);
    r.dv.resize(b*MK*HK*D,0);
    for (uint32_t row=0;row<b;++row)
        for (uint64_t qi=qo[row];qi<qo[row+1];++qi)
            for (uint64_t h=0;h<HQ;++h) {
                const uint64_t kh=h/(HQ/HK);
                std::vector<double> scores;
                for (uint64_t ki=ko[row];ki<ko[row+1];++ki) {
                    double dot=0;
                    for (uint64_t d=0;d<D;++d)
                        dot+=sourceF4(0,qi,h,d)*sourceF4(1,ki,kh,d);
                    scores.push_back(dot*SCALE);
                }
                if (scores.empty()) continue;
                const double maximum=*std::max_element(scores.begin(),scores.end());
                double sum=0;
                for (auto& s:scores) {s=std::exp(s-maximum);sum+=s;}
                for (auto& s:scores) s/=sum;
                std::vector<double> dy(D), dp(scores.size());
                for (uint64_t d=0;d<D;++d) {
                    double y=0;
                    for (size_t j=0;j<scores.size();++j)
                        y+=scores[j]*sourceF4(2,ko[row]+j,kh,d);
                    // The loss consumes actual FP16 attention output.
                    dy[d]=2.*__half2float(__float2half(static_cast<float>(y)));
                }
                double mean=0;
                for (size_t j=0;j<scores.size();++j) {
                    const uint64_t ki=ko[row]+j;
                    double contribution=0;
                    for (uint64_t d=0;d<D;++d) {
                        contribution+=dy[d]*sourceF4(2,ki,kh,d);
                        r.dv[(ki*HK+kh)*D+d]+=scores[j]*dy[d];
                    }
                    dp[j]=contribution;
                    mean+=scores[j]*contribution;
                }
                for (size_t j=0;j<scores.size();++j) {
                    const uint64_t ki=ko[row]+j;
                    const double ds=scores[j]*(dp[j]-mean)*SCALE;
                    for (uint64_t d=0;d<D;++d) {
                        r.dq[(qi*HQ+h)*D+d]+=ds*sourceF4(1,ki,kh,d);
                        r.dk[(ki*HK+kh)*D+d]+=ds*sourceF4(0,qi,h,d);
                    }
                }
            }
    return r;
}

std::shared_ptr<Api::ScaledDotProductAttention> findF4(Api::Network& net) {
    for (uint32_t i=0;i<net.getNumLayers();++i)
        if (auto p=std::dynamic_pointer_cast<Api::ScaledDotProductAttention>(net.getLayer(i)))
            return p;
    return {};
}

std::filesystem::path pathF4() {
    return std::filesystem::temp_directory_path()/
        ("thor_sdpa_f4_"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
}

Impl::DynamicExpression mseF4(bool gradient) {
    auto p=Impl::Expression::input("predictions",DataType::FP16,DataType::FP32);
    auto y=Impl::Expression::input("labels",DataType::FP16,DataType::FP32);
    auto diff=p-y;
    auto result=gradient
        ? diff*Impl::Expression(2.*Impl::Loss::getLossScalingFactor())
        : diff*diff;
    return Impl::DynamicExpression::fromExpressionDefinition(
        Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({{
            gradient?"predictions_grad":"loss",result.withOutputDType(DataType::FP16)}})));
}

void compareGradF4(const std::vector<float>& gpu, const std::vector<double>& expected,
                   uint64_t activeElements, const char* component) {
    ASSERT_GE(gpu.size(),activeElements);
    const float scale=Impl::Loss::getLossScalingFactor();
    bool nonzero=false;
    for (uint64_t i=0;i<activeElements;++i) {
        SCOPED_TRACE(::testing::Message()<<component<<" index="<<i);
        ASSERT_TRUE(std::isfinite(gpu[i]));
        const double got=gpu[i]/scale;
        const double reference=expected[i];
        const double tolerance=.009+std::abs(reference)*.20;
        EXPECT_NEAR(got,reference,tolerance);
        nonzero|=std::abs(reference)>1e-5;
    }
    EXPECT_TRUE(nonzero)<<component<<" reference must exercise nonzero gradients";
}
} // namespace

TEST(SdpaNeutralBackward2CF4, SavedSharedKvB2B7AccumulatesUpstreamKeyAndValueGradients) {
    int devices=0;
    if (cudaGetDeviceCount(&devices)!=cudaSuccess || devices==0)
        GTEST_SKIP()<<"CUDA device required for numerical SDPA backward";
    const auto path=pathF4();
    std::filesystem::remove_all(path);
    {
        Api::Network net("sdpa_neutral_shared_kv_2cf4");
        const auto q=Api::RaggedNetworkInput::Builder().network(net).name("q")
            .valuesDataType(DataType::FP16).offsetsDataType(DataType::UINT32)
            .trailingDimensions({HQ,D}).maxValuesPerRow(MQ).build();
        const auto kv=Api::RaggedNetworkInput::Builder().network(net).name("kv")
            .valuesDataType(DataType::FP16).offsetsDataType(DataType::UINT64)
            .trailingDimensions({HK,D}).maxValuesPerRow(MK).build();
        const auto label=Api::RaggedNetworkInput::Builder().network(net).name("labels")
            .valuesDataType(DataType::FP16).trailingDimensions({HQ,D}).partition(q).build();
        const auto qr=Api::GradientRivet::Builder().network(net).tensor(q.getValues()).build();
        const auto kvRivet=Api::GradientRivet::Builder().network(net).tensor(kv.getValues()).build();
        const auto sharedKv=kv.withValues(kvRivet.getFeatureOutput().value());
        auto sdpa=Api::ScaledDotProductAttention::Builder().network(net)
            .queryInput(q.withValues(qr.getFeatureOutput().value()))
            .keyInput(sharedKv)
            .valueInput(sharedKv)
            .computeDataType(DataType::FP32).build();
        ASSERT_TRUE(sdpa.getRaggedFeatureOutput().has_value());
        ASSERT_TRUE(sdpa.getRaggedFeatureOutput()->sharesPartitionWith(q));
        (void)Api::RaggedCustomLoss::Builder().network(net)
            .predictions(*sdpa.getRaggedFeatureOutput()).labels(label)
            .lossExpression(mseF4(false)).gradientExpression(mseF4(true)).build();
        // Expose RAW to ensure it remains a semantic training root.
        // Locate loss explicitly rather than assuming the builder owns a
        // trainable parameter or can turn a reporting consumer into the root.
        for (uint32_t i=0;i<net.getNumLayers();++i)
            if (auto loss=std::dynamic_pointer_cast<Api::RaggedCustomLoss>(net.getLayer(i))) {
                (void)Api::NetworkOutput::Builder().network(net).name("raw_loss")
                    .inputTensor(loss->getRaggedRawLoss().getValues()).dataType(DataType::FP16).build();
                break;
            }
        net.save(path.string(),true);
    }
    for (uint32_t b:{2u,7u}) {
        SCOPED_TRACE(::testing::Message()<<"B="<<b);
        Api::Network net("sdpa_neutral_shared_kv_2cf4");
        ASSERT_NO_THROW(net.load(path.string()));
        const auto sdpa=findF4(net);
        ASSERT_NE(sdpa,nullptr);
        ASSERT_EQ(sdpa->architectureJson().at("version"),"2.2.0");
        const auto original=net.architectureJson();
        const auto inputs=net.getExternalRaggedNetworkInputs();
        auto findInput=[&](const char *name)->Api::RaggedTensor {
            for (const auto& r:inputs) if (r.name==name) return r.raggedTensor;
            throw std::runtime_error(std::string("missing input: ")+name);
        };
        const auto q=findInput("q"),kv=findInput("kv"),labels=findInput("labels");
        ASSERT_FALSE(q.sharesPartitionWith(kv));
        std::vector<Event> events;
        auto placed=net.place(b,events,false);
        ASSERT_NE(placed,nullptr);
        for (auto& e:events) e.synchronize();
        auto physical=std::dynamic_pointer_cast<Impl::CustomLayer>(
            placed->getStampedNetwork(0).getPhysicalLayerFromApiLayer(sdpa->getId()));
        ASSERT_NE(physical,nullptr);
        ASSERT_TRUE(physical->getFeatureOutput().has_value());
        EXPECT_EQ(physical->getFeatureOutput()->getDimensions(),
                  (std::vector<uint64_t>{b*MQ,HQ,D}));
        // The test-only identity immediately before K/V fanout records the
        // already-accumulated upstream gradient after real RAW-loss training.
        std::shared_ptr<Impl::GradientRivet> sharedRivet;
        for (uint32_t i=0;i<net.getNumLayers();++i) {
            auto apiRivet=std::dynamic_pointer_cast<Api::GradientRivet>(net.getLayer(i));
            if (!apiRivet || !apiRivet->getFeatureInput().has_value() ||
                apiRivet->getFeatureInput().value() != kv.getValues()) continue;
            sharedRivet=std::dynamic_pointer_cast<Impl::GradientRivet>(
                placed->getStampedNetwork(0).getPhysicalLayerFromApiLayer(apiRivet->getId()));
            break;
        }
        ASSERT_NE(sharedRivet,nullptr);
        std::map<std::string,Impl::Tensor> outputs;
        std::map<std::string,Event> ready;
        auto done=placed->submitBatch(0,makeBatchF4(b,q,kv,labels),outputs,ready,false);
        done.synchronize();
        placed->synchronize();
        ASSERT_TRUE(outputs.contains("raw_loss"));
        const auto& error=physical->getErrorOutputs();
        // SDPA's declared input port order starts with query, key, value.
        ASSERT_GE(error.size(),3u);
        for (unsigned i=0;i<3;++i) ASSERT_TRUE(error[i].has_value())<<"port="<<i;
        const auto dq=readHalfF4(*error[0]);
        const auto dk=readHalfF4(*error[1]);
        const auto dv=readHalfF4(*error[2]);
        ASSERT_TRUE(sharedRivet->getErrorInput().has_value())
            << "Shared K/V rivet must receive the fanout-reduced gradient";
        const auto upstream=readHalfF4(*sharedRivet->getErrorInput());
        const auto ref=referenceF4(b);
        compareGradF4(dq,ref.dq,offsetsF4(b,true).back()*HQ*D,"dQ");
        compareGradF4(dk,ref.dk,offsetsF4(b,false).back()*HK*D,"dK");
        compareGradF4(dv,ref.dv,offsetsF4(b,false).back()*HK*D,"dV");
        std::vector<double> expectedUpstream(ref.dk.size());
        for (size_t i=0;i<expectedUpstream.size();++i)
            expectedUpstream[i]=ref.dk[i]+ref.dv[i];
        const uint64_t active=offsetsF4(b,false).back()*HK*D;
        compareGradF4(upstream,expectedUpstream,active,"dK+dV upstream");
        ASSERT_GE(upstream.size(),active);
        ASSERT_GE(dk.size(),active);
        ASSERT_GE(dv.size(),active);
        const double lossScale=Impl::Loss::getLossScalingFactor();
        bool distinctFromSingleBranch=false;
        for (uint64_t i=0;i<active;++i) {
            SCOPED_TRACE(::testing::Message()<<"K/V shared upstream element="<<i);
            const double fromPorts=(static_cast<double>(dk[i])+dv[i])/lossScale;
            EXPECT_NEAR(static_cast<double>(upstream[i])/lossScale,fromPorts,
                        .015+.05*std::abs(fromPorts));
            if (std::abs(expectedUpstream[i]-ref.dk[i]) > 1.e-4 &&
                std::abs(expectedUpstream[i]-ref.dv[i]) > 1.e-4)
                distinctFromSingleBranch=true;
        }
        EXPECT_TRUE(distinctFromSingleBranch)
            << "The summed gradient must exercise both independently nonzero K/V branches";
        EXPECT_EQ(net.architectureJson(),original);
    }
    std::filesystem::remove_all(path);
}
