// 2C-F1: independently placed neutral SDPA, real RAW-loss backward, and
// numerical Q/K/V gradients.  This intentionally leaves masking, dropout,
// mixed-domain backward and shared-use accumulation for later small gates.
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
#include <string>
#include <variant>
#include <vector>

namespace {
namespace Api = Thor;
namespace Impl = ThorImplementation;
using DataType = Impl::DataType;
constexpr uint64_t MQ = 4, MK = 5, HQ = 4, HK = 2, D = 32;
const double SCALE = 1.0 / std::sqrt(static_cast<double>(D));

std::vector<uint64_t> offsetsF1(uint32_t b, bool query) {
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

float sourceF1(int kind, uint64_t token, uint64_t head, uint64_t dim) {
    const float x = .07f*(kind+1) + .013f*static_cast<float>(
        (token*11+head*7+dim*5+kind*3)%31) - .16f
        + .009f*static_cast<float>(dim%7);
    return __half2float(__float2half(kind == 2 ? x + .13f*static_cast<float>(token%5)-.2f : x));
}

Impl::Tensor hostValuesF1(uint32_t b, uint64_t m, uint64_t h,
                          const std::vector<uint64_t>& off, int kind) {
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor value(cpu, Impl::TensorDescriptor(DataType::FP16,{b*m,h,D}));
    auto *p = value.getMemPtr<half>();
    for (uint64_t t=0; t<b*m; ++t)
        for (uint64_t head=0; head<h; ++head)
            for (uint64_t d=0; d<D; ++d)
                p[(t*h+head)*D+d] = t<off.back()
                    ? __float2half(sourceF1(kind,t,head,d))
                    : __float2half(std::numeric_limits<float>::quiet_NaN());
    return value;
}

Impl::Tensor hostOffsetsF1(const std::vector<uint64_t>& off, DataType dtype) {
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor result(cpu,Impl::TensorDescriptor(dtype,{off.size()}));
    for (size_t i=0;i<off.size();++i) {
        if (dtype==DataType::UINT32) result.getMemPtr<uint32_t>()[i]=static_cast<uint32_t>(off[i]);
        else result.getMemPtr<uint64_t>()[i]=off[i];
    }
    return result;
}

Batch makeBatchF1(uint32_t b, const Api::RaggedTensor& q,
                  const Api::RaggedTensor& k, const Api::RaggedTensor& v,
                  const Api::RaggedTensor& labels) {
    auto qo=offsetsF1(b,true), ko=offsetsF1(b,false);
    auto qp=Impl::RowPartitionRuntime(hostOffsetsF1(qo,DataType::UINT32),
                                       q.getDescriptorForPlacement(b).getRowPartition());
    auto kp=Impl::RowPartitionRuntime(hostOffsetsF1(ko,DataType::UINT64),
                                       k.getDescriptorForPlacement(b).getRowPartition());
    qp.setHostOffsets(qo);
    kp.setHostOffsets(ko);
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor zeros(cpu,labels.getDescriptorForPlacement(b).getValuesDescriptor());
    for (uint64_t i=0;i<b*MQ*HQ*D;++i)
        zeros.getMemPtr<half>()[i]=i<qo.back()*HQ*D
            ? __float2half(0.f) : __float2half(std::numeric_limits<float>::quiet_NaN());
    Batch batch;
    batch.insert("q", Impl::RaggedTensor(hostValuesF1(b,MQ,HQ,qo,0),qp));
    batch.insert("k", Impl::RaggedTensor(hostValuesF1(b,MK,HK,ko,1),kp));
    batch.insert("v", Impl::RaggedTensor(hostValuesF1(b,MK,HK,ko,2),kp));
    batch.insert("labels", Impl::RaggedTensor(zeros,qp));
    batch.setValidExampleCount(b);
    return batch;
}

std::vector<float> readHalfF1(const Impl::Tensor& src) {
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

struct ReferenceF1 {
    std::vector<double> dq,dk,dv;
};

ReferenceF1 referenceF1(uint32_t b) {
    const auto qo=offsetsF1(b,true), ko=offsetsF1(b,false);
    ReferenceF1 r;
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
                        dot+=sourceF1(0,qi,h,d)*sourceF1(1,ki,kh,d);
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
                        y+=scores[j]*sourceF1(2,ko[row]+j,kh,d);
                    // The loss consumes actual FP16 attention output.
                    dy[d]=2.*__half2float(__float2half(static_cast<float>(y)));
                }
                double mean=0;
                for (size_t j=0;j<scores.size();++j) {
                    const uint64_t ki=ko[row]+j;
                    double contribution=0;
                    for (uint64_t d=0;d<D;++d) {
                        contribution+=dy[d]*sourceF1(2,ki,kh,d);
                        r.dv[(ki*HK+kh)*D+d]+=scores[j]*dy[d];
                    }
                    dp[j]=contribution;
                    mean+=scores[j]*contribution;
                }
                for (size_t j=0;j<scores.size();++j) {
                    const uint64_t ki=ko[row]+j;
                    const double ds=scores[j]*(dp[j]-mean)*SCALE;
                    for (uint64_t d=0;d<D;++d) {
                        r.dq[(qi*HQ+h)*D+d]+=ds*sourceF1(1,ki,kh,d);
                        r.dk[(ki*HK+kh)*D+d]+=ds*sourceF1(0,qi,h,d);
                    }
                }
            }
    return r;
}

std::shared_ptr<Api::ScaledDotProductAttention> findF1(Api::Network& net) {
    for (uint32_t i=0;i<net.getNumLayers();++i)
        if (auto p=std::dynamic_pointer_cast<Api::ScaledDotProductAttention>(net.getLayer(i)))
            return p;
    return {};
}

std::filesystem::path pathF1() {
    return std::filesystem::temp_directory_path()/
        ("thor_sdpa_f1_"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
}

Impl::DynamicExpression mseF1(bool gradient) {
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

void compareGradF1(const std::vector<float>& gpu, const std::vector<double>& expected,
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

TEST(SdpaNeutralBackward2CF1, SavedSeparatePartitionsB2B7NumericalDQDKDV) {
    int devices=0;
    if (cudaGetDeviceCount(&devices)!=cudaSuccess || devices==0)
        GTEST_SKIP()<<"CUDA device required for numerical SDPA backward";
    const auto path=pathF1();
    std::filesystem::remove_all(path);
    {
        Api::Network net("sdpa_neutral_2cf1");
        const auto q=Api::RaggedNetworkInput::Builder().network(net).name("q")
            .valuesDataType(DataType::FP16).offsetsDataType(DataType::UINT32)
            .trailingDimensions({HQ,D}).maxValuesPerRow(MQ).build();
        const auto k=Api::RaggedNetworkInput::Builder().network(net).name("k")
            .valuesDataType(DataType::FP16).offsetsDataType(DataType::UINT64)
            .trailingDimensions({HK,D}).maxValuesPerRow(MK).build();
        const auto v=Api::RaggedNetworkInput::Builder().network(net).name("v")
            .valuesDataType(DataType::FP16).trailingDimensions({HK,D}).partition(k).build();
        const auto label=Api::RaggedNetworkInput::Builder().network(net).name("labels")
            .valuesDataType(DataType::FP16).trailingDimensions({HQ,D}).partition(q).build();
        const auto qr=Api::GradientRivet::Builder().network(net).tensor(q.getValues()).build();
        const auto kr=Api::GradientRivet::Builder().network(net).tensor(k.getValues()).build();
        const auto vr=Api::GradientRivet::Builder().network(net).tensor(v.getValues()).build();
        auto sdpa=Api::ScaledDotProductAttention::Builder().network(net)
            .queryInput(q.withValues(qr.getFeatureOutput().value()))
            .keyInput(k.withValues(kr.getFeatureOutput().value()))
            .valueInput(v.withValues(vr.getFeatureOutput().value()))
            .computeDataType(DataType::FP32).build();
        ASSERT_TRUE(sdpa.getRaggedFeatureOutput().has_value());
        ASSERT_TRUE(sdpa.getRaggedFeatureOutput()->sharesPartitionWith(q));
        (void)Api::RaggedCustomLoss::Builder().network(net)
            .predictions(*sdpa.getRaggedFeatureOutput()).labels(label)
            .lossExpression(mseF1(false)).gradientExpression(mseF1(true)).build();
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
        Api::Network net("sdpa_neutral_2cf1");
        ASSERT_NO_THROW(net.load(path.string()));
        const auto sdpa=findF1(net);
        ASSERT_NE(sdpa,nullptr);
        ASSERT_EQ(sdpa->architectureJson().at("version"),"2.2.0");
        const auto original=net.architectureJson();
        const auto inputs=net.getExternalRaggedNetworkInputs();
        auto findInput=[&](const char *name)->Api::RaggedTensor {
            for (const auto& r:inputs) if (r.name==name) return r.raggedTensor;
            throw std::runtime_error(std::string("missing input: ")+name);
        };
        const auto q=findInput("q"),k=findInput("k"),v=findInput("v"),labels=findInput("labels");
        ASSERT_FALSE(q.sharesPartitionWith(k));
        ASSERT_TRUE(k.sharesPartitionWith(v));
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
        std::map<std::string,Impl::Tensor> outputs;
        std::map<std::string,Event> ready;
        auto done=placed->submitBatch(0,makeBatchF1(b,q,k,v,labels),outputs,ready,false);
        done.synchronize();
        placed->synchronize();
        ASSERT_TRUE(outputs.contains("raw_loss"));
        const auto& error=physical->getErrorOutputs();
        // SDPA's declared input port order starts with query, key, value.
        ASSERT_GE(error.size(),3u);
        for (unsigned i=0;i<3;++i) ASSERT_TRUE(error[i].has_value())<<"port="<<i;
        const auto dq=readHalfF1(*error[0]);
        const auto dk=readHalfF1(*error[1]);
        const auto dv=readHalfF1(*error[2]);
        const auto ref=referenceF1(b);
        compareGradF1(dq,ref.dq,offsetsF1(b,true).back()*HQ*D,"dQ");
        compareGradF1(dk,ref.dk,offsetsF1(b,false).back()*HK*D,"dK");
        compareGradF1(dv,ref.dv,offsetsF1(b,false).back()*HK*D,"dV");
        EXPECT_EQ(net.architectureJson(),original);
    }
    std::filesystem::remove_all(path);
}
