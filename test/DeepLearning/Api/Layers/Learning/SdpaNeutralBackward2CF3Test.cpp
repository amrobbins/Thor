// 2C-F3: saved/re-placed dense-query, ragged-KV numerical backward.
// Keep shared K/V input fanout as a separate gate (2C-F4) so that this
// test isolates the mixed-domain SDPA and dense RAW-loss backward paths.
// No changes to SDPA physical kernels or placement contracts.
#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Layers/Learning/ScaledDotProductAttention.h"
#include "DeepLearning/Api/Layers/Loss/CustomLoss.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
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
constexpr uint64_t MQ3 = 4, MK3 = 5, HQ3 = 4, HK3 = 2, D3 = 32;
const double SCALE3 = 1.0 / std::sqrt(static_cast<double>(D3));

std::vector<uint64_t> kvOffsetsF3(uint32_t b) {
    std::vector<uint64_t> off(b+1,0);
    for (uint32_t row=0; row<b; ++row)
        off[row+1] = off[row] + 2u + (row*3u)%4u;
    return off;
}

float sourceF3(int kind, uint64_t token, uint64_t head, uint64_t dim) {
    const float value = .07f*(kind+1) + .013f*static_cast<float>(
        (token*11+head*7+dim*5+kind*3)%31) - .16f
        + .009f*static_cast<float>(dim%7);
    return __half2float(__float2half(kind==2 ?
        value+.13f*static_cast<float>(token%5)-.2f : value));
}

Impl::Tensor hostDenseF3(uint32_t b, int kind) {
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor data(cpu,Impl::TensorDescriptor(DataType::FP16,{b,MQ3,HQ3,D3}));
    auto* dst=data.getMemPtr<half>();
    for (uint64_t t=0; t<static_cast<uint64_t>(b)*MQ3; ++t)
        for (uint64_t h=0; h<HQ3; ++h)
            for (uint64_t d=0; d<D3; ++d)
                dst[(t*HQ3+h)*D3+d] = __float2half(sourceF3(kind,t,h,d));
    return data;
}

Impl::Tensor hostLabelsF3(uint32_t b) {
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor data(cpu,Impl::TensorDescriptor(DataType::FP16,{b,MQ3,HQ3,D3}));
    auto* dst=data.getMemPtr<half>();
    for (uint64_t i=0; i<static_cast<uint64_t>(b)*MQ3*HQ3*D3; ++i)
        dst[i] = __float2half(0.f);
    return data;
}

Impl::Tensor hostOffsetsF3(const std::vector<uint64_t>& off) {
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor data(cpu,Impl::TensorDescriptor(DataType::UINT64,{off.size()}));
    auto* dst=data.getMemPtr<uint64_t>();
    for (size_t i=0; i<off.size(); ++i) dst[i]=off[i];
    return data;
}

Impl::Tensor hostRaggedValuesF3(uint32_t b,const std::vector<uint64_t>& off,int kind) {
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor data(cpu,Impl::TensorDescriptor(DataType::FP16,{b*MK3,HK3,D3}));
    auto* dst=data.getMemPtr<half>();
    for (uint64_t t=0; t<static_cast<uint64_t>(b)*MK3; ++t)
        for (uint64_t h=0; h<HK3; ++h)
            for (uint64_t d=0; d<D3; ++d)
                dst[(t*HK3+h)*D3+d] = t<off.back()
                    ? __float2half(sourceF3(kind,t,h,d))
                    : __float2half(std::numeric_limits<float>::quiet_NaN());
    return data;
}

Batch makeBatchF3(uint32_t b,const Api::RaggedTensor& k) {
    const auto off=kvOffsetsF3(b);
    Impl::RowPartitionRuntime partition(hostOffsetsF3(off),
        k.getDescriptorForPlacement(b).getRowPartition());
    partition.setHostOffsets(off);
    Batch batch;
    batch.insert("q",hostDenseF3(b,0));
    batch.insert("labels",hostLabelsF3(b));
    batch.insert("k",Impl::RaggedTensor(hostRaggedValuesF3(b,off,1),partition));
    batch.insert("v",Impl::RaggedTensor(hostRaggedValuesF3(b,off,2),partition));
    batch.setValidExampleCount(b);
    return batch;
}

std::vector<float> readHalfF3(const Impl::Tensor& src) {
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor dst=src.clone(cpu), from=src;
    Stream stream=Stream::getNextDownloadStream(src.getPlacement().getDeviceNum());
    dst.copyFromAsync(from,stream);
    stream.synchronize();
    const half* p=dst.getMemPtr<half>();
    std::vector<float> out(dst.getTotalNumElements());
    for (size_t i=0;i<out.size();++i) out[i]=__half2float(p[i]);
    return out;
}

struct GradientsF3 {
    std::vector<double> dq,dk,dv;
};

GradientsF3 cpuReferenceF3(uint32_t b) {
    const auto off=kvOffsetsF3(b);
    GradientsF3 ref;
    ref.dq.resize(b*MQ3*HQ3*D3,0.);
    ref.dk.resize(b*MK3*HK3*D3,0.);
    ref.dv.resize(b*MK3*HK3*D3,0.);
    for (uint32_t row=0;row<b;++row)
        for (uint64_t localQ=0;localQ<MQ3;++localQ) {
            const uint64_t qi=row*MQ3+localQ;
            for (uint64_t head=0;head<HQ3;++head) {
                const uint64_t kh=head/(HQ3/HK3);
                std::vector<double> scores;
                for (uint64_t ki=off[row];ki<off[row+1];++ki) {
                    double dot=0;
                    for (uint64_t d=0;d<D3;++d)
                        dot+=sourceF3(0,qi,head,d)*sourceF3(1,ki,kh,d);
                    scores.push_back(dot*SCALE3);
                }
                if (scores.empty()) continue;
                const double maximum=*std::max_element(scores.begin(),scores.end());
                double sum=0;
                for (double& s:scores) {s=std::exp(s-maximum);sum+=s;}
                for (double& s:scores) s/=sum;
                std::vector<double> dy(D3),dp(scores.size());
                for (uint64_t d=0;d<D3;++d) {
                    double output=0;
                    for (size_t j=0;j<scores.size();++j)
                        output+=scores[j]*sourceF3(2,off[row]+j,kh,d);
                    // RAW squared-error loss consumes the rounded FP16 SDPA output.
                    dy[d]=2.*__half2float(__float2half(static_cast<float>(output)));
                }
                double weightedDp=0;
                for (size_t j=0;j<scores.size();++j) {
                    const uint64_t ki=off[row]+j;
                    double dpContribution=0;
                    for (uint64_t d=0;d<D3;++d) {
                        dpContribution+=dy[d]*sourceF3(2,ki,kh,d);
                        ref.dv[(ki*HK3+kh)*D3+d]+=scores[j]*dy[d];
                    }
                    dp[j]=dpContribution;
                    weightedDp+=scores[j]*dpContribution;
                }
                for (size_t j=0;j<scores.size();++j) {
                    const uint64_t ki=off[row]+j;
                    const double ds=scores[j]*(dp[j]-weightedDp)*SCALE3;
                    for (uint64_t d=0;d<D3;++d) {
                        ref.dq[(qi*HQ3+head)*D3+d]+=ds*sourceF3(1,ki,kh,d);
                        ref.dk[(ki*HK3+kh)*D3+d]+=ds*sourceF3(0,qi,head,d);
                    }
                }
            }
        }
    return ref;
}

void compareGradientF3(const std::vector<float>& actual,
                       const std::vector<double>& expected,
                       uint64_t activeElements,const char* port) {
    ASSERT_GE(actual.size(),activeElements);
    const float lossScale=Impl::Loss::getLossScalingFactor();
    bool nonzero=false;
    for (uint64_t i=0;i<activeElements;++i) {
        SCOPED_TRACE(::testing::Message()<<port<<" element="<<i);
        ASSERT_TRUE(std::isfinite(actual[i]));
        const double value=actual[i]/lossScale;
        const double reference=expected[i];
        EXPECT_NEAR(value,reference,.009+std::abs(reference)*.20);
        nonzero|=std::abs(reference)>1.e-5;
    }
    EXPECT_TRUE(nonzero)<<port<<" reference must exercise nonzero gradients";
}

Impl::DynamicExpression squaredLossF3(bool gradient) {
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

std::shared_ptr<Api::ScaledDotProductAttention> findSdpaF3(Api::Network& net) {
    for (uint32_t i=0;i<net.getNumLayers();++i)
        if (auto sdpa=std::dynamic_pointer_cast<Api::ScaledDotProductAttention>(net.getLayer(i)))
            return sdpa;
    return {};
}

std::filesystem::path archiveF3() {
    return std::filesystem::temp_directory_path()/
        ("sdpa_neutral_f3_"+
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
}
} // namespace

TEST(SdpaNeutralBackward2CF3, SavedDenseQueryRaggedKvB2B7NumericalDQDKDV) {
    int devices=0;
    if (cudaGetDeviceCount(&devices)!=cudaSuccess || devices==0)
        GTEST_SKIP()<<"CUDA device required for numerical SDPA backward";
    const auto path=archiveF3();
    std::filesystem::remove_all(path);
    {
        Api::Network net("sdpa_neutral_2cf3");
        const auto q=Api::NetworkInput::Builder().network(net).name("q")
            .dataType(DataType::FP16).dimensions({MQ3,HQ3,D3}).build();
        const auto labels=Api::NetworkInput::Builder().network(net).name("labels")
            .dataType(DataType::FP16).dimensions({MQ3,HQ3,D3}).build();
        const auto k=Api::RaggedNetworkInput::Builder().network(net).name("k")
            .valuesDataType(DataType::FP16).offsetsDataType(DataType::UINT64)
            .trailingDimensions({HK3,D3}).maxValuesPerRow(MK3).build();
        const auto v=Api::RaggedNetworkInput::Builder().network(net).name("v")
            .valuesDataType(DataType::FP16).trailingDimensions({HK3,D3}).partition(k).build();
        ASSERT_TRUE(q.getFeatureOutput().has_value());
        ASSERT_TRUE(labels.getFeatureOutput().has_value());
        const auto qr=Api::GradientRivet::Builder().network(net)
            .tensor(*q.getFeatureOutput()).build();
        const auto kr=Api::GradientRivet::Builder().network(net)
            .tensor(k.getValues()).build();
        const auto vr=Api::GradientRivet::Builder().network(net)
            .tensor(v.getValues()).build();
        const auto sdpa=Api::ScaledDotProductAttention::Builder().network(net)
            .queryInput(qr.getFeatureOutput().value())
            .keyInput(k.withValues(kr.getFeatureOutput().value()))
            .valueInput(v.withValues(vr.getFeatureOutput().value()))
            .computeDataType(DataType::FP32).build();
        ASSERT_FALSE(sdpa.getRaggedFeatureOutput().has_value());
        const auto loss=Api::CustomLoss::Builder().network(net)
            .predictions(sdpa.getOutput("output")).labels(*labels.getFeatureOutput())
            .lossExpression(squaredLossF3(false))
            .gradientExpression(squaredLossF3(true))
            .reportsRawLoss().build();
        (void)Api::NetworkOutput::Builder().network(net).name("raw_loss")
            .inputTensor(loss.getRawLoss()).dataType(DataType::FP16).build();
        net.save(path.string(),true);
    }
    for (uint32_t b:{2u,7u}) {
        SCOPED_TRACE(::testing::Message()<<"B="<<b);
        Api::Network net("sdpa_neutral_2cf3");
        ASSERT_NO_THROW(net.load(path.string()));
        const auto sdpa=findSdpaF3(net);
        ASSERT_NE(sdpa,nullptr);
        ASSERT_EQ(sdpa->architectureJson().at("version"),"2.2.0");
        EXPECT_FALSE(sdpa->getQueryIsRagged());
        EXPECT_TRUE(sdpa->getKeyValueIsRagged());
        const auto before=net.architectureJson();
        const auto inputs=net.getExternalRaggedNetworkInputs();
        auto it=std::find_if(inputs.begin(),inputs.end(),[](const auto& input){return input.name=="k";});
        ASSERT_NE(it,inputs.end());
        const auto k=it->raggedTensor;
        const auto itV=std::find_if(inputs.begin(),inputs.end(),[](const auto& input){return input.name=="v";});
        ASSERT_NE(itV,inputs.end());
        EXPECT_TRUE(k.sharesPartitionWith(itV->raggedTensor));
        std::vector<Event> events;
        auto placed=net.place(b,events,false);
        ASSERT_NE(placed,nullptr);
        for (auto& event:events) event.synchronize();
        auto& stamp=placed->getStampedNetwork(0);
        auto physical=std::dynamic_pointer_cast<Impl::CustomLayer>(
            stamp.getPhysicalLayerFromApiLayer(sdpa->getId()));
        ASSERT_NE(physical,nullptr);
        ASSERT_TRUE(physical->getFeatureOutput().has_value());
        EXPECT_EQ(physical->getFeatureOutput()->getDimensions(),
                  (std::vector<uint64_t>{b,MQ3,HQ3,D3}));
        std::map<std::string,Impl::Tensor> outputs;
        std::map<std::string,Event> ready;
        auto done=placed->submitBatch(0,makeBatchF3(b,k),outputs,ready,false);
        done.synchronize();
        placed->synchronize();
        ASSERT_TRUE(outputs.contains("raw_loss"));
        const auto& gradients=physical->getErrorOutputs();
        ASSERT_GE(gradients.size(),3u);
        for (unsigned i=0;i<3;++i) ASSERT_TRUE(gradients[i].has_value())<<"port="<<i;
        const auto dq=readHalfF3(*gradients[0]);
        const auto dk=readHalfF3(*gradients[1]);
        const auto dv=readHalfF3(*gradients[2]);
        const auto expected=cpuReferenceF3(b);
        compareGradientF3(dq,expected.dq,b*MQ3*HQ3*D3,"dQ");
        compareGradientF3(dk,expected.dk,kvOffsetsF3(b).back()*HK3*D3,"dK");
        compareGradientF3(dv,expected.dv,kvOffsetsF3(b).back()*HK3*D3,"dV");
        EXPECT_EQ(net.architectureJson(),before);
    }
    std::filesystem::remove_all(path);
}
