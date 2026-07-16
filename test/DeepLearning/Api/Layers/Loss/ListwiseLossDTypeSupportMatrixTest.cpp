#include "DeepLearning/Api/Layers/Loss/ListNetLoss.h"
#include "DeepLearning/Api/Layers/Loss/ListwiseSoftmaxCrossEntropyLoss.h"

#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "gtest/gtest.h"

#include <exception>
#include <optional>
#include <string>
#include <vector>

using namespace std;

namespace Api = Thor;

namespace {

string dtypeName(Api::DataType dtype) { return ThorImplementation::TensorDescriptor::getElementTypeName(dtype); }

Api::Tensor valueTensor(Api::DataType dtype) { return Api::Tensor(dtype, {4}); }

optional<Api::Tensor> optionalMaskTensor(optional<Api::DataType> dtype) {
    if (!dtype.has_value())
        return nullopt;
    return Api::Tensor(dtype.value(), {4});
}

void buildListwiseSoftmaxCrossEntropy(Api::DataType predictionsDType,
                                      Api::DataType labelsDType,
                                      Api::DataType lossDType,
                                      optional<Api::DataType> maskDType = nullopt) {
    Api::Network network("listwise_softmax_cross_entropy_dtype_support");
    Api::ListwiseSoftmaxCrossEntropyLoss::Builder builder;
    builder.network(network)
        .predictions(valueTensor(predictionsDType))
        .labels(valueTensor(labelsDType))
        .temperature(0.75f)
        .lossDataType(lossDType)
        .reportsRawLoss();
    optional<Api::Tensor> mask = optionalMaskTensor(maskDType);
    if (mask.has_value())
        builder.mask(mask.value());
    Api::ListwiseSoftmaxCrossEntropyLoss loss = builder.build();
    EXPECT_EQ(loss.getLoss().getDataType(), lossDType);
}

void buildListNet(Api::DataType predictionsDType,
                  Api::DataType labelsDType,
                  Api::DataType lossDType,
                  optional<Api::DataType> maskDType = nullopt) {
    Api::Network network("list_net_dtype_support");
    Api::ListNetLoss::Builder builder;
    builder.network(network)
        .predictions(valueTensor(predictionsDType))
        .labels(valueTensor(labelsDType))
        .scoreTemperature(0.8f)
        .labelTemperature(0.6f)
        .lossDataType(lossDType)
        .reportsRawLoss();
    optional<Api::Tensor> mask = optionalMaskTensor(maskDType);
    if (mask.has_value())
        builder.mask(mask.value());
    Api::ListNetLoss loss = builder.build();
    EXPECT_EQ(loss.getLoss().getDataType(), lossDType);
}

void expectBothListwiseLossesAccept(Api::DataType predictionsDType,
                                    Api::DataType labelsDType,
                                    Api::DataType lossDType,
                                    optional<Api::DataType> maskDType = nullopt) {
    SCOPED_TRACE("predictions=" + dtypeName(predictionsDType) + ", labels=" + dtypeName(labelsDType) +
                 ", loss=" + dtypeName(lossDType) +
                 ", mask=" + (maskDType.has_value() ? dtypeName(maskDType.value()) : string("none")));
    EXPECT_NO_THROW(buildListwiseSoftmaxCrossEntropy(predictionsDType, labelsDType, lossDType, maskDType));
    EXPECT_NO_THROW(buildListNet(predictionsDType, labelsDType, lossDType, maskDType));
}

void expectBothListwiseLossesReject(Api::DataType predictionsDType,
                                    Api::DataType labelsDType,
                                    Api::DataType lossDType,
                                    optional<Api::DataType> maskDType = nullopt) {
    SCOPED_TRACE("predictions=" + dtypeName(predictionsDType) + ", labels=" + dtypeName(labelsDType) +
                 ", loss=" + dtypeName(lossDType) +
                 ", mask=" + (maskDType.has_value() ? dtypeName(maskDType.value()) : string("none")));
    EXPECT_THROW(buildListwiseSoftmaxCrossEntropy(predictionsDType, labelsDType, lossDType, maskDType), std::exception);
    EXPECT_THROW(buildListNet(predictionsDType, labelsDType, lossDType, maskDType), std::exception);
}

}  // namespace

TEST(ListwiseLossDTypeSupportMatrix, AcceptsFp16AndFp32ValueAndLossTensors) {
    const vector<Api::DataType> supportedValueDTypes = {Api::DataType::FP16, Api::DataType::FP32};
    for (Api::DataType predictionsDType : supportedValueDTypes) {
        for (Api::DataType labelsDType : supportedValueDTypes) {
            for (Api::DataType lossDType : supportedValueDTypes) {
                expectBothListwiseLossesAccept(predictionsDType, labelsDType, lossDType);
            }
        }
    }
}

TEST(ListwiseLossDTypeSupportMatrix, AcceptsBoolUint8Fp16AndFp32Masks) {
    const vector<Api::DataType> supportedMaskDTypes = {
        Api::DataType::BOOLEAN, Api::DataType::UINT8, Api::DataType::FP16, Api::DataType::FP32};
    for (Api::DataType maskDType : supportedMaskDTypes)
        expectBothListwiseLossesAccept(Api::DataType::FP32, Api::DataType::FP32, Api::DataType::FP32, maskDType);
}

TEST(ListwiseLossDTypeSupportMatrix, RejectsUnsupportedPredictionLabelAndLossDTypes) {
    const vector<Api::DataType> unsupportedValueDTypes = {
        Api::DataType::BOOLEAN, Api::DataType::UINT8, Api::DataType::UINT16, Api::DataType::INT32, Api::DataType::FP64,
        Api::DataType::BF16};

    for (Api::DataType unsupportedDType : unsupportedValueDTypes) {
        expectBothListwiseLossesReject(unsupportedDType, Api::DataType::FP32, Api::DataType::FP32);
        expectBothListwiseLossesReject(Api::DataType::FP32, unsupportedDType, Api::DataType::FP32);
        expectBothListwiseLossesReject(Api::DataType::FP32, Api::DataType::FP32, unsupportedDType);
    }
}

TEST(ListwiseLossDTypeSupportMatrix, RejectsUnsupportedMaskDTypes) {
    const vector<Api::DataType> unsupportedMaskDTypes = {
        Api::DataType::UINT16, Api::DataType::INT32, Api::DataType::FP64, Api::DataType::BF16};
    for (Api::DataType maskDType : unsupportedMaskDTypes)
        expectBothListwiseLossesReject(Api::DataType::FP32, Api::DataType::FP32, Api::DataType::FP32, maskDType);
}
