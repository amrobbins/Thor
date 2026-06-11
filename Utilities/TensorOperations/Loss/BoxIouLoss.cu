#include "Utilities/TensorOperations/Loss/BoxIouLoss.h"

#include "DeepLearning/Implementation/ThorError.h"

#include <cuda_runtime.h>

namespace {

constexpr float kPi = 3.141592653589793238462643383279502884f;

struct Box4f {
    float x1;
    float y1;
    float x2;
    float y2;
};

template <typename T>
__device__ float toFloat(T value) {
    return static_cast<float>(value);
}

template <typename T>
__device__ T fromFloat(float value) {
    return static_cast<T>(value);
}

template <>
__device__ half fromFloat<half>(float value) {
    return __float2half_rn(value);
}

template <typename T>
__device__ Box4f loadBox4(const T* boxes, uint32_t boxIndex) {
    const T* box = boxes + boxIndex * 4u;
    return Box4f{toFloat(box[0]), toFloat(box[1]), toFloat(box[2]), toFloat(box[3])};
}

template <>
__device__ Box4f loadBox4<float>(const float* boxes, uint32_t boxIndex) {
    // A box is a naturally contiguous [x1, y1, x2, y2] record. Loading it as
    // float4 gives each thread a single 16-byte transaction and makes each warp
    // consume a contiguous run of box records.
    const float4 packed = reinterpret_cast<const float4*>(boxes)[boxIndex];
    return Box4f{packed.x, packed.y, packed.z, packed.w};
}

template <>
__device__ Box4f loadBox4<half>(const half* boxes, uint32_t boxIndex) {
    // Half boxes are 8-byte records. Load as two half2 values to keep the same
    // contiguous-record access pattern without relying on a non-standard half4.
    const half2* packed = reinterpret_cast<const half2*>(boxes + boxIndex * 4u);
    const float2 xy = __half22float2(packed[0]);
    const float2 zw = __half22float2(packed[1]);
    return Box4f{xy.x, xy.y, zw.x, zw.y};
}

template <typename T>
__device__ void storeBox4(T* boxes, uint32_t boxIndex, const float* values) {
    T* box = boxes + boxIndex * 4u;
    box[0] = fromFloat<T>(values[0]);
    box[1] = fromFloat<T>(values[1]);
    box[2] = fromFloat<T>(values[2]);
    box[3] = fromFloat<T>(values[3]);
}

template <>
__device__ void storeBox4<float>(float* boxes, uint32_t boxIndex, const float* values) {
    reinterpret_cast<float4*>(boxes)[boxIndex] = make_float4(values[0], values[1], values[2], values[3]);
}

template <>
__device__ void storeBox4<half>(half* boxes, uint32_t boxIndex, const float* values) {
    half2* packed = reinterpret_cast<half2*>(boxes + boxIndex * 4u);
    packed[0] = __floats2half2_rn(values[0], values[1]);
    packed[1] = __floats2half2_rn(values[2], values[3]);
}


__device__ float positiveLengthScalar(float hi, float lo) { return fmaxf(hi - lo, 0.0f); }
__device__ float squareScalar(float value) { return value * value; }

__device__ float boxIouLossValueScalar(BoxIouLossKind kind, const Box4f& p, const Box4f& t, float eps) {
    const float pW = positiveLengthScalar(p.x2, p.x1);
    const float pH = positiveLengthScalar(p.y2, p.y1);
    const float tW = positiveLengthScalar(t.x2, t.x1);
    const float tH = positiveLengthScalar(t.y2, t.y1);
    const float pArea = pW * pH;
    const float tArea = tW * tH;

    const float interW = positiveLengthScalar(fminf(p.x2, t.x2), fmaxf(p.x1, t.x1));
    const float interH = positiveLengthScalar(fminf(p.y2, t.y2), fmaxf(p.y1, t.y1));
    const float interArea = interW * interH;
    const float unionArea = fmaxf(pArea + tArea - interArea, eps);
    const float iou = interArea / (unionArea + eps);

    float loss = 1.0f - iou;
    if (kind == BoxIouLossKind::GIOU) {
        const float enclosingW = positiveLengthScalar(fmaxf(p.x2, t.x2), fminf(p.x1, t.x1));
        const float enclosingH = positiveLengthScalar(fmaxf(p.y2, t.y2), fminf(p.y1, t.y1));
        const float enclosingArea = fmaxf(enclosingW * enclosingH, eps);
        loss += (enclosingArea - unionArea) / (enclosingArea + eps);
    } else if (kind == BoxIouLossKind::DIOU || kind == BoxIouLossKind::CIOU) {
        const float enclosingW = positiveLengthScalar(fmaxf(p.x2, t.x2), fminf(p.x1, t.x1));
        const float enclosingH = positiveLengthScalar(fmaxf(p.y2, t.y2), fminf(p.y1, t.y1));
        const float enclosingDiagSq = fmaxf(squareScalar(enclosingW) + squareScalar(enclosingH) + eps, eps);
        const float pCx2 = p.x1 + p.x2;
        const float pCy2 = p.y1 + p.y2;
        const float tCx2 = t.x1 + t.x2;
        const float tCy2 = t.y1 + t.y2;
        const float centerDistanceSq = (squareScalar(pCx2 - tCx2) + squareScalar(pCy2 - tCy2)) * 0.25f;
        loss += centerDistanceSq / enclosingDiagSq;

        if (kind == BoxIouLossKind::CIOU) {
            const float v = squareScalar(atanf(tW / (tH + eps)) - atanf(pW / (pH + eps))) * (4.0f / (kPi * kPi));
            const float alpha = v / ((1.0f - iou) + v + eps);
            loss += alpha * v;
        }
    }

    return loss;
}

struct D4 {
    float v;
    float g[4];
};

__device__ D4 constant(float value) {
    D4 result;
    result.v = value;
#pragma unroll
    for (int i = 0; i < 4; ++i)
        result.g[i] = 0.0f;
    return result;
}

__device__ D4 variable(float value, int coordinate) {
    D4 result = constant(value);
    result.g[coordinate] = 1.0f;
    return result;
}

__device__ D4 operator+(const D4& a, const D4& b) {
    D4 result;
    result.v = a.v + b.v;
#pragma unroll
    for (int i = 0; i < 4; ++i)
        result.g[i] = a.g[i] + b.g[i];
    return result;
}

__device__ D4 operator+(const D4& a, float b) { return a + constant(b); }

__device__ D4 operator-(const D4& a, const D4& b) {
    D4 result;
    result.v = a.v - b.v;
#pragma unroll
    for (int i = 0; i < 4; ++i)
        result.g[i] = a.g[i] - b.g[i];
    return result;
}

__device__ D4 operator-(float a, const D4& b) { return constant(a) - b; }

__device__ D4 operator*(const D4& a, const D4& b) {
    D4 result;
    result.v = a.v * b.v;
#pragma unroll
    for (int i = 0; i < 4; ++i)
        result.g[i] = a.g[i] * b.v + a.v * b.g[i];
    return result;
}

__device__ D4 operator*(const D4& a, float b) {
    D4 result;
    result.v = a.v * b;
#pragma unroll
    for (int i = 0; i < 4; ++i)
        result.g[i] = a.g[i] * b;
    return result;
}
__device__ D4 operator/(const D4& a, const D4& b) {
    D4 result;
    const float invB = 1.0f / b.v;
    const float invB2 = invB * invB;
    result.v = a.v * invB;
#pragma unroll
    for (int i = 0; i < 4; ++i)
        result.g[i] = (a.g[i] * b.v - a.v * b.g[i]) * invB2;
    return result;
}

__device__ D4 dmin(const D4& a, const D4& b) { return (a.v <= b.v) ? a : b; }
__device__ D4 dmax(const D4& a, const D4& b) { return (a.v >= b.v) ? a : b; }
__device__ D4 positiveLength(const D4& hi, const D4& lo) { return dmax(hi - lo, constant(0.0f)); }
__device__ D4 square(const D4& value) { return value * value; }

__device__ D4 datan(const D4& value) {
    D4 result;
    result.v = atanf(value.v);
    const float scale = 1.0f / (1.0f + value.v * value.v);
#pragma unroll
    for (int i = 0; i < 4; ++i)
        result.g[i] = value.g[i] * scale;
    return result;
}

struct BoxTerms {
    D4 x1;
    D4 y1;
    D4 x2;
    D4 y2;
    D4 w;
    D4 h;
    D4 area;
};

__device__ BoxTerms predictionTerms(const Box4f& box) {
    D4 x1 = variable(box.x1, 0);
    D4 y1 = variable(box.y1, 1);
    D4 x2 = variable(box.x2, 2);
    D4 y2 = variable(box.y2, 3);
    D4 w = positiveLength(x2, x1);
    D4 h = positiveLength(y2, y1);
    return BoxTerms{x1, y1, x2, y2, w, h, w * h};
}

__device__ BoxTerms labelTerms(const Box4f& box) {
    D4 x1 = constant(box.x1);
    D4 y1 = constant(box.y1);
    D4 x2 = constant(box.x2);
    D4 y2 = constant(box.y2);
    D4 w = positiveLength(x2, x1);
    D4 h = positiveLength(y2, y1);
    return BoxTerms{x1, y1, x2, y2, w, h, w * h};
}

__device__ D4 boxIouLossValue(BoxIouLossKind kind, const Box4f& predictionBox, const Box4f& labelBox, float eps) {
    BoxTerms p = predictionTerms(predictionBox);
    BoxTerms t = labelTerms(labelBox);

    const D4 interX1 = dmax(p.x1, t.x1);
    const D4 interY1 = dmax(p.y1, t.y1);
    const D4 interX2 = dmin(p.x2, t.x2);
    const D4 interY2 = dmin(p.y2, t.y2);
    const D4 interW = positiveLength(interX2, interX1);
    const D4 interH = positiveLength(interY2, interY1);
    const D4 interArea = interW * interH;
    const D4 unionArea = dmax(p.area + t.area - interArea, constant(eps));
    const D4 iou = interArea / (unionArea + eps);

    const D4 enclosingX1 = dmin(p.x1, t.x1);
    const D4 enclosingY1 = dmin(p.y1, t.y1);
    const D4 enclosingX2 = dmax(p.x2, t.x2);
    const D4 enclosingY2 = dmax(p.y2, t.y2);
    const D4 enclosingW = positiveLength(enclosingX2, enclosingX1);
    const D4 enclosingH = positiveLength(enclosingY2, enclosingY1);
    const D4 enclosingArea = dmax(enclosingW * enclosingH, constant(eps));
    const D4 enclosingDiagSq = dmax(square(enclosingW) + square(enclosingH) + eps, constant(eps));

    D4 loss = 1.0f - iou;
    if (kind == BoxIouLossKind::GIOU) {
        loss = loss + ((enclosingArea - unionArea) / (enclosingArea + eps));
    } else if (kind == BoxIouLossKind::DIOU || kind == BoxIouLossKind::CIOU) {
        const D4 pCx2 = p.x1 + p.x2;
        const D4 pCy2 = p.y1 + p.y2;
        const D4 tCx2 = t.x1 + t.x2;
        const D4 tCy2 = t.y1 + t.y2;
        const D4 centerDistanceSq = (square(pCx2 - tCx2) + square(pCy2 - tCy2)) * 0.25f;
        loss = loss + (centerDistanceSq / enclosingDiagSq);

        if (kind == BoxIouLossKind::CIOU) {
            const D4 atanTarget = datan(t.w / (t.h + eps));
            const D4 atanPred = datan(p.w / (p.h + eps));
            const D4 v = square(atanTarget - atanPred) * (4.0f / (kPi * kPi));
            const D4 alpha = v / ((1.0f - iou) + v + eps);
            loss = loss + (alpha * v);
        }
    }

    return loss;
}

template <typename LABEL_TYPE, typename PREDICTION_TYPE, typename LOSS_TYPE, bool HasLossWeight>
__global__ void boxIouLossKernel(const LABEL_TYPE* __restrict__ labels,
                                 const PREDICTION_TYPE* __restrict__ predictions,
                                 LOSS_TYPE* __restrict__ loss,
                                 PREDICTION_TYPE* __restrict__ gradient,
                                 uint32_t numBoxes,
                                 BoxIouLossKind kind,
                                 float eps,
                                 bool computeGradient,
                                 float lossScalingFactor,
                                 float lossWeight) {
    const uint32_t boxIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (boxIndex >= numBoxes)
        return;

    const Box4f predictionBox = loadBox4(predictions, boxIndex);
    const Box4f labelBox = loadBox4(labels, boxIndex);

    if (!computeGradient) {
        float rawLoss = boxIouLossValueScalar(kind, predictionBox, labelBox, eps);
        if constexpr (HasLossWeight) {
            rawLoss *= lossWeight;
        }
        loss[boxIndex] = fromFloat<LOSS_TYPE>(rawLoss);
        return;
    }

    const D4 value = boxIouLossValue(kind, predictionBox, labelBox, eps);
    float rawLoss = value.v;
    float gradientScale = lossScalingFactor;
    if constexpr (HasLossWeight) {
        rawLoss *= lossWeight;
        gradientScale *= lossWeight;
    }
    loss[boxIndex] = fromFloat<LOSS_TYPE>(rawLoss);

    float scaledGradient[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
        scaledGradient[i] = value.g[i] * gradientScale;
    storeBox4(gradient, boxIndex, scaledGradient);
}

}  // namespace

template <typename LABEL_TYPE, typename PREDICTION_TYPE, typename LOSS_TYPE>
void launchBoxIouLoss(void* labels_d,
                      void* predictions_d,
                      void* loss_d,
                      void* gradient_d,
                      uint32_t numBoxes,
                      BoxIouLossKind kind,
                      float eps,
                      bool computeGradient,
                      float lossScalingFactor,
                      std::optional<float> lossWeight,
                      Stream stream) {
    constexpr uint32_t threadsPerBlock = 256;
    const uint32_t blocks = (numBoxes + threadsPerBlock - 1u) / threadsPerBlock;
    if (blocks == 0)
        return;

    if (lossWeight.has_value()) {
        boxIouLossKernel<LABEL_TYPE, PREDICTION_TYPE, LOSS_TYPE, true><<<blocks, threadsPerBlock, 0, stream.getStream()>>>(
            static_cast<const LABEL_TYPE*>(labels_d),
            static_cast<const PREDICTION_TYPE*>(predictions_d),
            static_cast<LOSS_TYPE*>(loss_d),
            static_cast<PREDICTION_TYPE*>(gradient_d),
            numBoxes,
            kind,
            eps,
            computeGradient,
            lossScalingFactor,
            lossWeight.value());
    } else {
        boxIouLossKernel<LABEL_TYPE, PREDICTION_TYPE, LOSS_TYPE, false><<<blocks, threadsPerBlock, 0, stream.getStream()>>>(
            static_cast<const LABEL_TYPE*>(labels_d),
            static_cast<const PREDICTION_TYPE*>(predictions_d),
            static_cast<LOSS_TYPE*>(loss_d),
            static_cast<PREDICTION_TYPE*>(gradient_d),
            numBoxes,
            kind,
            eps,
            computeGradient,
            lossScalingFactor,
            1.0f);
    }
}

template void launchBoxIouLoss<half, half, half>(void*, void*, void*, void*, uint32_t, BoxIouLossKind, float, bool, float, std::optional<float>, Stream);
template void launchBoxIouLoss<half, half, float>(void*, void*, void*, void*, uint32_t, BoxIouLossKind, float, bool, float, std::optional<float>, Stream);
template void launchBoxIouLoss<float, half, half>(void*, void*, void*, void*, uint32_t, BoxIouLossKind, float, bool, float, std::optional<float>, Stream);
template void launchBoxIouLoss<float, half, float>(void*, void*, void*, void*, uint32_t, BoxIouLossKind, float, bool, float, std::optional<float>, Stream);
template void launchBoxIouLoss<half, float, half>(void*, void*, void*, void*, uint32_t, BoxIouLossKind, float, bool, float, std::optional<float>, Stream);
template void launchBoxIouLoss<half, float, float>(void*, void*, void*, void*, uint32_t, BoxIouLossKind, float, bool, float, std::optional<float>, Stream);
template void launchBoxIouLoss<float, float, half>(void*, void*, void*, void*, uint32_t, BoxIouLossKind, float, bool, float, std::optional<float>, Stream);
template void launchBoxIouLoss<float, float, float>(void*, void*, void*, void*, uint32_t, BoxIouLossKind, float, bool, float, std::optional<float>, Stream);
