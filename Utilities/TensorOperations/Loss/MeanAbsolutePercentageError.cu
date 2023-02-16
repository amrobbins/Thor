#include "MeanAbsolutePercentageError.h"

using namespace std;

/**
 * MAPE(batch_of_predictions, batch_of_labels) = abs( (batch_of_labels - batch_of_predictions) / batch_of_labels) * 100
 * <p/>
 * When there are multiple predictions, there must be the corresponding number of labels.
 * This is enforced via assertion, the loss layer will not run if the size is not correct.
 * In that case the computation goes as:
 * <p/>
 * MSE(batch_of_predictions[0], batch_of_labels[0]) = (1/batchSize) * abs(batch_of_predictions[0] - batch_of_labels[0])
 * MSE(batch_of_predictions[1], batch_of_labels[1]) = (1/batchSize) * abs(batch_of_predictions[1] - batch_of_labels[1])
 * ...
 * <p/>
 * So, the number of losses computed is equal to the number of predictions that are made, and each loss gradient back propagates
 * through the associated prediction only.
 * <p/>
 * d/dx(100 abs((y - x)/y)) = 100 * (x - y) * (sqrt( (y-x)^2 / y^2 ) / (y - x)^2)
 *
 * <p/>
 * <p/>
 *
 *    Note: for stability
 *    when epsilon > label >= 0.0 label = epsilon
 *    when -epsilon < label < 0.0 label = -epsilon
 *    To turn off this feature set epsilon = 0 - but then the stability of the metric is dependent on the input values
 *    <p/>
 *    Note: for stability
 *    The max loss is capped at maxValue, gradient is clipped between -maxValue and maxValue
 *    To turn off this feature set maxValue = 0 - but then the stability of the metric is dependent on the input values
 */
__global__ void meanAbsolutePercentageError(half *labels,
                                            half *predictions,
                                            half *elementLoss,
                                            half *gradient,
                                            uint32_t numElements,
                                            bool computeGradient,
                                            float epsilon,
                                            float maxMagnitude) {
    int element = blockIdx.x * 1024 + (4 * threadIdx.x);

    if (element >= numElements)
        return;

    const half2 zero(0.0f, 0.0f);
    const half2 negativeOne(-1.0f, -1.0f);
    const half2 oneHundred(100.0f, 100.0f);
    const half2 positiveEpsilon(epsilon, epsilon);
    const half2 negativeEpsilon(-epsilon, -epsilon);
    const half2 maxValue(maxMagnitude, maxMagnitude);
    const half2 minValue(-maxMagnitude, -maxMagnitude);

    double *predictions_half4 = (double *)predictions;
    double predictionsBuffer_half4[1];
    predictionsBuffer_half4[0] = predictions_half4[element / 4];
    half *predictionsBuffer = (half *)predictionsBuffer_half4;
    half elementLossBuffer[4];
    half gradientBuffer[4];

    // Always process 4 elements, even when past last element because tensors are always padded to
    // be multiples of 8 bytes (4 half variables) to allow this. This is done for performance reasons.
    double *labels_half4 = (double *)labels;
    double labelsBuffer_half4[1];
    labelsBuffer_half4[0] = labels_half4[element / 4];
    half *labelsBuffer = (half *)labelsBuffer_half4;

    // When prediction and label are both exactly 0, the error and gradient should be 0:
    half2 labelsPredictionsEqual0 = __heq2(((half2 *)labelsBuffer)[0], ((half2 *)predictionsBuffer)[0]);
    half2 labelsPredictionsEqual1 = __heq2(((half2 *)labelsBuffer)[1], ((half2 *)predictionsBuffer)[1]);

    // Ensure label is not between (epsilon and negative epsilon)
    half2 isLessThanZero = __hlt2(((half2 *)labelsBuffer)[0], zero);
    half2 isGreaterThanNegativeEpsilon = __hgt2(((half2 *)labelsBuffer)[0], negativeEpsilon);
    half2 isGreaterThanPositiveEpsilon = __hgt2(((half2 *)labelsBuffer)[0], positiveEpsilon);
    if (positiveEpsilon.x > zero.x) {
        if (isLessThanZero.x) {
            if (isGreaterThanNegativeEpsilon.x)
                labelsBuffer[0] = negativeEpsilon.x;
        } else {
            if (!isGreaterThanPositiveEpsilon.x)
                labelsBuffer[0] = positiveEpsilon.x;
        }
        if (isLessThanZero.y) {
            if (isGreaterThanNegativeEpsilon.y)
                labelsBuffer[1] = negativeEpsilon.y;
        } else {
            if (!isGreaterThanPositiveEpsilon.y)
                labelsBuffer[1] = positiveEpsilon.y;
        }
    }

    isLessThanZero = __hlt2(((half2 *)labelsBuffer)[1], zero);
    isGreaterThanNegativeEpsilon = __hgt2(((half2 *)labelsBuffer)[1], negativeEpsilon);
    isGreaterThanPositiveEpsilon = __hgt2(((half2 *)labelsBuffer)[1], positiveEpsilon);
    if (positiveEpsilon.x > zero.x) {
        if (isLessThanZero.x) {
            if (isGreaterThanNegativeEpsilon.x)
                labelsBuffer[2] = negativeEpsilon.x;
        } else {
            if (!isGreaterThanPositiveEpsilon.x)
                labelsBuffer[2] = positiveEpsilon.x;
        }
        if (isLessThanZero.y) {
            if (isGreaterThanNegativeEpsilon.y)
                labelsBuffer[3] = negativeEpsilon.y;
        } else {
            if (!isGreaterThanPositiveEpsilon.y)
                labelsBuffer[3] = positiveEpsilon.y;
        }
    }

    half2 xMinusY0, xMinusY1;

    // MAPE(batch_of_predictions, batch_of_labels) = abs( (batch_of_labels - batch_of_predictions) / batch_of_labels) * 100
    half2 isLessThanMax;
    half2 isGreaterThanMin;
    xMinusY0 = __hsub2(((half2 *)predictionsBuffer)[0], ((half2 *)labelsBuffer)[0]);
    ((half2 *)elementLossBuffer)[0] = __hmul2(__habs2(__h2div(xMinusY0, ((half2 *)labelsBuffer)[0])), oneHundred);
    if (maxMagnitude > 0.0f) {
        isLessThanMax = __hlt2(((half2 *)elementLossBuffer)[0], maxValue);
        if (!isLessThanMax.x) {
            elementLossBuffer[0] = maxValue.x;
        }
        if (!isLessThanMax.y) {
            elementLossBuffer[1] = maxValue.x;
        }
    }

    xMinusY1 = __hsub2(((half2 *)predictionsBuffer)[1], ((half2 *)labelsBuffer)[1]);
    ((half2 *)elementLossBuffer)[1] = __hmul2(__habs2(__h2div(xMinusY1, ((half2 *)labelsBuffer)[1])), oneHundred);
    if (maxMagnitude > 0.0f) {
        isLessThanMax = __hlt2(((half2 *)elementLossBuffer)[1], maxValue);
        if (!isLessThanMax.x) {
            elementLossBuffer[2] = maxValue.x;
        }
        if (!isLessThanMax.y) {
            elementLossBuffer[3] = maxValue.x;
        }
    }

    if (computeGradient) {
        // d/dx(100 abs((y - x)/y)) = 100 * (x - y) * (sqrt( (y-x)^2 / y^2 ) / (y - x)^2)
        half2 yMinusXSquared = __hmul2(xMinusY0, xMinusY0);
        half2 ySquared = __hmul2(((half2 *)labelsBuffer)[0], ((half2 *)labelsBuffer)[0]);
        half2 sqrtYMinusXSquaredOverYSquared = h2sqrt(__h2div(yMinusXSquared, ySquared));
        ((half2 *)gradientBuffer)[0] = __h2div(__hmul2(oneHundred, __hmul2(xMinusY0, sqrtYMinusXSquaredOverYSquared)), yMinusXSquared);
        // if (half(fabsf(labelsBuffer[0])) == half(epsilon) && predictionsBuffer[0] == half(0.0f))
        //     printf("predictions %f labels %f xmy %f ymxsq %f ysq %f sqrtQ %f loss %f\n", (float)predictionsBuffer[0],
        //     (float)labelsBuffer[0], (float)xMinusY0.x, (float)yMinusXSquared.x, (float)ySquared.x,
        //     (float)sqrtYMinusXSquaredOverYSquared.x, (float)elementLossBuffer[0]);
        // if (half(fabsf(labelsBuffer[1])) == half(epsilon) && predictionsBuffer[1] == half(0.0f))
        //     printf("predictions %f labels %f xmy %f ymxsq %f ysq %f sqrtQ %f loss %f\n", (float)predictionsBuffer[1],
        //     (float)labelsBuffer[1], (float)xMinusY0.y, (float)yMinusXSquared.y, (float)ySquared.y,
        //     (float)sqrtYMinusXSquaredOverYSquared.y, (float)elementLossBuffer[1]);

        if (maxMagnitude > 0.0f) {
            isLessThanMax = __hlt2(((half2 *)gradientBuffer)[0], maxValue);
            isGreaterThanMin = __hgt2(((half2 *)gradientBuffer)[0], minValue);
            if (isLessThanMax.x) {
                if (!isGreaterThanMin.x)
                    gradientBuffer[0] = minValue.x;
            } else {
                gradientBuffer[0] = maxValue.x;
            }
            if (isLessThanMax.y) {
                if (!isGreaterThanMin.y)
                    gradientBuffer[1] = minValue.x;
            } else {
                gradientBuffer[1] = maxValue.x;
            }
        }

        yMinusXSquared = __hmul2(xMinusY1, xMinusY1);
        ySquared = __hmul2(((half2 *)labelsBuffer)[1], ((half2 *)labelsBuffer)[1]);
        sqrtYMinusXSquaredOverYSquared = h2sqrt(__h2div(yMinusXSquared, ySquared));
        ((half2 *)gradientBuffer)[1] = __h2div(__hmul2(oneHundred, __hmul2(xMinusY1, sqrtYMinusXSquaredOverYSquared)), yMinusXSquared);
        // if (half(fabsf(labelsBuffer[2])) == half(epsilon) && predictionsBuffer[2] == half(0.0f))
        //     printf("predictions %f labels %f xmy %f ymxsq %f ysq %f sqrtQ %f loss %f\n", (float)predictionsBuffer[2],
        //     (float)labelsBuffer[2], (float)xMinusY1.x, (float)yMinusXSquared.x, (float)ySquared.x,
        //     (float)sqrtYMinusXSquaredOverYSquared.x, (float)elementLossBuffer[2]);
        // if (half(fabsf(labelsBuffer[3])) == half(epsilon) && predictionsBuffer[3] == half(0.0f))
        //     printf("predictions %f labels %f xmy %f ymxsq %f ysq %f sqrtQ %f loss %f\n", (float)predictionsBuffer[3],
        //     (float)labelsBuffer[3], (float)xMinusY1.y, (float)yMinusXSquared.y, (float)ySquared.y,
        //     (float)sqrtYMinusXSquaredOverYSquared.y, (float)elementLossBuffer[3]);

        if (maxMagnitude > 0.0f) {
            isLessThanMax = __hlt2(((half2 *)gradientBuffer)[1], maxValue);
            isGreaterThanMin = __hgt2(((half2 *)gradientBuffer)[1], minValue);
            if (isLessThanMax.x) {
                if (!isGreaterThanMin.x)
                    gradientBuffer[2] = minValue.x;
            } else {
                gradientBuffer[2] = maxValue.x;
            }
            if (isLessThanMax.y) {
                if (!isGreaterThanMin.y)
                    gradientBuffer[3] = minValue.x;
            } else {
                gradientBuffer[3] = maxValue.x;
            }
        }
    }

    if (labelsPredictionsEqual0.x) {
        gradientBuffer[0] = 0.0f;
        elementLossBuffer[0] = 0.0f;
    }
    if (labelsPredictionsEqual0.y) {
        gradientBuffer[1] = 0.0f;
        elementLossBuffer[1] = 0.0f;
    }
    if (labelsPredictionsEqual1.x) {
        gradientBuffer[2] = 0.0f;
        elementLossBuffer[2] = 0.0f;
    }
    if (labelsPredictionsEqual1.y) {
        gradientBuffer[3] = 0.0f;
        elementLossBuffer[3] = 0.0f;
    }

    if (computeGradient) {
        double *gradientBuffer_half4 = (double *)gradientBuffer;
        double *gradient_half4 = (double *)gradient;
        gradient_half4[element / 4] = gradientBuffer_half4[0];
    }

    double *elementLossBuffer_half4 = (double *)elementLossBuffer;
    double *elementLoss_half4 = (double *)elementLoss;
    elementLoss_half4[element / 4] = elementLossBuffer_half4[0];
}

__global__ void meanAbsolutePercentageError(float *labels,
                                            half *predictions,
                                            half *elementLoss,
                                            half *gradient,
                                            uint32_t numElements,
                                            bool computeGradient,
                                            float epsilon,
                                            float maxMagnitude) {
    int element = blockIdx.x * 1024 + (4 * threadIdx.x);

    if (element >= numElements)
        return;

    const half2 zero(0.0f, 0.0f);
    const half2 negativeOne(-1.0f, -1.0f);
    const half2 oneHundred(100.0f, 100.0f);
    const half2 positiveEpsilon(epsilon, epsilon);
    const half2 negativeEpsilon(-epsilon, -epsilon);
    const half2 maxValue(maxMagnitude, maxMagnitude);
    const half2 minValue(-maxMagnitude, -maxMagnitude);

    double *predictions_half4 = (double *)predictions;
    double predictionsBuffer_half4[1];
    predictionsBuffer_half4[0] = predictions_half4[element / 4];
    half *predictionsBuffer = (half *)predictionsBuffer_half4;
    half elementLossBuffer[4];
    half gradientBuffer[4];

    // Always process 4 elements, even when past last element because tensors are always padded to
    // be multiples of 8 bytes (4 half variables) to allow this. This is done for performance reasons.
    float4 *labels_float4 = (float4 *)labels;
    float4 labelsBuffer_float4[1];
    labelsBuffer_float4[0] = labels_float4[element / 4];
    half2 labelsBuffer_half4[2];
    labelsBuffer_half4[0] = __float22half2_rn(((float2 *)labelsBuffer_float4)[0]);
    labelsBuffer_half4[1] = __float22half2_rn(((float2 *)labelsBuffer_float4)[1]);
    half *labelsBuffer = (half *)labelsBuffer_half4;

    // When prediction and label are both exactly 0, the error and gradient should be 0:
    half2 labelsPredictionsEqual0 = __heq2(((half2 *)labelsBuffer)[0], ((half2 *)predictionsBuffer)[0]);
    half2 labelsPredictionsEqual1 = __heq2(((half2 *)labelsBuffer)[1], ((half2 *)predictionsBuffer)[1]);

    // Ensure label is not between (epsilon and negative epsilon)
    half2 isLessThanZero = __hlt2(((half2 *)labelsBuffer)[0], zero);
    half2 isGreaterThanNegativeEpsilon = __hgt2(((half2 *)labelsBuffer)[0], negativeEpsilon);
    half2 isGreaterThanPositiveEpsilon = __hgt2(((half2 *)labelsBuffer)[0], positiveEpsilon);
    if (positiveEpsilon.x > zero.x) {
        if (isLessThanZero.x) {
            if (isGreaterThanNegativeEpsilon.x)
                labelsBuffer[0] = negativeEpsilon.x;
        } else {
            if (!isGreaterThanPositiveEpsilon.x)
                labelsBuffer[0] = positiveEpsilon.x;
        }
        if (isLessThanZero.y) {
            if (isGreaterThanNegativeEpsilon.y)
                labelsBuffer[1] = negativeEpsilon.y;
        } else {
            if (!isGreaterThanPositiveEpsilon.y)
                labelsBuffer[1] = positiveEpsilon.y;
        }
    }

    isLessThanZero = __hlt2(((half2 *)labelsBuffer)[1], zero);
    isGreaterThanNegativeEpsilon = __hgt2(((half2 *)labelsBuffer)[1], negativeEpsilon);
    isGreaterThanPositiveEpsilon = __hgt2(((half2 *)labelsBuffer)[1], positiveEpsilon);
    if (positiveEpsilon.x > zero.x) {
        if (isLessThanZero.x) {
            if (isGreaterThanNegativeEpsilon.x)
                labelsBuffer[2] = negativeEpsilon.x;
        } else {
            if (!isGreaterThanPositiveEpsilon.x)
                labelsBuffer[2] = positiveEpsilon.x;
        }
        if (isLessThanZero.y) {
            if (isGreaterThanNegativeEpsilon.y)
                labelsBuffer[3] = negativeEpsilon.y;
        } else {
            if (!isGreaterThanPositiveEpsilon.y)
                labelsBuffer[3] = positiveEpsilon.y;
        }
    }

    half2 xMinusY0, xMinusY1;

    // MAPE(batch_of_predictions, batch_of_labels) = abs( (batch_of_labels - batch_of_predictions) / batch_of_labels) * 100
    half2 isLessThanMax;
    half2 isGreaterThanMin;
    xMinusY0 = __hsub2(((half2 *)predictionsBuffer)[0], ((half2 *)labelsBuffer)[0]);
    ((half2 *)elementLossBuffer)[0] = __hmul2(__habs2(__h2div(xMinusY0, ((half2 *)labelsBuffer)[0])), oneHundred);
    if (maxMagnitude > 0.0f) {
        isLessThanMax = __hlt2(((half2 *)elementLossBuffer)[0], maxValue);
        if (!isLessThanMax.x) {
            elementLossBuffer[0] = maxValue.x;
        }
        if (!isLessThanMax.y) {
            elementLossBuffer[1] = maxValue.x;
        }
    }

    xMinusY1 = __hsub2(((half2 *)predictionsBuffer)[1], ((half2 *)labelsBuffer)[1]);
    ((half2 *)elementLossBuffer)[1] = __hmul2(__habs2(__h2div(xMinusY1, ((half2 *)labelsBuffer)[1])), oneHundred);
    if (maxMagnitude > 0.0f) {
        isLessThanMax = __hlt2(((half2 *)elementLossBuffer)[1], maxValue);
        if (!isLessThanMax.x) {
            elementLossBuffer[2] = maxValue.x;
        }
        if (!isLessThanMax.y) {
            elementLossBuffer[3] = maxValue.x;
        }
    }

    if (computeGradient) {
        // d/dx(100 abs((y - x)/y)) = 100 * (x - y) * (sqrt( (y-x)^2 / y^2 ) / (y - x)^2)
        half2 yMinusXSquared = __hmul2(xMinusY0, xMinusY0);
        half2 ySquared = __hmul2(((half2 *)labelsBuffer)[0], ((half2 *)labelsBuffer)[0]);
        half2 sqrtYMinusXSquaredOverYSquared = h2sqrt(__h2div(yMinusXSquared, ySquared));
        ((half2 *)gradientBuffer)[0] = __h2div(__hmul2(oneHundred, __hmul2(xMinusY0, sqrtYMinusXSquaredOverYSquared)), yMinusXSquared);
        // if (half(fabsf(labelsBuffer[0])) == half(epsilon) && predictionsBuffer[0] == half(0.0f))
        //     printf("predictions %f labels %f xmy %f ymxsq %f ysq %f sqrtQ %f loss %f\n", (float)predictionsBuffer[0],
        //     (float)labelsBuffer[0], (float)xMinusY0.x, (float)yMinusXSquared.x, (float)ySquared.x,
        //     (float)sqrtYMinusXSquaredOverYSquared.x, (float)elementLossBuffer[0]);
        // if (half(fabsf(labelsBuffer[1])) == half(epsilon) && predictionsBuffer[1] == half(0.0f))
        //     printf("predictions %f labels %f xmy %f ymxsq %f ysq %f sqrtQ %f loss %f\n", (float)predictionsBuffer[1],
        //     (float)labelsBuffer[1], (float)xMinusY0.y, (float)yMinusXSquared.y, (float)ySquared.y,
        //     (float)sqrtYMinusXSquaredOverYSquared.y, (float)elementLossBuffer[1]);

        if (maxMagnitude > 0.0f) {
            isLessThanMax = __hlt2(((half2 *)gradientBuffer)[0], maxValue);
            isGreaterThanMin = __hgt2(((half2 *)gradientBuffer)[0], minValue);
            if (isLessThanMax.x) {
                if (!isGreaterThanMin.x)
                    gradientBuffer[0] = minValue.x;
            } else {
                gradientBuffer[0] = maxValue.x;
            }
            if (isLessThanMax.y) {
                if (!isGreaterThanMin.y)
                    gradientBuffer[1] = minValue.x;
            } else {
                gradientBuffer[1] = maxValue.x;
            }
        }

        yMinusXSquared = __hmul2(xMinusY1, xMinusY1);
        ySquared = __hmul2(((half2 *)labelsBuffer)[1], ((half2 *)labelsBuffer)[1]);
        sqrtYMinusXSquaredOverYSquared = h2sqrt(__h2div(yMinusXSquared, ySquared));
        ((half2 *)gradientBuffer)[1] = __h2div(__hmul2(oneHundred, __hmul2(xMinusY1, sqrtYMinusXSquaredOverYSquared)), yMinusXSquared);
        // if (half(fabsf(labelsBuffer[2])) == half(epsilon) && predictionsBuffer[2] == half(0.0f))
        //     printf("predictions %f labels %f xmy %f ymxsq %f ysq %f sqrtQ %f loss %f\n", (float)predictionsBuffer[2],
        //     (float)labelsBuffer[2], (float)xMinusY1.x, (float)yMinusXSquared.x, (float)ySquared.x,
        //     (float)sqrtYMinusXSquaredOverYSquared.x, (float)elementLossBuffer[2]);
        // if (half(fabsf(labelsBuffer[3])) == half(epsilon) && predictionsBuffer[3] == half(0.0f))
        //     printf("predictions %f labels %f xmy %f ymxsq %f ysq %f sqrtQ %f loss %f\n", (float)predictionsBuffer[3],
        //     (float)labelsBuffer[3], (float)xMinusY1.y, (float)yMinusXSquared.y, (float)ySquared.y,
        //     (float)sqrtYMinusXSquaredOverYSquared.y, (float)elementLossBuffer[3]);

        if (maxMagnitude > 0.0f) {
            isLessThanMax = __hlt2(((half2 *)gradientBuffer)[1], maxValue);
            isGreaterThanMin = __hgt2(((half2 *)gradientBuffer)[1], minValue);
            if (isLessThanMax.x) {
                if (!isGreaterThanMin.x)
                    gradientBuffer[2] = minValue.x;
            } else {
                gradientBuffer[2] = maxValue.x;
            }
            if (isLessThanMax.y) {
                if (!isGreaterThanMin.y)
                    gradientBuffer[3] = minValue.x;
            } else {
                gradientBuffer[3] = maxValue.x;
            }
        }
    }

    if (labelsPredictionsEqual0.x) {
        gradientBuffer[0] = 0.0f;
        elementLossBuffer[0] = 0.0f;
    }
    if (labelsPredictionsEqual0.y) {
        gradientBuffer[1] = 0.0f;
        elementLossBuffer[1] = 0.0f;
    }
    if (labelsPredictionsEqual1.x) {
        gradientBuffer[2] = 0.0f;
        elementLossBuffer[2] = 0.0f;
    }
    if (labelsPredictionsEqual1.y) {
        gradientBuffer[3] = 0.0f;
        elementLossBuffer[3] = 0.0f;
    }

    if (computeGradient) {
        double *gradientBuffer_half4 = (double *)gradientBuffer;
        double *gradient_half4 = (double *)gradient;
        gradient_half4[element / 4] = gradientBuffer_half4[0];
    }

    double *elementLossBuffer_half4 = (double *)elementLossBuffer;
    double *elementLoss_half4 = (double *)elementLoss;
    elementLoss_half4[element / 4] = elementLossBuffer_half4[0];
}

__global__ void meanAbsolutePercentageError(float *labels,
                                            float *predictions,
                                            float *elementLoss,
                                            float *gradient,
                                            uint32_t numElements,
                                            bool computeGradient,
                                            float epsilon,
                                            float maxMagnitude) {
    int element = blockIdx.x * 1024 + (2 * threadIdx.x);

    if (element >= numElements)
        return;

    float2 *labels_float2 = (float2 *)labels;
    float2 labelsBuffer;

    float2 *predictions_float2 = (float2 *)predictions;
    float2 predictionsBuffer;

    float2 elementLossBuffer;
    float2 *elementLoss_float2 = (float2 *)elementLoss;

    float2 gradientBuffer;
    float2 *gradient_float2 = (float2 *)gradient;

    float xMinusY0, xMinusY1;
    float yMinusXSquared;
    const float positiveEpsilon = epsilon;
    const float negativeEpsilon = -epsilon;
    const float maxValue = maxMagnitude;
    const float minValue = -maxMagnitude;
    bool labelsPredictionsEqual0;
    bool labelsPredictionsEqual1;

    labelsBuffer = labels_float2[element / 2];
    predictionsBuffer = predictions_float2[element / 2];
    labelsPredictionsEqual0 = labelsBuffer.x == predictionsBuffer.x;
    labelsPredictionsEqual1 = labelsBuffer.y == predictionsBuffer.y;

    if (positiveEpsilon > 0.0f) {
        if (labelsBuffer.x < 0.0f) {
            if (labelsBuffer.x > negativeEpsilon)
                labelsBuffer.x = negativeEpsilon;
        } else {
            if (!(labelsBuffer.x > positiveEpsilon))
                labelsBuffer.x = positiveEpsilon;
        }
        if (labelsBuffer.y < 0.0f) {
            if (labelsBuffer.y > negativeEpsilon)
                labelsBuffer.y = negativeEpsilon;
        } else {
            if (!(labelsBuffer.y > positiveEpsilon))
                labelsBuffer.y = positiveEpsilon;
        }
    }

    // MAPE(batch_of_predictions, batch_of_labels) = abs( (batch_of_labels - batch_of_predictions) / batch_of_labels ) * 100
    // d/dx(100 abs((y - x)/y)) = 100 * (x - y) * (sqrt( (y-x)^2 / y^2 ) / (y - x)^2)
    xMinusY0 = predictionsBuffer.x - labelsBuffer.x;
    elementLossBuffer.x = 100.0f * fabsf(xMinusY0 / labelsBuffer.x);
    xMinusY1 = predictionsBuffer.y - labelsBuffer.y;
    elementLossBuffer.y = 100.0f * fabsf(xMinusY1 / labelsBuffer.y);
    if (maxMagnitude > 0.0f) {
        if (!(elementLossBuffer.x < maxValue)) {
            elementLossBuffer.x = maxValue;
        }
        if (!(elementLossBuffer.y < maxValue)) {
            elementLossBuffer.y = maxValue;
        }
    }

    if (computeGradient) {
        yMinusXSquared = xMinusY0 * xMinusY0;
        gradientBuffer.x = 100.0f * xMinusY0 * sqrtf(yMinusXSquared / (labelsBuffer.x * labelsBuffer.x)) / yMinusXSquared;

        yMinusXSquared = xMinusY1 * xMinusY1;
        gradientBuffer.y = 100.0f * xMinusY1 * sqrtf(yMinusXSquared / (labelsBuffer.y * labelsBuffer.y)) / yMinusXSquared;

        if (maxMagnitude > 0.0f) {
            if (gradientBuffer.x < maxValue) {
                if (gradientBuffer.x < minValue)
                    gradientBuffer.x = minValue;
            } else {
                gradientBuffer.x = maxValue;
            }
            if (gradientBuffer.y < maxValue) {
                if (gradientBuffer.y < minValue)
                    gradientBuffer.y = minValue;
            } else {
                gradientBuffer.y = maxValue;
            }
        }

        if (labelsPredictionsEqual0)
            gradientBuffer.x = 0.0f;
        if (labelsPredictionsEqual1)
            gradientBuffer.y = 0.0f;

        gradient_float2[element / 2] = gradientBuffer;
    }

    if (labelsPredictionsEqual0)
        elementLossBuffer.x = 0.0f;
    if (labelsPredictionsEqual1)
        elementLossBuffer.y = 0.0f;

    // Tensors are always padded to be multiples of 8 bytes (4 half variables) to allow this, without the possibility
    // of indexing out of bounds.
    elementLoss_float2[element / 2] = elementLossBuffer;

    element += 512;
    if (element >= numElements)
        return;

    labelsBuffer = labels_float2[element / 2];
    predictionsBuffer = predictions_float2[element / 2];
    labelsPredictionsEqual0 = labelsBuffer.x == predictionsBuffer.x;
    labelsPredictionsEqual1 = labelsBuffer.y == predictionsBuffer.y;

    if (positiveEpsilon > 0.0f) {
        if (labelsBuffer.x < 0.0f) {
            if (labelsBuffer.x > negativeEpsilon)
                labelsBuffer.x = negativeEpsilon;
        } else {
            if (!(labelsBuffer.x > positiveEpsilon))
                labelsBuffer.x = positiveEpsilon;
        }
        if (labelsBuffer.y < 0.0f) {
            if (labelsBuffer.y > negativeEpsilon)
                labelsBuffer.y = negativeEpsilon;
        } else {
            if (!(labelsBuffer.y > positiveEpsilon))
                labelsBuffer.y = positiveEpsilon;
        }
    }

    // MAPE(batch_of_predictions, batch_of_labels) = abs( (batch_of_labels - batch_of_predictions) / batch_of_labels ) * 100
    // d/dx(100 abs((y - x)/y)) = 100 * (x - y) * (sqrt( (y-x)^2 / y^2 ) / (y - x)^2)
    xMinusY0 = predictionsBuffer.x - labelsBuffer.x;
    elementLossBuffer.x = 100.0f * fabsf(xMinusY0 / labelsBuffer.x);
    xMinusY1 = predictionsBuffer.y - labelsBuffer.y;
    elementLossBuffer.y = 100.0f * fabsf(xMinusY1 / labelsBuffer.y);
    if (maxMagnitude > 0.0f) {
        if (!(elementLossBuffer.x < maxValue)) {
            elementLossBuffer.x = maxValue;
        }
        if (!(elementLossBuffer.y < maxValue)) {
            elementLossBuffer.y = maxValue;
        }
    }

    if (computeGradient) {
        yMinusXSquared = xMinusY0 * xMinusY0;
        gradientBuffer.x = 100.0f * xMinusY0 * sqrtf(yMinusXSquared / (labelsBuffer.x * labelsBuffer.x)) / yMinusXSquared;

        yMinusXSquared = xMinusY1 * xMinusY1;
        gradientBuffer.y = 100.0f * xMinusY1 * sqrtf(yMinusXSquared / (labelsBuffer.y * labelsBuffer.y)) / yMinusXSquared;

        if (maxMagnitude > 0.0f) {
            if (gradientBuffer.x < maxValue) {
                if (gradientBuffer.x < minValue)
                    gradientBuffer.x = minValue;
            } else {
                gradientBuffer.x = maxValue;
            }
            if (gradientBuffer.y < maxValue) {
                if (gradientBuffer.y < minValue)
                    gradientBuffer.y = minValue;
            } else {
                gradientBuffer.y = maxValue;
            }
        }

        if (labelsPredictionsEqual0)
            gradientBuffer.x = 0.0f;
        if (labelsPredictionsEqual1)
            gradientBuffer.y = 0.0f;

        gradient_float2[element / 2] = gradientBuffer;
    }

    if (labelsPredictionsEqual0)
        elementLossBuffer.x = 0.0f;
    if (labelsPredictionsEqual1)
        elementLossBuffer.y = 0.0f;

    // Tensors are always padded to be multiples of 8 bytes (4 half variables) to allow this, without the possibility
    // of indexing out of bounds.
    elementLoss_float2[element / 2] = elementLossBuffer;
}

template <typename LABEL_TYPE, typename PREDICTION_TYPE, typename LOSS_TYPE>
__global__ void meanAbsolutePercentageError(LABEL_TYPE *labels,
                                            PREDICTION_TYPE *predictions,
                                            LOSS_TYPE *elementLoss,
                                            PREDICTION_TYPE *gradient,
                                            uint32_t numElements,
                                            bool computeGradient,
                                            float epsilon,
                                            float maxMagnitude) {
    for (int32_t element = blockIdx.x * 1024 + threadIdx.x, iteration = 0; element < numElements && iteration < 4;
         element += 256, iteration += 1) {
        float labelsBuffer;
        float predictionsBuffer;
        float elementLossBuffer;
        float gradientBuffer;

        float xMinusY;
        float yMinusXSquared;
        const float positiveEpsilon = epsilon;
        const float negativeEpsilon = -epsilon;
        const float maxValue = maxMagnitude;
        const float minValue = -maxMagnitude;
        bool labelsPredictionsEqual;

        labelsBuffer = (float)labels[element];
        predictionsBuffer = (float)predictions[element];
        labelsPredictionsEqual = labelsBuffer == predictionsBuffer;

        if (positiveEpsilon > 0.0f) {
            if (labelsBuffer < 0.0f) {
                if (labelsBuffer > negativeEpsilon)
                    labelsBuffer = negativeEpsilon;
            } else {
                if (!(labelsBuffer > positiveEpsilon))
                    labelsBuffer = positiveEpsilon;
            }
        }

        // MAPE(batch_of_predictions, batch_of_labels) = abs( (batch_of_labels - batch_of_predictions) / batch_of_labels ) * 100
        // d/dx(100 abs((y - x)/y)) = 100 * (x - y) * (sqrt( (y-x)^2 / y^2 ) / (y - x)^2)
        xMinusY = predictionsBuffer - labelsBuffer;
        elementLossBuffer = 100.0f * fabsf(xMinusY / labelsBuffer);
        if (maxMagnitude > 0.0f) {
            if (!(elementLossBuffer < maxValue)) {
                elementLossBuffer = maxValue;
            }
        }

        if (computeGradient) {
            yMinusXSquared = xMinusY * xMinusY;
            gradientBuffer = 100.0f * xMinusY * sqrtf(yMinusXSquared / (labelsBuffer * labelsBuffer)) / yMinusXSquared;

            if (maxMagnitude > 0.0f) {
                if (gradientBuffer < maxValue) {
                    if (gradientBuffer < minValue)
                        gradientBuffer = minValue;
                } else {
                    gradientBuffer = maxValue;
                }
            }

            if (labelsPredictionsEqual)
                gradientBuffer = 0.0f;
            gradient[element] = (PREDICTION_TYPE)gradientBuffer;
        }

        // Tensors are always padded to be multiples of 8 bytes (4 half variables) to allow this, without the possibility
        // of indexing out of bounds.
        if (labelsPredictionsEqual)
            elementLossBuffer = 0.0f;
        elementLoss[element] = (LOSS_TYPE)elementLossBuffer;
    }
}

template <typename LABEL_TYPE, typename PREDICTION_TYPE, typename LOSS_TYPE>
void launchMeanAbsolutePercentageError(void *labels_d,
                                       void *predictions_d,
                                       void *elementLoss_d,
                                       void *gradient_d,
                                       uint32_t numPredictions,
                                       uint32_t batchSize,
                                       bool computeGradient,
                                       Stream stream,
                                       float epsilon,
                                       float maxMagnitude) {
    uint32_t numElements = batchSize * numPredictions;

    dim3 blockSize(min(256, numElements));
    dim3 gridSize((numElements + 1023) / 1024);
    ScopedGpu scopedGpu(stream.getGpuNum());

    meanAbsolutePercentageError<<<gridSize, blockSize, 0, stream>>>((LABEL_TYPE *)labels_d,
                                                                    (PREDICTION_TYPE *)predictions_d,
                                                                    (LOSS_TYPE *)elementLoss_d,
                                                                    (PREDICTION_TYPE *)gradient_d,
                                                                    numElements,
                                                                    computeGradient,
                                                                    epsilon,
                                                                    maxMagnitude);
}

template void launchMeanAbsolutePercentageError<half, half, half>(void *labels_d,
                                                                  void *predictions_d,
                                                                  void *elementLoss_d,
                                                                  void *gradient,
                                                                  uint32_t numPredictions,
                                                                  uint32_t batchSize,
                                                                  bool computeGradient,
                                                                  Stream stream,
                                                                  float epsilon,
                                                                  float maxMagnitude);

template void launchMeanAbsolutePercentageError<half, half, float>(void *labels_d,
                                                                   void *predictions_d,
                                                                   void *elementLoss_d,
                                                                   void *gradient,
                                                                   uint32_t numPredictions,
                                                                   uint32_t batchSize,
                                                                   bool computeGradient,
                                                                   Stream stream,
                                                                   float epsilon,
                                                                   float maxMagnitude);

template void launchMeanAbsolutePercentageError<half, float, half>(void *labels_d,
                                                                   void *predictions_d,
                                                                   void *elementLoss_d,
                                                                   void *gradient,
                                                                   uint32_t numPredictions,
                                                                   uint32_t batchSize,
                                                                   bool computeGradient,
                                                                   Stream stream,
                                                                   float epsilon,
                                                                   float maxMagnitude);

template void launchMeanAbsolutePercentageError<half, float, float>(void *labels_d,
                                                                    void *predictions_d,
                                                                    void *elementLoss_d,
                                                                    void *gradient,
                                                                    uint32_t numPredictions,
                                                                    uint32_t batchSize,
                                                                    bool computeGradient,
                                                                    Stream stream,
                                                                    float epsilon,
                                                                    float maxMagnitude);

template void launchMeanAbsolutePercentageError<float, half, half>(void *labels_d,
                                                                   void *predictions_d,
                                                                   void *elementLoss_d,
                                                                   void *gradient,
                                                                   uint32_t numPredictions,
                                                                   uint32_t batchSize,
                                                                   bool computeGradient,
                                                                   Stream stream,
                                                                   float epsilon,
                                                                   float maxMagnitude);

template void launchMeanAbsolutePercentageError<float, half, float>(void *labels_d,
                                                                    void *predictions_d,
                                                                    void *elementLoss_d,
                                                                    void *gradient,
                                                                    uint32_t numPredictions,
                                                                    uint32_t batchSize,
                                                                    bool computeGradient,
                                                                    Stream stream,
                                                                    float epsilon,
                                                                    float maxMagnitude);

template void launchMeanAbsolutePercentageError<float, float, half>(void *labels_d,
                                                                    void *predictions_d,
                                                                    void *elementLoss_d,
                                                                    void *gradient,
                                                                    uint32_t numPredictions,
                                                                    uint32_t batchSize,
                                                                    bool computeGradient,
                                                                    Stream stream,
                                                                    float epsilon,
                                                                    float maxMagnitude);

template void launchMeanAbsolutePercentageError<float, float, float>(void *labels_d,
                                                                     void *predictions_d,
                                                                     void *elementLoss_d,
                                                                     void *gradient,
                                                                     uint32_t numPredictions,
                                                                     uint32_t batchSize,
                                                                     bool computeGradient,
                                                                     Stream stream,
                                                                     float epsilon,
                                                                     float maxMagnitude);

// uint32_t
template void launchMeanAbsolutePercentageError<uint32_t, half, half>(void *labels_d,
                                                                      void *predictions_d,
                                                                      void *elementLoss_d,
                                                                      void *gradient,
                                                                      uint32_t numPredictions,
                                                                      uint32_t batchSize,
                                                                      bool computeGradient,
                                                                      Stream stream,
                                                                      float epsilon,
                                                                      float maxMagnitude);

template void launchMeanAbsolutePercentageError<uint32_t, half, float>(void *labels_d,
                                                                       void *predictions_d,
                                                                       void *elementLoss_d,
                                                                       void *gradient,
                                                                       uint32_t numPredictions,
                                                                       uint32_t batchSize,
                                                                       bool computeGradient,
                                                                       Stream stream,
                                                                       float epsilon,
                                                                       float maxMagnitude);

template void launchMeanAbsolutePercentageError<uint32_t, float, half>(void *labels_d,
                                                                       void *predictions_d,
                                                                       void *elementLoss_d,
                                                                       void *gradient,
                                                                       uint32_t numPredictions,
                                                                       uint32_t batchSize,
                                                                       bool computeGradient,
                                                                       Stream stream,
                                                                       float epsilon,
                                                                       float maxMagnitude);

template void launchMeanAbsolutePercentageError<uint32_t, float, float>(void *labels_d,
                                                                        void *predictions_d,
                                                                        void *elementLoss_d,
                                                                        void *gradient,
                                                                        uint32_t numPredictions,
                                                                        uint32_t batchSize,
                                                                        bool computeGradient,
                                                                        Stream stream,
                                                                        float epsilon,
                                                                        float maxMagnitude);

// uint16_t
template void launchMeanAbsolutePercentageError<uint16_t, half, half>(void *labels_d,
                                                                      void *predictions_d,
                                                                      void *elementLoss_d,
                                                                      void *gradient,
                                                                      uint32_t numPredictions,
                                                                      uint32_t batchSize,
                                                                      bool computeGradient,
                                                                      Stream stream,
                                                                      float epsilon,
                                                                      float maxMagnitude);

template void launchMeanAbsolutePercentageError<uint16_t, half, float>(void *labels_d,
                                                                       void *predictions_d,
                                                                       void *elementLoss_d,
                                                                       void *gradient,
                                                                       uint32_t numPredictions,
                                                                       uint32_t batchSize,
                                                                       bool computeGradient,
                                                                       Stream stream,
                                                                       float epsilon,
                                                                       float maxMagnitude);

template void launchMeanAbsolutePercentageError<uint16_t, float, half>(void *labels_d,
                                                                       void *predictions_d,
                                                                       void *elementLoss_d,
                                                                       void *gradient,
                                                                       uint32_t numPredictions,
                                                                       uint32_t batchSize,
                                                                       bool computeGradient,
                                                                       Stream stream,
                                                                       float epsilon,
                                                                       float maxMagnitude);

template void launchMeanAbsolutePercentageError<uint16_t, float, float>(void *labels_d,
                                                                        void *predictions_d,
                                                                        void *elementLoss_d,
                                                                        void *gradient,
                                                                        uint32_t numPredictions,
                                                                        uint32_t batchSize,
                                                                        bool computeGradient,
                                                                        Stream stream,
                                                                        float epsilon,
                                                                        float maxMagnitude);

// uint8_t
template void launchMeanAbsolutePercentageError<uint8_t, half, half>(void *labels_d,
                                                                     void *predictions_d,
                                                                     void *elementLoss_d,
                                                                     void *gradient,
                                                                     uint32_t numPredictions,
                                                                     uint32_t batchSize,
                                                                     bool computeGradient,
                                                                     Stream stream,
                                                                     float epsilon,
                                                                     float maxMagnitude);

template void launchMeanAbsolutePercentageError<uint8_t, half, float>(void *labels_d,
                                                                      void *predictions_d,
                                                                      void *elementLoss_d,
                                                                      void *gradient,
                                                                      uint32_t numPredictions,
                                                                      uint32_t batchSize,
                                                                      bool computeGradient,
                                                                      Stream stream,
                                                                      float epsilon,
                                                                      float maxMagnitude);

template void launchMeanAbsolutePercentageError<uint8_t, float, half>(void *labels_d,
                                                                      void *predictions_d,
                                                                      void *elementLoss_d,
                                                                      void *gradient,
                                                                      uint32_t numPredictions,
                                                                      uint32_t batchSize,
                                                                      bool computeGradient,
                                                                      Stream stream,
                                                                      float epsilon,
                                                                      float maxMagnitude);

template void launchMeanAbsolutePercentageError<uint8_t, float, float>(void *labels_d,
                                                                       void *predictions_d,
                                                                       void *elementLoss_d,
                                                                       void *gradient,
                                                                       uint32_t numPredictions,
                                                                       uint32_t batchSize,
                                                                       bool computeGradient,
                                                                       Stream stream,
                                                                       float epsilon,
                                                                       float maxMagnitude);

// int32_t
template void launchMeanAbsolutePercentageError<int32_t, half, half>(void *labels_d,
                                                                     void *predictions_d,
                                                                     void *elementLoss_d,
                                                                     void *gradient,
                                                                     uint32_t numPredictions,
                                                                     uint32_t batchSize,
                                                                     bool computeGradient,
                                                                     Stream stream,
                                                                     float epsilon,
                                                                     float maxMagnitude);

template void launchMeanAbsolutePercentageError<int32_t, half, float>(void *labels_d,
                                                                      void *predictions_d,
                                                                      void *elementLoss_d,
                                                                      void *gradient,
                                                                      uint32_t numPredictions,
                                                                      uint32_t batchSize,
                                                                      bool computeGradient,
                                                                      Stream stream,
                                                                      float epsilon,
                                                                      float maxMagnitude);

template void launchMeanAbsolutePercentageError<int32_t, float, half>(void *labels_d,
                                                                      void *predictions_d,
                                                                      void *elementLoss_d,
                                                                      void *gradient,
                                                                      uint32_t numPredictions,
                                                                      uint32_t batchSize,
                                                                      bool computeGradient,
                                                                      Stream stream,
                                                                      float epsilon,
                                                                      float maxMagnitude);

template void launchMeanAbsolutePercentageError<int32_t, float, float>(void *labels_d,
                                                                       void *predictions_d,
                                                                       void *elementLoss_d,
                                                                       void *gradient,
                                                                       uint32_t numPredictions,
                                                                       uint32_t batchSize,
                                                                       bool computeGradient,
                                                                       Stream stream,
                                                                       float epsilon,
                                                                       float maxMagnitude);

// int16_t
template void launchMeanAbsolutePercentageError<int16_t, half, half>(void *labels_d,
                                                                     void *predictions_d,
                                                                     void *elementLoss_d,
                                                                     void *gradient,
                                                                     uint32_t numPredictions,
                                                                     uint32_t batchSize,
                                                                     bool computeGradient,
                                                                     Stream stream,
                                                                     float epsilon,
                                                                     float maxMagnitude);

template void launchMeanAbsolutePercentageError<int16_t, half, float>(void *labels_d,
                                                                      void *predictions_d,
                                                                      void *elementLoss_d,
                                                                      void *gradient,
                                                                      uint32_t numPredictions,
                                                                      uint32_t batchSize,
                                                                      bool computeGradient,
                                                                      Stream stream,
                                                                      float epsilon,
                                                                      float maxMagnitude);

template void launchMeanAbsolutePercentageError<int16_t, float, half>(void *labels_d,
                                                                      void *predictions_d,
                                                                      void *elementLoss_d,
                                                                      void *gradient,
                                                                      uint32_t numPredictions,
                                                                      uint32_t batchSize,
                                                                      bool computeGradient,
                                                                      Stream stream,
                                                                      float epsilon,
                                                                      float maxMagnitude);

template void launchMeanAbsolutePercentageError<int16_t, float, float>(void *labels_d,
                                                                       void *predictions_d,
                                                                       void *elementLoss_d,
                                                                       void *gradient,
                                                                       uint32_t numPredictions,
                                                                       uint32_t batchSize,
                                                                       bool computeGradient,
                                                                       Stream stream,
                                                                       float epsilon,
                                                                       float maxMagnitude);

// int8_t
template void launchMeanAbsolutePercentageError<int8_t, half, half>(void *labels_d,
                                                                    void *predictions_d,
                                                                    void *elementLoss_d,
                                                                    void *gradient,
                                                                    uint32_t numPredictions,
                                                                    uint32_t batchSize,
                                                                    bool computeGradient,
                                                                    Stream stream,
                                                                    float epsilon,
                                                                    float maxMagnitude);

template void launchMeanAbsolutePercentageError<int8_t, half, float>(void *labels_d,
                                                                     void *predictions_d,
                                                                     void *elementLoss_d,
                                                                     void *gradient,
                                                                     uint32_t numPredictions,
                                                                     uint32_t batchSize,
                                                                     bool computeGradient,
                                                                     Stream stream,
                                                                     float epsilon,
                                                                     float maxMagnitude);

template void launchMeanAbsolutePercentageError<int8_t, float, half>(void *labels_d,
                                                                     void *predictions_d,
                                                                     void *elementLoss_d,
                                                                     void *gradient,
                                                                     uint32_t numPredictions,
                                                                     uint32_t batchSize,
                                                                     bool computeGradient,
                                                                     Stream stream,
                                                                     float epsilon,
                                                                     float maxMagnitude);

template void launchMeanAbsolutePercentageError<int8_t, float, float>(void *labels_d,
                                                                      void *predictions_d,
                                                                      void *elementLoss_d,
                                                                      void *gradient,
                                                                      uint32_t numPredictions,
                                                                      uint32_t batchSize,
                                                                      bool computeGradient,
                                                                      Stream stream,
                                                                      float epsilon,
                                                                      float maxMagnitude);

// bool
template void launchMeanAbsolutePercentageError<bool, half, half>(void *labels_d,
                                                                  void *predictions_d,
                                                                  void *elementLoss_d,
                                                                  void *gradient,
                                                                  uint32_t numPredictions,
                                                                  uint32_t batchSize,
                                                                  bool computeGradient,
                                                                  Stream stream,
                                                                  float epsilon,
                                                                  float maxMagnitude);

template void launchMeanAbsolutePercentageError<bool, half, float>(void *labels_d,
                                                                   void *predictions_d,
                                                                   void *elementLoss_d,
                                                                   void *gradient,
                                                                   uint32_t numPredictions,
                                                                   uint32_t batchSize,
                                                                   bool computeGradient,
                                                                   Stream stream,
                                                                   float epsilon,
                                                                   float maxMagnitude);

template void launchMeanAbsolutePercentageError<bool, float, half>(void *labels_d,
                                                                   void *predictions_d,
                                                                   void *elementLoss_d,
                                                                   void *gradient,
                                                                   uint32_t numPredictions,
                                                                   uint32_t batchSize,
                                                                   bool computeGradient,
                                                                   Stream stream,
                                                                   float epsilon,
                                                                   float maxMagnitude);

template void launchMeanAbsolutePercentageError<bool, float, float>(void *labels_d,
                                                                    void *predictions_d,
                                                                    void *elementLoss_d,
                                                                    void *gradient,
                                                                    uint32_t numPredictions,
                                                                    uint32_t batchSize,
                                                                    bool computeGradient,
                                                                    Stream stream,
                                                                    float epsilon,
                                                                    float maxMagnitude);
