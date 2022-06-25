#include "MeanSquaredError.h"

using namespace std;

/**
 * MSE(batch_of_predictions, batch_of_labels) = (1/batchSize) * (batch_of_predictions - batch_of_labels)^2
 *
 * Where the subtraction and squaring are performed element-wise.
 *
 * When there are multiple predictions, there must be the corresponding number of labels.
 * This is enforced via assertion, the loss layer will not run if the size is not correct.
 * In that case the computation goes as:
 *
 * MSE(batch_of_predictions[0], batch_of_labels[0]) = (1/batchSize) * (batch_of_predictions[0] - batch_of_labels[0])^2
 * MSE(batch_of_predictions[1], batch_of_labels[1]) = (1/batchSize) * (batch_of_predictions[1] - batch_of_labels[1])^2
 * ...
 *
 * So, the number of losses computed is equal to the number of predictions that are made, and each loss back propagates
 * through the associated prediction only.
 */
template <typename LABEL_TYPE, typename PREDICTION_TYPE, typename LOSS_TYPE>
__global__ void meanSquaredError(LABEL_TYPE *labels, PREDICTION_TYPE *predictions, LOSS_TYPE *workspace, uint64_t numElements) {
    int32_t element = blockIdx.x * 1024 + threadIdx.x;

    LOSS_TYPE buffer;

    if (element >= numElements)
        return;
    buffer = (LOSS_TYPE)(float)labels[element] - (LOSS_TYPE)(float)predictions[element];
    workspace[element] = buffer * buffer;

    element += 256;
    if (element >= numElements)
        return;
    buffer = (LOSS_TYPE)(float)labels[element] - (LOSS_TYPE)(float)predictions[element];
    workspace[element] = buffer * buffer;

    element += 256;
    if (element >= numElements)
        return;
    buffer = (LOSS_TYPE)(float)labels[element] - (LOSS_TYPE)(float)predictions[element];
    workspace[element] = buffer * buffer;

    element += 256;
    if (element >= numElements)
        return;
    buffer = (LOSS_TYPE)(float)labels[element] - (LOSS_TYPE)(float)predictions[element];
    workspace[element] = buffer * buffer;
}

__global__ void meanSquaredError(half *labels, half *predictions, half *workspace, uint64_t numElements) {
    int element = blockIdx.x * 1024 + (4 * threadIdx.x);

    if (element >= numElements)
        return;

    // Always process 4 elements, even when past last element because tensors are always padded to
    // be multiples of 8 bytes (4 half variables) to allow this. This is done for performance reasons.
    double *labels_half4 = (double *)labels;
    double labelsBuffer_half4[1];
    labelsBuffer_half4[0] = labels_half4[element / 4];
    half *labelsBuffer = (half *)labelsBuffer_half4;

    double *predictions_half4 = (double *)predictions;
    double predictionsBuffer_half4[1];
    predictionsBuffer_half4[0] = predictions_half4[element / 4];
    half *predictionsBuffer = (half *)predictionsBuffer_half4;
    half workspaceBuffer[4];

    half2 buffer;

    buffer = __hsub2(((half2 *)labelsBuffer)[0], ((half2 *)predictionsBuffer)[0]);
    ((half2 *)workspaceBuffer)[0] = __hmul2(buffer, buffer);
    buffer = __hsub2(((half2 *)labelsBuffer)[1], ((half2 *)predictionsBuffer)[1]);
    ((half2 *)workspaceBuffer)[1] = __hmul2(buffer, buffer);

    double *workspaceBuffer_half4 = (double *)workspaceBuffer;
    double *workspace_half4 = (double *)workspace;
    workspace_half4[element / 4] = workspaceBuffer_half4[0];
}

__global__ void meanSquaredError(float *labels, half *predictions, half *workspace, uint64_t numElements) {
    int element = blockIdx.x * 1024 + (4 * threadIdx.x);

    if (element >= numElements)
        return;

    const half zero[2] = {(half)0.0f, (half)0.0f};

    // Always process 4 elements, even when past last element because tensors are always padded to
    // be multiples of 8 bytes (4 half variables) to allow this. This is done for performance reasons.
    float2 *labels_float2 = (float2 *)labels;
    float2 labelsBuffer_float2;
    half labelsBuffer_half4[4];

    labelsBuffer_float2 = labels_float2[element / 2];
    ((half2 *)labelsBuffer_half4)[0] = __float22half2_rn(labelsBuffer_float2);
    if (numElements + 2 >= numElements) {
        ((half2 *)labelsBuffer_half4)[1] = ((half2 *)zero)[0];
    } else {
        labelsBuffer_float2 = labels_float2[(element / 2) + 1];
        ((half2 *)labelsBuffer_half4)[1] = __float22half2_rn(labelsBuffer_float2);
    }
    half *labelsBuffer = (half *)labelsBuffer_half4;

    double *predictions_half4 = (double *)predictions;
    double predictionsBuffer_half4[1];
    predictionsBuffer_half4[0] = predictions_half4[element / 4];
    half *predictionsBuffer = (half *)predictionsBuffer_half4;
    half workspaceBuffer[4];

    half2 buffer;

    buffer = __hsub2(((half2 *)labelsBuffer)[0], ((half2 *)predictionsBuffer)[0]);
    ((half2 *)workspaceBuffer)[0] = __hmul2(buffer, buffer);
    buffer = __hsub2(((half2 *)labelsBuffer)[1], ((half2 *)predictionsBuffer)[1]);
    ((half2 *)workspaceBuffer)[1] = __hmul2(buffer, buffer);

    double *workspaceBuffer_half4 = (double *)workspaceBuffer;
    double *workspace_half4 = (double *)workspace;
    workspace_half4[element / 4] = workspaceBuffer_half4[0];
}

__global__ void meanSquaredError(float *labels, float *predictions, float *workspace, uint64_t numElements) {
    int element = blockIdx.x * 1024 + (2 * threadIdx.x);

    if (element >= numElements)
        return;

    float2 *labels_float2 = (float2 *)labels;
    float2 labelsBuffer;

    float2 *predictions_float2 = (float2 *)predictions;
    float2 predictionsBuffer;

    float2 workspaceBuffer;
    float2 *workspace_float2 = (float2 *)workspace;

    float buffer;

    labelsBuffer = labels_float2[element / 2];
    predictionsBuffer = predictions_float2[element / 2];

    buffer = labelsBuffer.x - predictionsBuffer.x;
    workspaceBuffer.x = buffer * buffer;

    if (element + 1 < numElements) {
        buffer = labelsBuffer.y - predictionsBuffer.y;
        workspaceBuffer.y = buffer * buffer;
    }

    // Tensors are always padded to be multiples of 8 bytes (4 half variables) to allow this, without the possibility
    // of indexing out of bounds.
    workspace_float2[element / 2] = workspaceBuffer;

    element += 512;
    if (element >= numElements)
        return;

    labelsBuffer = labels_float2[element / 2];
    predictionsBuffer = predictions_float2[element / 2];

    buffer = labelsBuffer.x - predictionsBuffer.x;
    workspaceBuffer.x = buffer * buffer;

    if (element + 1 < numElements) {
        buffer = labelsBuffer.y - predictionsBuffer.y;
        workspaceBuffer.y = buffer * buffer;
    }

    // Tensors are always padded to be multiples of 8 bytes (4 half variables) to allow this, without the possibility
    // of indexing out of bounds.
    workspace_float2[element / 2] = workspaceBuffer;
}

__global__ void meanSquaredError(float *labels, half *predictions, float *workspace, uint64_t numElements) {
    int element = blockIdx.x * 1024 + (2 * threadIdx.x);

    if (element >= numElements)
        return;

    float2 *labels_float2 = (float2 *)labels;
    float2 labelsBuffer;

    half2 *predictions_half2 = (half2 *)predictions;
    half2 predictionsBuffer_half2;
    float2 predictionsBuffer;

    float2 workspaceBuffer;
    float2 *workspace_float2 = (float2 *)workspace;

    float buffer;

    labelsBuffer = labels_float2[element / 2];
    predictionsBuffer_half2 = predictions_half2[element / 2];
    predictionsBuffer = __half22float2(predictionsBuffer_half2);

    buffer = labelsBuffer.x - predictionsBuffer.x;
    workspaceBuffer.x = buffer * buffer;

    if (element + 1 < numElements) {
        buffer = labelsBuffer.y - predictionsBuffer.y;
        workspaceBuffer.y = buffer * buffer;
    }

    // Tensors are always padded to be multiples of 8 bytes (4 half variables) to allow this, without the possibility
    // of indexing out of bounds.
    workspace_float2[element / 2] = workspaceBuffer;

    element += 512;
    if (element >= numElements)
        return;

    labelsBuffer = labels_float2[element / 2];
    predictionsBuffer_half2 = predictions_half2[element / 2];
    predictionsBuffer = __half22float2(predictionsBuffer_half2);

    buffer = labelsBuffer.x - predictionsBuffer.x;
    workspaceBuffer.x = buffer * buffer;

    if (element + 1 < numElements) {
        buffer = labelsBuffer.y - predictionsBuffer.y;
        workspaceBuffer.y = buffer * buffer;
    }

    // Tensors are always padded to be multiples of 8 bytes (4 half variables) to allow this, without the possibility
    // of indexing out of bounds.
    workspace_float2[element / 2] = workspaceBuffer;
}

template <typename LABEL_TYPE, typename PREDICTION_TYPE, typename LOSS_TYPE>
void launchMeanSquaredError(LABEL_TYPE *labels_d,
                            PREDICTION_TYPE *predictions_d,
                            LOSS_TYPE *loss_d,
                            LOSS_TYPE *workspace_d,
                            uint32_t numPredictions,
                            uint32_t batchSize,
                            Stream stream,
                            BatchReduce *batchReduce) {
    uint32_t numElements = batchSize * numPredictions;

    dim3 blockSize(min(256, batchSize * numPredictions));
    dim3 gridSize((numElements + 1023) / 1024);
    ScopedGpu scopedGpu(stream.getGpuNum());

    meanSquaredError<<<gridSize, blockSize, 0, stream>>>(labels_d, predictions_d, workspace_d, numElements);

    // Ensure that batchReduce stream is synchronized properly
    // It is done this way since the cudnnHandle belongs to the stream that batchReduce uses.
    Stream batchReduceStream = batchReduce->getStream();
    if (batchReduceStream != stream)
        batchReduceStream.waitEvent(stream.putEvent());

    batchReduce->reduce(workspace_d, loss_d);

    if (batchReduceStream != stream)
        stream.waitEvent(batchReduceStream.putEvent());
}

// hhh is custom

template void launchMeanSquaredError<half, half, float>(half *labels_d,
                                                        half *predictions_d,
                                                        float *loss_d,
                                                        float *workspace_d,
                                                        uint32_t numPredictions,
                                                        uint32_t batchSize,
                                                        Stream stream,
                                                        BatchReduce *batchReduce);

template void launchMeanSquaredError<half, float, half>(half *labels_d,
                                                        float *predictions_d,
                                                        half *loss_d,
                                                        half *workspace_d,
                                                        uint32_t numPredictions,
                                                        uint32_t batchSize,
                                                        Stream stream,
                                                        BatchReduce *batchReduce);

template void launchMeanSquaredError<half, float, float>(half *labels_d,
                                                         float *predictions_d,
                                                         float *loss_d,
                                                         float *workspace_d,
                                                         uint32_t numPredictions,
                                                         uint32_t batchSize,
                                                         Stream stream,
                                                         BatchReduce *batchReduce);

// fhh is custom

// fhf is custom

template void launchMeanSquaredError<float, float, half>(float *labels_d,
                                                         float *predictions_d,
                                                         half *loss_d,
                                                         half *workspace_d,
                                                         uint32_t numPredictions,
                                                         uint32_t batchSize,
                                                         Stream stream,
                                                         BatchReduce *batchReduce);

// fff is custom

// uint64_t
template void launchMeanSquaredError<uint64_t, half, half>(uint64_t *labels_d,
                                                           half *predictions_d,
                                                           half *loss_d,
                                                           half *workspace_d,
                                                           uint32_t numPredictions,
                                                           uint32_t batchSize,
                                                           Stream stream,
                                                           BatchReduce *batchReduce);

template void launchMeanSquaredError<uint64_t, half, float>(uint64_t *labels_d,
                                                            half *predictions_d,
                                                            float *loss_d,
                                                            float *workspace_d,
                                                            uint32_t numPredictions,
                                                            uint32_t batchSize,
                                                            Stream stream,
                                                            BatchReduce *batchReduce);

template void launchMeanSquaredError<uint64_t, float, half>(uint64_t *labels_d,
                                                            float *predictions_d,
                                                            half *loss_d,
                                                            half *workspace_d,
                                                            uint32_t numPredictions,
                                                            uint32_t batchSize,
                                                            Stream stream,
                                                            BatchReduce *batchReduce);

template void launchMeanSquaredError<uint64_t, float, float>(uint64_t *labels_d,
                                                             float *predictions_d,
                                                             float *loss_d,
                                                             float *workspace_d,
                                                             uint32_t numPredictions,
                                                             uint32_t batchSize,
                                                             Stream stream,
                                                             BatchReduce *batchReduce);

// uint32_t
template void launchMeanSquaredError<uint32_t, half, half>(uint32_t *labels_d,
                                                           half *predictions_d,
                                                           half *loss_d,
                                                           half *workspace_d,
                                                           uint32_t numPredictions,
                                                           uint32_t batchSize,
                                                           Stream stream,
                                                           BatchReduce *batchReduce);

template void launchMeanSquaredError<uint32_t, half, float>(uint32_t *labels_d,
                                                            half *predictions_d,
                                                            float *loss_d,
                                                            float *workspace_d,
                                                            uint32_t numPredictions,
                                                            uint32_t batchSize,
                                                            Stream stream,
                                                            BatchReduce *batchReduce);

template void launchMeanSquaredError<uint32_t, float, half>(uint32_t *labels_d,
                                                            float *predictions_d,
                                                            half *loss_d,
                                                            half *workspace_d,
                                                            uint32_t numPredictions,
                                                            uint32_t batchSize,
                                                            Stream stream,
                                                            BatchReduce *batchReduce);

template void launchMeanSquaredError<uint32_t, float, float>(uint32_t *labels_d,
                                                             float *predictions_d,
                                                             float *loss_d,
                                                             float *workspace_d,
                                                             uint32_t numPredictions,
                                                             uint32_t batchSize,
                                                             Stream stream,
                                                             BatchReduce *batchReduce);

// uint16_t
template void launchMeanSquaredError<uint16_t, half, half>(uint16_t *labels_d,
                                                           half *predictions_d,
                                                           half *loss_d,
                                                           half *workspace_d,
                                                           uint32_t numPredictions,
                                                           uint32_t batchSize,
                                                           Stream stream,
                                                           BatchReduce *batchReduce);

template void launchMeanSquaredError<uint16_t, half, float>(uint16_t *labels_d,
                                                            half *predictions_d,
                                                            float *loss_d,
                                                            float *workspace_d,
                                                            uint32_t numPredictions,
                                                            uint32_t batchSize,
                                                            Stream stream,
                                                            BatchReduce *batchReduce);

template void launchMeanSquaredError<uint16_t, float, half>(uint16_t *labels_d,
                                                            float *predictions_d,
                                                            half *loss_d,
                                                            half *workspace_d,
                                                            uint32_t numPredictions,
                                                            uint32_t batchSize,
                                                            Stream stream,
                                                            BatchReduce *batchReduce);

template void launchMeanSquaredError<uint16_t, float, float>(uint16_t *labels_d,
                                                             float *predictions_d,
                                                             float *loss_d,
                                                             float *workspace_d,
                                                             uint32_t numPredictions,
                                                             uint32_t batchSize,
                                                             Stream stream,
                                                             BatchReduce *batchReduce);

// uint8_t
template void launchMeanSquaredError<uint8_t, half, half>(uint8_t *labels_d,
                                                          half *predictions_d,
                                                          half *loss_d,
                                                          half *workspace_d,
                                                          uint32_t numPredictions,
                                                          uint32_t batchSize,
                                                          Stream stream,
                                                          BatchReduce *batchReduce);

template void launchMeanSquaredError<uint8_t, half, float>(uint8_t *labels_d,
                                                           half *predictions_d,
                                                           float *loss_d,
                                                           float *workspace_d,
                                                           uint32_t numPredictions,
                                                           uint32_t batchSize,
                                                           Stream stream,
                                                           BatchReduce *batchReduce);

template void launchMeanSquaredError<uint8_t, float, half>(uint8_t *labels_d,
                                                           float *predictions_d,
                                                           half *loss_d,
                                                           half *workspace_d,
                                                           uint32_t numPredictions,
                                                           uint32_t batchSize,
                                                           Stream stream,
                                                           BatchReduce *batchReduce);

template void launchMeanSquaredError<uint8_t, float, float>(uint8_t *labels_d,
                                                            float *predictions_d,
                                                            float *loss_d,
                                                            float *workspace_d,
                                                            uint32_t numPredictions,
                                                            uint32_t batchSize,
                                                            Stream stream,
                                                            BatchReduce *batchReduce);

// int64_t
template void launchMeanSquaredError<int64_t, half, half>(int64_t *labels_d,
                                                          half *predictions_d,
                                                          half *loss_d,
                                                          half *workspace_d,
                                                          uint32_t numPredictions,
                                                          uint32_t batchSize,
                                                          Stream stream,
                                                          BatchReduce *batchReduce);

template void launchMeanSquaredError<int64_t, half, float>(int64_t *labels_d,
                                                           half *predictions_d,
                                                           float *loss_d,
                                                           float *workspace_d,
                                                           uint32_t numPredictions,
                                                           uint32_t batchSize,
                                                           Stream stream,
                                                           BatchReduce *batchReduce);

template void launchMeanSquaredError<int64_t, float, half>(int64_t *labels_d,
                                                           float *predictions_d,
                                                           half *loss_d,
                                                           half *workspace_d,
                                                           uint32_t numPredictions,
                                                           uint32_t batchSize,
                                                           Stream stream,
                                                           BatchReduce *batchReduce);

template void launchMeanSquaredError<int64_t, float, float>(int64_t *labels_d,
                                                            float *predictions_d,
                                                            float *loss_d,
                                                            float *workspace_d,
                                                            uint32_t numPredictions,
                                                            uint32_t batchSize,
                                                            Stream stream,
                                                            BatchReduce *batchReduce);

// int32_t
template void launchMeanSquaredError<int32_t, half, half>(int32_t *labels_d,
                                                          half *predictions_d,
                                                          half *loss_d,
                                                          half *workspace_d,
                                                          uint32_t numPredictions,
                                                          uint32_t batchSize,
                                                          Stream stream,
                                                          BatchReduce *batchReduce);

template void launchMeanSquaredError<int32_t, half, float>(int32_t *labels_d,
                                                           half *predictions_d,
                                                           float *loss_d,
                                                           float *workspace_d,
                                                           uint32_t numPredictions,
                                                           uint32_t batchSize,
                                                           Stream stream,
                                                           BatchReduce *batchReduce);

template void launchMeanSquaredError<int32_t, float, half>(int32_t *labels_d,
                                                           float *predictions_d,
                                                           half *loss_d,
                                                           half *workspace_d,
                                                           uint32_t numPredictions,
                                                           uint32_t batchSize,
                                                           Stream stream,
                                                           BatchReduce *batchReduce);

template void launchMeanSquaredError<int32_t, float, float>(int32_t *labels_d,
                                                            float *predictions_d,
                                                            float *loss_d,
                                                            float *workspace_d,
                                                            uint32_t numPredictions,
                                                            uint32_t batchSize,
                                                            Stream stream,
                                                            BatchReduce *batchReduce);

// int16_t
template void launchMeanSquaredError<int16_t, half, half>(int16_t *labels_d,
                                                          half *predictions_d,
                                                          half *loss_d,
                                                          half *workspace_d,
                                                          uint32_t numPredictions,
                                                          uint32_t batchSize,
                                                          Stream stream,
                                                          BatchReduce *batchReduce);

template void launchMeanSquaredError<int16_t, half, float>(int16_t *labels_d,
                                                           half *predictions_d,
                                                           float *loss_d,
                                                           float *workspace_d,
                                                           uint32_t numPredictions,
                                                           uint32_t batchSize,
                                                           Stream stream,
                                                           BatchReduce *batchReduce);

template void launchMeanSquaredError<int16_t, float, half>(int16_t *labels_d,
                                                           float *predictions_d,
                                                           half *loss_d,
                                                           half *workspace_d,
                                                           uint32_t numPredictions,
                                                           uint32_t batchSize,
                                                           Stream stream,
                                                           BatchReduce *batchReduce);

template void launchMeanSquaredError<int16_t, float, float>(int16_t *labels_d,
                                                            float *predictions_d,
                                                            float *loss_d,
                                                            float *workspace_d,
                                                            uint32_t numPredictions,
                                                            uint32_t batchSize,
                                                            Stream stream,
                                                            BatchReduce *batchReduce);

// int8_t
template void launchMeanSquaredError<int8_t, half, half>(int8_t *labels_d,
                                                         half *predictions_d,
                                                         half *loss_d,
                                                         half *workspace_d,
                                                         uint32_t numPredictions,
                                                         uint32_t batchSize,
                                                         Stream stream,
                                                         BatchReduce *batchReduce);

template void launchMeanSquaredError<int8_t, half, float>(int8_t *labels_d,
                                                          half *predictions_d,
                                                          float *loss_d,
                                                          float *workspace_d,
                                                          uint32_t numPredictions,
                                                          uint32_t batchSize,
                                                          Stream stream,
                                                          BatchReduce *batchReduce);

template void launchMeanSquaredError<int8_t, float, half>(int8_t *labels_d,
                                                          float *predictions_d,
                                                          half *loss_d,
                                                          half *workspace_d,
                                                          uint32_t numPredictions,
                                                          uint32_t batchSize,
                                                          Stream stream,
                                                          BatchReduce *batchReduce);

template void launchMeanSquaredError<int8_t, float, float>(int8_t *labels_d,
                                                           float *predictions_d,
                                                           float *loss_d,
                                                           float *workspace_d,
                                                           uint32_t numPredictions,
                                                           uint32_t batchSize,
                                                           Stream stream,
                                                           BatchReduce *batchReduce);

// bool
template void launchMeanSquaredError<bool, half, half>(bool *labels_d,
                                                       half *predictions_d,
                                                       half *loss_d,
                                                       half *workspace_d,
                                                       uint32_t numPredictions,
                                                       uint32_t batchSize,
                                                       Stream stream,
                                                       BatchReduce *batchReduce);

template void launchMeanSquaredError<bool, half, float>(bool *labels_d,
                                                        half *predictions_d,
                                                        float *loss_d,
                                                        float *workspace_d,
                                                        uint32_t numPredictions,
                                                        uint32_t batchSize,
                                                        Stream stream,
                                                        BatchReduce *batchReduce);

template void launchMeanSquaredError<bool, float, half>(bool *labels_d,
                                                        float *predictions_d,
                                                        half *loss_d,
                                                        half *workspace_d,
                                                        uint32_t numPredictions,
                                                        uint32_t batchSize,
                                                        Stream stream,
                                                        BatchReduce *batchReduce);

template void launchMeanSquaredError<bool, float, float>(bool *labels_d,
                                                         float *predictions_d,
                                                         float *loss_d,
                                                         float *workspace_d,
                                                         uint32_t numPredictions,
                                                         uint32_t batchSize,
                                                         Stream stream,
                                                         BatchReduce *batchReduce);
