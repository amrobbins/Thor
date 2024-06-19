#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "Utilities/TensorOperations/Misc/Map.h"

#include <type_traits>

namespace ThorImplementation {

template <typename INDEX_TYPE>
class Map : public Layer {
   public:
    virtual ~Map() {}

    Map() { uninitialized = true; }

    // There is one entry per element in the destination tensor, the entry contains the index of the element in the
    // source tensor that will be populated in the dest tensor.
    // The shape of the dest tensor is the same as the shape of the mappingOfSourceTensorIntoDestTensor Tensor.
    //
    // Note: if more than one instance of a mapping is wanted, all instances after the first need to be called as:
    // Map(mappingOfSourceTensorIntoDestTensor, sourceDimensions, existingInstance.getReverseMapping()) {
    // This is because creating the backward pass mapping is a very heavy-weight operation that only needs to be done once.
    Map(Tensor mappingOfSourceTensorIntoDestTensor, std::vector<unsigned long> sourceDimensions) {
        setup(mappingOfSourceTensorIntoDestTensor, sourceDimensions);
        createBackwardPassMapping(mappingOfSourceTensorIntoDestTensor);
    }

    Map(Tensor mappingOfSourceTensorIntoDestTensor,
        std::vector<unsigned long> sourceDimensions,
        std::map<unsigned int, Tensor> backwardPassMappingHost,
        std::map<unsigned int, Tensor> backwardPassMappingDevice) {
        setup(mappingOfSourceTensorIntoDestTensor, sourceDimensions);
        backwardPassMappingOfNTo1Host = backwardPassMappingHost;
        backwardPassMappingOfNTo1Device = backwardPassMappingDevice;
    }

    virtual Optional<Tensor> createFeatureOutputTensor() {
        assert(!uninitialized);
        assert(featureInput.isPresent());

        TensorDescriptor newDescriptor = TensorDescriptor(featureInput.get().getDescriptor().getDataType(),
                                                          mappingOfSourceTensorIntoDestTensor.getDescriptor().getDimensions());
        return Tensor(featureInput.get().getPlacement(), newDescriptor);
    }

    virtual void compile() {
        assert(featureInput.isPresent());
        assert(featureInput.get().getDescriptor().getDimensions() == sourceDimensions);

        assert(mappingOfSourceTensorIntoDestTensor.getPlacement() == featureInput.get().getPlacement());
        assert(featureOutput.get().getDescriptor().getDimensions() == mappingOfSourceTensorIntoDestTensor.getDescriptor().getDimensions());
        if (errorInput.isPresent())
            assert(errorInput.get().getDescriptor().getDimensions() == mappingOfSourceTensorIntoDestTensor.getDescriptor().getDimensions());
        if (errorOutput.isPresent() && backwardPassMappingOfNTo1Device.empty()) {
            assert(!backwardPassMappingOfNTo1Host.empty());
            for (auto it = backwardPassMappingOfNTo1Host.begin(); it != backwardPassMappingOfNTo1Host.end(); ++it) {
                uint64_t N = it->first;
                Tensor mappingTensorHost = it->second;
                backwardPassMappingOfNTo1Device[N] = mappingTensorHost.clone(featureInput.get().getPlacement());
                backwardPassMappingOfNTo1Device[N].copyFromAsync(featureInput, stream);
            }
            stream.synchronize();
        }
    }

    virtual void infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream) {
        assert(!uninitialized);
        assert(inputTensor.isPresent());
        assert(outputTensor.isPresent());

        ScopedGpu scopedGpu(inputTensor.get().getPlacement().getDeviceNum());
        launchMap<INDEX_TYPE>((half *)outputTensor.get().getMemPtr(),
                              (half *)inputTensor.get().getMemPtr(),
                              (INDEX_TYPE *)mappingOfSourceTensorIntoDestTensor.getMemPtr(),
                              (INDEX_TYPE)mappingOfSourceTensorIntoDestTensor.getDescriptor().getTotalNumElements(),
                              stream);
    }

    virtual void backProp(Optional<Tensor> dataIn, Optional<Tensor> errorIn, Optional<Tensor> errorOut, Stream stream) {
        assert(!uninitialized);
        if (errorOut.isEmpty())
            return;
        assert(errorIn.isPresent());

        ScopedGpu scopedGpu(errorIn.get().getPlacement().getDeviceNum());

        for (auto it = backwardPassMappingOfNTo1Device.begin(); it != backwardPassMappingOfNTo1Device.end(); ++it) {
            int N = it->first;
            Tensor nMappingTensor = it->second;
            assert(nMappingTensor.getDescriptor().getTotalNumElements() % (N + 1) == 0);
            launchMapNInto1<INDEX_TYPE>(N,
                                        (half *)errorOut.get().getMemPtr(),
                                        (half *)errorIn.get().getMemPtr(),
                                        (INDEX_TYPE *)nMappingTensor.getMemPtr(),
                                        (INDEX_TYPE)nMappingTensor.getDescriptor().getTotalNumElements() / (N + 1),
                                        stream);
        }
    }

    std::map<unsigned int, Tensor> getBackwardPassMappingOnHost() {
        assert(!uninitialized);
        return backwardPassMappingOfNTo1Host;
    }

    std::map<unsigned int, Tensor> getBackwardPassMappingOnDevice() {
        assert(!uninitialized);
        return backwardPassMappingOfNTo1Device;
    }

   private:
    bool uninitialized;

    Tensor mappingOfSourceTensorIntoDestTensor;
    std::map<unsigned int, Tensor> backwardPassMappingOfNTo1Host;
    std::map<unsigned int, Tensor> backwardPassMappingOfNTo1Device;

    std::vector<unsigned long> sourceDimensions;

    void setup(Tensor mappingOfSourceTensorIntoDestTensor, std::vector<unsigned long> sourceDimensions) {
        assert(std::is_integral<INDEX_TYPE>());
        assert(!std::is_signed<INDEX_TYPE>());
        if (sizeof(INDEX_TYPE) < sizeof(uint64_t)) {
            uint64_t numSourceElements = TensorDescriptor(TensorDescriptor::DataType::BOOLEAN, sourceDimensions).getTotalNumElements();
            uint64_t numDestElements = mappingOfSourceTensorIntoDestTensor.getDescriptor().getTotalNumElements();
            uint64_t maxElementsForIndexType = 1l << (8 * sizeof(INDEX_TYPE));
            assert(numSourceElements < maxElementsForIndexType);
            assert(numDestElements < maxElementsForIndexType);
        }

        uninitialized = false;
        this->mappingOfSourceTensorIntoDestTensor = mappingOfSourceTensorIntoDestTensor;
        this->sourceDimensions = sourceDimensions;
    }

    // To reverse the mapping, in general, the number of errorInputs that map to each errorOutput is noted,
    // then all errorInputs that map to a given errorOutput are summed and passed as the value of the errorOutput.
    // When there are 0 errorInputs that map to a given errorOutput, a 0 is back propagated as the value of that errorOutput.
    void createBackwardPassMapping(Tensor mappingOfSourceTensorIntoDestTensor) {
        assert(!uninitialized);

        Stream stream(0);

        Tensor cpuMappingTensor = mappingOfSourceTensorIntoDestTensor.clone(TensorPlacement::MemDevices::CPU);
        cpuMappingTensor.copyFromAsync(mappingOfSourceTensorIntoDestTensor, stream);
        stream.synchronize();
        INDEX_TYPE *cpuMappingTensorMem = (INDEX_TYPE *)cpuMappingTensor.getMemPtr();

        // For each input feature element, get the list of output feature elements it maps to.
        //  featureInput    {featureOutputs...}     (dense over outputs)
        std::map<INDEX_TYPE, std::priority_queue<INDEX_TYPE, std::vector<INDEX_TYPE>, std::greater<INDEX_TYPE>>>
            outputDestinationsOfInputElement;
        INDEX_TYPE numFeatureOutputElements = mappingOfSourceTensorIntoDestTensor.getDescriptor().getTotalNumElements();
        for (INDEX_TYPE outputElementFlatIndex = 0; outputElementFlatIndex < numFeatureOutputElements; ++outputElementFlatIndex) {
            INDEX_TYPE inputElementFlatIndex = cpuMappingTensorMem[outputElementFlatIndex];
            outputDestinationsOfInputElement[inputElementFlatIndex].push(outputElementFlatIndex);
        }

        // For each output error element, describe the N to 1 summation from the corresponding error input elements.
        // N=0 is used where necessary to create a dense mapping of errorInputs to errorOutputs
        //                N             featureInput    {featureOutputs...}
        std::map<unsigned int, std::map<INDEX_TYPE, std::priority_queue<INDEX_TYPE, std::vector<INDEX_TYPE>, std::greater<INDEX_TYPE>>>>
            nGroupingsOfOutputDestinationsOfInputElement;

        TensorDescriptor elementPopulatedTensorDescriptor(TensorDescriptor::DataType::BOOLEAN, sourceDimensions);
        Tensor elementPopulatedTensor(TensorPlacement::MemDevices::CPU, elementPopulatedTensorDescriptor);
        bool *elementPopulated = (bool *)elementPopulatedTensor.getMemPtr();
        INDEX_TYPE numErrorOutputElements = elementPopulatedTensorDescriptor.getTotalNumElements();
        memset(elementPopulated, false, elementPopulatedTensorDescriptor.getTotalNumElements());
        for (auto it = outputDestinationsOfInputElement.begin(); it != outputDestinationsOfInputElement.end(); ++it) {
            unsigned int N = it->second.size();
            nGroupingsOfOutputDestinationsOfInputElement[N][it->first] = it->second;
            elementPopulated[it->first] = true;
        }
        for (INDEX_TYPE errorOutputFlatIndex = 0; errorOutputFlatIndex < numErrorOutputElements; ++errorOutputFlatIndex) {
            if (!elementPopulated[errorOutputFlatIndex])
                nGroupingsOfOutputDestinationsOfInputElement[0][errorOutputFlatIndex] =
                    std::priority_queue<INDEX_TYPE, std::vector<INDEX_TYPE>, std::greater<INDEX_TYPE>>();
        }

        // Now I have all summations grouped by N (the number of error outputs that map to the error input)
        // I will launch 1 kernel for each value of N to do the necessary summations, to do this I will create N flat maps.
        for (auto it = nGroupingsOfOutputDestinationsOfInputElement.begin(); it != nGroupingsOfOutputDestinationsOfInputElement.end();
             ++it) {
            unsigned int N = it->first;
            std::map<INDEX_TYPE, std::priority_queue<INDEX_TYPE, std::vector<INDEX_TYPE>, std::greater<INDEX_TYPE>>> &nMapping = it->second;
            INDEX_TYPE mappingsOfSizeN = nMapping.size();
            std::vector<unsigned long> flatDimension;
            // { errorOutputIndex1, {errorInputIndex1, ..., errorInputIndexN},
            //   errorOutputIndex2, {errorInputIndex1, ..., errorInputIndexN} }
            // ...
            //   errorOutputIndex_mappingsOfSizeN, {errorInputIndex1, ..., errorInputIndexN} }
            flatDimension.push_back((N + 1) * mappingsOfSizeN);

            Tensor backwardMappingTensor =
                Tensor(TensorPlacement::MemDevices::CPU, TensorDescriptor(TensorDescriptor::DataType::UINT64, flatDimension));
            INDEX_TYPE *backwardMappingMem = (INDEX_TYPE *)backwardMappingTensor.getMemPtr();

            INDEX_TYPE i = 0;
            for (auto mappingIt = nMapping.begin(); mappingIt != nMapping.end(); ++mappingIt) {
                INDEX_TYPE destinationErrorOutput = mappingIt->first;
                std::priority_queue<INDEX_TYPE, std::vector<INDEX_TYPE>, std::greater<INDEX_TYPE>> &mappedErrorInputs = mappingIt->second;

                backwardMappingMem[i] = destinationErrorOutput;
                i += 1;
                while (!mappedErrorInputs.empty()) {
                    backwardMappingMem[i] = mappedErrorInputs.top();
                    mappedErrorInputs.pop();
                    i += 1;
                }
            }

            backwardPassMappingOfNTo1Host[N] = backwardMappingTensor;
        }
        // Now I have a mapping constructed for every N grouping that exists in the mapping, these will be directly consumed, 1 per kernel
        // launch.
    }
};

}  // namespace ThorImplementation
