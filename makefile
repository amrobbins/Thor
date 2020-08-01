CUDA_INCLUDE_DIRS = -I /usr/local/cuda/include -I /usr/include
CUDA_LIBRARIES = -L /usr/local/cuda/lib64 -l cublas -l cublasLt -l cusolver -l cudart -L /usr/lib/x86_64-linux-gnu -l cudnn
CUDA = $(CUDA_INCLUDE_DIRS) $(CUDA_LIBRARIES)
COMPUTE_CAPABILITIES_MOBILE_DEVICES = -gencode=arch=compute_53,code=compute_53 -gencode=arch=compute_53,code=sm_53 \
                                      -gencode=arch=compute_62,code=compute_62 -gencode=arch=compute_62,code=sm_62 \
                                      -gencode=arch=compute_72,code=compute_72 -gencode=arch=compute_72,code=sm_72
#COMPUTE_CAPABILITIES = -gencode=arch=compute_52,code=compute_52 -gencode=arch=compute_52,code=sm_52 \
#                       -gencode=arch=compute_60,code=compute_60 -gencode=arch=compute_60,code=sm_60 \
#                       -gencode=arch=compute_61,code=compute_61 -gencode=arch=compute_61,code=sm_61 \
#                       -gencode=arch=compute_70,code=compute_70 -gencode=arch=compute_70,code=sm_70 \
#                       -gencode=arch=compute_75,code=compute_75 -gencode=arch=compute_75,code=sm_75
COMPUTE_CAPABILITIES = -gencode=arch=compute_52,code=compute_52 -gencode=arch=compute_52,code=sm_52 \
                       -gencode=arch=compute_75,code=compute_75 -gencode=arch=compute_75,code=sm_75


#COMPUTE_CAPABILITIES_WITH_TENSOR_CORES = -gencode=arch=compute_70,code=compute_70 -gencode=arch=compute_70,code=sm_70 \
#                       -gencode=arch=compute_75,code=compute_75 -gencode=arch=compute_75,code=sm_75
COMPUTE_CAPABILITIES_WITH_TENSOR_CORES = -gencode=arch=compute_75,code=compute_75 -gencode=arch=compute_75,code=sm_75


BOOST_INCLUDE_DIR = -I /usr/local/boost

MLDEV_LIBS = $(CUDA) -I./ -L./ -lMLDev

INCLUDE_HOME_DIR = -I ./
INCLUDE_DIRS = $(INCLUDE_HOME_DIR) $(CUDA_INCLUDE_DIRS) $(BOOST_INCLUDE_DIR)

INCLUDE_DIRS_TEST = $(INCLUDE_DIRS) -I build/test/googletest/include
LIB_DIRS_TEST = -L build/test/googletest
LIBS_TEST = -lgtest -lgtest_main -pthread
TEST_COMPILE_DEPENDENCIES = $(INCLUDE_DIRS_TEST) $(LIB_DIRS_TEST) $(LIBS_TEST) $(CUDA)

DEBUG = -ggdb
NVCC_DEBUG = -g

Gpp = g++ -Wall -Werror
Nvcc = nvcc

RUN_ALL_TESTS = build/test/DeepLearning/Implementation/Layers/NeuralNetwork/Convolution2dTest && \
                build/test/DeepLearning/Implementation/Layers/Loss/CategoricalCrossEntropyLossTest && \
                build/test/Utilities/TensorOperations/DeepLearning/CrossEntropyLossTest && \
                build/test/Utilities/TensorOperations/Arithmetic/ArithmeticTest && \
                build/test/Utilities/Common/OptionalTest && \
                build/test/DeepLearning/Implementation/Layers/NeuralNetwork/DropOutTest && \
                build/test/Utilities/TensorOperations/Misc/MiscTest && \
                build/test/DeepLearning/Implementation/Layers/Utility/UtilityLayerTest && \
                build/test/DeepLearning/Implementation/Layers/Activations/ActivationsLayerTest && \
                build/test/Utilities/TensorOperations/TypeConversions/TypeConverterTest && \
                build/test/DeepLearning/Implementation/Tensor/tensorTest && \
                build/test/Utilities/ComputeTopology/machineEvaluatorTest && \
                build/test/Utilities/TensorOperations/GpuMatrixMultiply/gpuMatrixMultiplyTest && \
                build/test/Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiplyTest && \
                build/test/Utilities/TensorOperations/GpuConvolution/GpuConvolutionTest && \
                build/test/Utilities/WorkQueue/WorkQueueUnorderedTest && \
                build/test/Utilities/WorkQueue/WorkQueueTest && \
                build/test/Utilities/TensorOperations/GpuMatrixTranspose/gpuMatrixTransposeTest \

ALL_TESTS = build/test/DeepLearning/Implementation/Layers/Loss/CategoricalCrossEntropyLossTest \
            build/test/Utilities/TensorOperations/DeepLearning/CrossEntropyLossTest \
            build/test/Utilities/TensorOperations/Arithmetic/ArithmeticTest \
            build/test/Utilities/TensorOperations/Misc/MiscTest \
            build/test/DeepLearning/Implementation/Layers/Utility/UtilityLayerTest \
            build/test/DeepLearning/Implementation/Layers/NeuralNetwork/DropOutTest \
            build/test/DeepLearning/Implementation/Layers/Activations/ActivationsLayerTest \
            build/test/Utilities/TensorOperations/TypeConversions/TypeConverterTest \
            build/test/DeepLearning/Implementation/Tensor/tensorTest \
            build/test/Utilities/WorkQueue/WorkQueueUnorderedTest \
            build/test/Utilities/WorkQueue/WorkQueueTest \
            build/test/Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiplyTest \
            build/test/Utilities/TensorOperations/GpuConvolution/GpuConvolutionTest \
            build/test/DeepLearning/Implementation/Layers/NeuralNetwork/Convolution2dTest \
            build/test/Utilities/TensorOperations/GpuMatrixTranspose/gpuMatrixTransposeTest \
            build/test/Utilities/TensorOperations/GpuMatrixMultiply/gpuMatrixMultiplyTest \
            build/test/Utilities/ComputeTopology/machineEvaluatorTest \
            build/test/Utilities/Common/OptionalTest \

ALL_OBJECT_FILES = build/Utilities/TensorOperations/GpuMatrixTranspose/gpuMatrixTransposeKernels.o \
                   build/Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiply.o \
                   build/Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiplyBatch8Reg64.o \
                   build/Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiplyBatch16Reg64.o \
                   build/Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiplyBatch32.o \
                   build/Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiplyBatch48.o \
                   build/Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiplyBatch64.o \
                   build/Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiplyBatch80.o \
                   build/Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiplyBatch96.o \
                   build/Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiplyBatch112.o \
                   build/Utilities/TensorOperations/GpuMatrixMultiply/Reductions.o \
                   build/Utilities/TensorOperations/Arithmetic/Average.o \
                   build/Utilities/TensorOperations/Arithmetic/Tanh.o \
                   build/Utilities/TensorOperations/Arithmetic/Relu.o \
                   build/Utilities/TensorOperations/Arithmetic/Sum.o \
                   build/Utilities/TensorOperations/Arithmetic/SumScale.o \
                   build/Utilities/TensorOperations/Arithmetic/SumManyToOne.o \
                   build/Utilities/TensorOperations/Arithmetic/Exponentiation.o \
                   build/Utilities/TensorOperations/Arithmetic/ElementwiseSubtract.o \
                   build/Utilities/TensorOperations/Arithmetic/MultiplyByScalar.o \
                   build/Utilities/TensorOperations/DeepLearning/CrossEntropyLoss.o \
                   build/Utilities/TensorOperations/Misc/Map.o \
                   build/Utilities/TensorOperations/Misc/Split.o \
                   build/Utilities/TensorOperations/Misc/Pad.o \
                   build/Utilities/TensorOperations/Misc/Extract.o \
                   build/Utilities/TensorOperations/Misc/Concatenate.o \
                   build/Utilities/TensorOperations/GpuMatrixMultiply/gpuMatrixMultiply.o \
                   build/Utilities/TensorOperations/GpuConvolution/GpuConvolution.o \
                   build/Utilities/TensorOperations/GpuConvolution/GpuConvolutionKernels.o \
                   build/Utilities/TensorOperations/GpuMatrixMultiply/CublasKernel.o \
                   build/Utilities/TensorOperations/GpuMatrixMultiply/CublasMatrixMultiply.o \
                   build/Utilities/ComputeTopology/MachineEvaluator.o \
                   build/DeepLearning/Implementation/Tensor/Tensor.o \
                   build/DeepLearning/Implementation/Tensor/DistributedTensor.o \
                   build/DeepLearning/Implementation/Layers/Layer.o \
                   build/Utilities/TensorOperations/TypeConversions/TypeConverter.o \
                   build/Utilities/TensorOperations/TypeConversions/TypeConverterKernels.o \

ML_DEV = libMLDev.a MLDev.h




# Overall make targets

all: $(ML_DEV)
	$(MAKE) $(ALL_TESTS)
	$(RUN_ALL_TESTS)
	@echo ""
	@echo ""
	@echo "Tests Passed"
	@echo "Build Succeeded"
	@echo ""

clean:
	rm -rf build
	rm -f libMLDev.a
	rm -f MLDev.h
	cd googletest && cmake ./
	$(MAKE) clean -C googletest


softclean:
	rm -rf build
	rm -f libMLDev.a
	rm -f MLDev.h


# Library

libMLDev.a: MLDev.h
	git config core.hooksPath .githooks
	$(MAKE) $(ALL_OBJECT_FILES)
	ar rcs libMLDev.a $(ALL_OBJECT_FILES)

build/headerlist.txt: libMLDev.a
	mkdir -p build
	find ./ -name "*.h" | grep -v ./googletest/ | grep -v ./test/ > build/headerlist.txt

build/buildTools/createMasterHeader: buildTools/createMasterHeader.cpp
	mkdir -p build
	mkdir -p build/buildTools
	$(Gpp) -o build/buildTools/createMasterHeader -O3 -std=c++11 buildTools/createMasterHeader.cpp

MLDev.h: build/headerlist.txt build/buildTools/createMasterHeader
	build/buildTools/createMasterHeader build/headerlist.txt


# Object Files

build/Utilities/TensorOperations/GpuMatrixTranspose/gpuMatrixTransposeKernels.o: Utilities/TensorOperations/GpuMatrixTranspose/gpuMatrixTranspose.h Utilities/TensorOperations/GpuMatrixTranspose/gpuMatrixTransposeKernels.cu
	mkdir -p build/Utilities/TensorOperations/GpuMatrixTranspose
	$(Nvcc) -O3 -ccbin g++ -o build/Utilities/TensorOperations/GpuMatrixTranspose/gpuMatrixTransposeKernels.o -c --cudart static -std=c++11 $(COMPUTE_CAPABILITIES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/GpuMatrixTranspose/gpuMatrixTransposeKernels.cu

build/Utilities/TensorOperations/TypeConversions/TypeConverterKernels.o: Utilities/TensorOperations/TypeConversions/TypeConverter.h Utilities/TensorOperations/TypeConversions/TypeConverterKernels.cu
	mkdir -p build/Utilities/TensorOperations/TypeConversions
	$(Nvcc) -O3 -ccbin g++ -o build/Utilities/TensorOperations/TypeConversions/TypeConverterKernels.o -c --cudart static -std=c++11 $(COMPUTE_CAPABILITIES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/TypeConversions/TypeConverterKernels.cu

build/Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiplyBatch16Reg64.o: Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiply.h Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiplyBatch16Reg64.cu
	mkdir -p build/Utilities/TensorOperations/GpuMatrixMultiply
	$(Nvcc) -O3 -ccbin g++ -o build/Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiplyBatch16Reg64.o -c --maxrregcount 64 --cudart static -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiplyBatch16Reg64.cu

build/Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiplyBatch8Reg64.o: Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiply.h Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiplyBatch8Reg64.cu
	mkdir -p build/Utilities/TensorOperations/GpuMatrixMultiply
	$(Nvcc) -O3 -ccbin g++ -o build/Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiplyBatch8Reg64.o -c --maxrregcount 64 --cudart static -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiplyBatch8Reg64.cu

build/Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiplyBatch32.o: Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiply.h Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiplyBatch32.cu
	mkdir -p build/Utilities/TensorOperations/GpuMatrixMultiply
	$(Nvcc) -O3 -ccbin g++ -o build/Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiplyBatch32.o -c --maxrregcount 128 --cudart static -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiplyBatch32.cu

build/Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiplyBatch48.o: Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiply.h Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiplyBatch48.cu
	mkdir -p build/Utilities/TensorOperations/GpuMatrixMultiply
	$(Nvcc) -O3 -ccbin g++ -o build/Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiplyBatch48.o -c --maxrregcount 128 --cudart static -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiplyBatch48.cu

build/Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiplyBatch64.o: Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiply.h Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiplyBatch64.cu
	mkdir -p build/Utilities/TensorOperations/GpuMatrixMultiply
	$(Nvcc) -O3 -ccbin g++ -o build/Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiplyBatch64.o -c --maxrregcount 128 --cudart static -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiplyBatch64.cu

build/Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiplyBatch80.o: Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiply.h Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiplyBatch80.cu
	mkdir -p build/Utilities/TensorOperations/GpuMatrixMultiply
	$(Nvcc) -O3 -ccbin g++ -o build/Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiplyBatch80.o -c --maxrregcount 128 --cudart static -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiplyBatch80.cu

build/Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiplyBatch96.o: Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiply.h Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiplyBatch96.cu
	mkdir -p build/Utilities/TensorOperations/GpuMatrixMultiply
	$(Nvcc) -O3 -ccbin g++ -o build/Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiplyBatch96.o -c --maxrregcount 128 --cudart static -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiplyBatch96.cu

build/Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiplyBatch112.o: Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiply.h Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiplyBatch112.cu
	mkdir -p build/Utilities/TensorOperations/GpuMatrixMultiply
	$(Nvcc) -O3 -ccbin g++ -o build/Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiplyBatch112.o -c --maxrregcount 128 --cudart static -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiplyBatch112.cu

build/Utilities/TensorOperations/GpuMatrixMultiply/Reductions.o: Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiply.h Utilities/TensorOperations/GpuMatrixMultiply/Reductions.cu
	mkdir -p build/Utilities/TensorOperations/GpuMatrixMultiply
	$(Nvcc) -O3 -ccbin g++ -o build/Utilities/TensorOperations/GpuMatrixMultiply/Reductions.o -c --maxrregcount 128 --cudart static -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/GpuMatrixMultiply/Reductions.cu

build/Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiply.o: Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiply.h Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiply.cu build/Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiplyBatch8Reg64.o build/Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiplyBatch16Reg64.o build/Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiplyBatch32.o build/Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiplyBatch48.o build/Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiplyBatch64.o build/Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiplyBatch80.o build/Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiplyBatch96.o build/Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiplyBatch112.o build/Utilities/TensorOperations/GpuMatrixMultiply/Reductions.o
	mkdir -p build/Utilities/TensorOperations/GpuMatrixMultiply
	$(Nvcc) -O3 -ccbin g++ -o build/Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiply.o -c --maxrregcount 128 --cudart static -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiply.cu

build/Utilities/TensorOperations/Arithmetic/Average.o: Utilities/TensorOperations/Arithmetic/Average.h Utilities/TensorOperations/Arithmetic/Average.cu
	mkdir -p build/Utilities/TensorOperations/Arithmetic
	$(Nvcc) -O3 -ccbin g++ -o build/Utilities/TensorOperations/Arithmetic/Average.o -c --maxrregcount 128 --cudart static -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/Arithmetic/Average.cu

build/Utilities/TensorOperations/Arithmetic/Tanh.o: Utilities/TensorOperations/Arithmetic/Tanh.h Utilities/TensorOperations/Arithmetic/Tanh.cu
	mkdir -p build/Utilities/TensorOperations/Arithmetic
	$(Nvcc) -O3 -ccbin g++ -o build/Utilities/TensorOperations/Arithmetic/Tanh.o -c --maxrregcount 128 --cudart static -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/Arithmetic/Tanh.cu

build/Utilities/TensorOperations/Arithmetic/Relu.o: Utilities/TensorOperations/Arithmetic/Relu.h Utilities/TensorOperations/Arithmetic/Relu.cu
	mkdir -p build/Utilities/TensorOperations/Arithmetic
	$(Nvcc) -O3 -ccbin g++ -o build/Utilities/TensorOperations/Arithmetic/Relu.o -c --maxrregcount 128 --cudart static -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/Arithmetic/Relu.cu

build/Utilities/TensorOperations/Arithmetic/Sum.o: Utilities/TensorOperations/Arithmetic/Sum.h Utilities/TensorOperations/Arithmetic/Sum.cu
	mkdir -p build/Utilities/TensorOperations/Arithmetic
	$(Nvcc) -O3 -ccbin g++ -o build/Utilities/TensorOperations/Arithmetic/Sum.o -c --maxrregcount 128 --cudart static -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/Arithmetic/Sum.cu

build/Utilities/TensorOperations/Arithmetic/SumScale.o: Utilities/TensorOperations/Arithmetic/SumScale.h Utilities/TensorOperations/Arithmetic/SumScale.cu
	mkdir -p build/Utilities/TensorOperations/Arithmetic
	$(Nvcc) -O3 -ccbin g++ -o build/Utilities/TensorOperations/Arithmetic/SumScale.o -c --maxrregcount 128 --cudart static -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/Arithmetic/SumScale.cu

build/Utilities/TensorOperations/Arithmetic/SumManyToOne.o: Utilities/TensorOperations/Arithmetic/SumManyToOne.h Utilities/TensorOperations/Arithmetic/SumManyToOne.cu
	mkdir -p build/Utilities/TensorOperations/Arithmetic
	$(Nvcc) -O3 -ccbin g++ -o build/Utilities/TensorOperations/Arithmetic/SumManyToOne.o -c --maxrregcount 128 --cudart static -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/Arithmetic/SumManyToOne.cu

build/Utilities/TensorOperations/Arithmetic/ElementwiseSubtract.o: Utilities/TensorOperations/Arithmetic/ElementwiseSubtract.h Utilities/TensorOperations/Arithmetic/ElementwiseSubtract.cu
	mkdir -p build/Utilities/TensorOperations/Arithmetic
	$(Nvcc) -O3 -ccbin g++ -o build/Utilities/TensorOperations/Arithmetic/ElementwiseSubtract.o -c --maxrregcount 128 --cudart static -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/Arithmetic/ElementwiseSubtract.cu

build/Utilities/TensorOperations/Arithmetic/MultiplyByScalar.o: Utilities/TensorOperations/Arithmetic/MultiplyByScalar.h Utilities/TensorOperations/Arithmetic/MultiplyByScalar.cu
	mkdir -p build/Utilities/TensorOperations/Arithmetic
	$(Nvcc) -O3 -ccbin g++ -o build/Utilities/TensorOperations/Arithmetic/MultiplyByScalar.o -c --maxrregcount 128 --cudart static -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/Arithmetic/MultiplyByScalar.cu

build/Utilities/TensorOperations/DeepLearning/CrossEntropyLoss.o: Utilities/TensorOperations/DeepLearning/CrossEntropyLoss.h Utilities/TensorOperations/DeepLearning/CrossEntropyLoss.cu
	mkdir -p build/Utilities/TensorOperations/DeepLearning
	$(Nvcc) -O3 -ccbin g++ -o build/Utilities/TensorOperations/DeepLearning/CrossEntropyLoss.o -c --maxrregcount 128 --cudart static -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/DeepLearning/CrossEntropyLoss.cu

build/Utilities/TensorOperations/Arithmetic/Exponentiation.o: Utilities/TensorOperations/Arithmetic/Exponentiation.h Utilities/TensorOperations/Arithmetic/Exponentiation.cu
	mkdir -p build/Utilities/TensorOperations/Arithmetic
	$(Nvcc) -O3 -ccbin g++ -o build/Utilities/TensorOperations/Arithmetic/Exponentiation.o -c --maxrregcount 128 --cudart static -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/Arithmetic/Exponentiation.cu

build/Utilities/TensorOperations/Misc/Map.o: Utilities/TensorOperations/Misc/Map.h Utilities/TensorOperations/Misc/Map.cu
	mkdir -p build/Utilities/TensorOperations/Misc
	$(Nvcc) -O3 -ccbin g++ -o build/Utilities/TensorOperations/Misc/Map.o -c --maxrregcount 64 --cudart static -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/Misc/Map.cu

build/Utilities/TensorOperations/Misc/Split.o: Utilities/TensorOperations/Misc/Split.h Utilities/TensorOperations/Misc/Split.cu
	mkdir -p build/Utilities/TensorOperations/Misc
	$(Nvcc) -O3 -ccbin g++ -o build/Utilities/TensorOperations/Misc/Split.o -c --maxrregcount 64 --cudart static -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/Misc/Split.cu

build/Utilities/TensorOperations/Misc/Pad.o: Utilities/TensorOperations/Misc/Pad.h Utilities/TensorOperations/Misc/Pad.cu
	mkdir -p build/Utilities/TensorOperations/Misc
	$(Nvcc) -O3 -ccbin g++ -o build/Utilities/TensorOperations/Misc/Pad.o -c --maxrregcount 64 --cudart static -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/Misc/Pad.cu

build/Utilities/TensorOperations/Misc/Extract.o: Utilities/TensorOperations/Misc/Extract.h Utilities/TensorOperations/Misc/Extract.cu
	mkdir -p build/Utilities/TensorOperations/Misc
	$(Nvcc) -O3 -ccbin g++ -o build/Utilities/TensorOperations/Misc/Extract.o -c --maxrregcount 64 --cudart static -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/Misc/Extract.cu

build/Utilities/TensorOperations/Misc/Concatenate.o: Utilities/TensorOperations/Misc/Concatenate.h Utilities/TensorOperations/Misc/Concatenate.cu
	mkdir -p build/Utilities/TensorOperations/Misc
	$(Nvcc) -O3 -ccbin g++ -o build/Utilities/TensorOperations/Misc/Concatenate.o -c --maxrregcount 64 --cudart static -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/Misc/Concatenate.cu

build/Utilities/TensorOperations/TypeConversions/TypeConverter.o: Utilities/TensorOperations/TypeConversions/TypeConverter.h Utilities/TensorOperations/TypeConversions/TypeConverter.cpp
	mkdir -p build/Utilities/TensorOperations/TypeConversions
	$(Gpp) -c -O3 -std=c++11 Utilities/TensorOperations/TypeConversions/TypeConverter.cpp $(CUDA) $(INCLUDE_DIRS) -o build/Utilities/TensorOperations/TypeConversions/TypeConverter.o

build/Utilities/TensorOperations/GpuMatrixMultiply/gpuMatrixMultiply.o: Utilities/TensorOperations/GpuMatrixMultiply/gpuMatrixMultiply.h Utilities/TensorOperations/GpuMatrixMultiply/gpuMatrixMultiply.cpp
	mkdir -p build/Utilities/TensorOperations/GpuMatrixMultiply
	$(Gpp) -c -O3 -std=c++11 Utilities/TensorOperations/GpuMatrixMultiply/gpuMatrixMultiply.cpp $(CUDA) $(INCLUDE_DIRS) -o build/Utilities/TensorOperations/GpuMatrixMultiply/gpuMatrixMultiply.o

build/Utilities/ComputeTopology/MachineEvaluator.o: Utilities/ComputeTopology/MachineEvaluator.h Utilities/ComputeTopology/MachineEvaluator.cpp
	mkdir -p build/Utilities/ComputeTopology
	$(Gpp) -c -O3 -std=c++11 Utilities/ComputeTopology/MachineEvaluator.cpp $(CUDA) $(INCLUDE_DIRS) -o build/Utilities/ComputeTopology/MachineEvaluator.o

build/DeepLearning/Implementation/Tensor/Tensor.o: DeepLearning/Implementation/Tensor/Tensor.h DeepLearning/Implementation/Tensor/DistributedTensor.h DeepLearning/Implementation/Tensor/Tensor.cpp DeepLearning/Implementation/Tensor/TensorDescriptor.h DeepLearning/Implementation/Tensor/TensorPlacement.h
	mkdir -p build/DeepLearning/Implementation/Tensor
	$(Gpp) -c -O3 -I./ -std=c++11 DeepLearning/Implementation/Tensor/Tensor.cpp $(CUDA) $(INCLUDE_DIRS) -o build/DeepLearning/Implementation/Tensor/Tensor.o

build/DeepLearning/Implementation/Tensor/DistributedTensor.o: DeepLearning/Implementation/Tensor/Tensor.h DeepLearning/Implementation/Tensor/DistributedTensor.h DeepLearning/Implementation/Tensor/DistributedTensor.cpp DeepLearning/Implementation/Tensor/TensorDescriptor.h DeepLearning/Implementation/Tensor/TensorPlacement.h
	mkdir -p build/DeepLearning/Implementation/Tensor
	$(Gpp) -c -O3 -I./ -std=c++11 DeepLearning/Implementation/Tensor/DistributedTensor.cpp $(CUDA) $(INCLUDE_DIRS) -o build/DeepLearning/Implementation/Tensor/DistributedTensor.o

build/DeepLearning/Implementation/Layers/Layer.o:  DeepLearning/Implementation/Layers/Layer.h DeepLearning/Implementation/Layers/NeuralNetwork/DropOut.h
	mkdir -p build/DeepLearning/Implementation/Layers/Layer
	$(Gpp) -c -O3 -I./ -std=c++11 DeepLearning/Implementation/Layers/Layer.cpp $(CUDA) $(INCLUDE_DIRS) -o build/DeepLearning/Implementation/Layers/Layer.o

build/Utilities/TensorOperations/GpuConvolution/GpuConvolution.o: Utilities/TensorOperations/GpuConvolution/GpuConvolution.h Utilities/TensorOperations/GpuConvolution/GpuConvolution.cpp
	mkdir -p build/Utilities/TensorOperations/GpuConvolution
	$(Gpp) -c -O3 -std=c++11 Utilities/TensorOperations/GpuConvolution/GpuConvolution.cpp $(CUDA) $(INCLUDE_DIRS) -o build/Utilities/TensorOperations/GpuConvolution/GpuConvolution.o

build/Utilities/TensorOperations/GpuConvolution/GpuConvolutionKernels.o: Utilities/TensorOperations/GpuConvolution/GpuConvolution.h Utilities/TensorOperations/GpuConvolution/GpuConvolutionKernels.cu
	mkdir -p build/Utilities/TensorOperations/GpuConvolution
	$(Nvcc) -O3 -ccbin g++ -o build/Utilities/TensorOperations/GpuConvolution/GpuConvolutionKernels.o -c --maxrregcount 64 --cudart static -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/GpuConvolution/GpuConvolutionKernels.cu

build/Utilities/TensorOperations/GpuMatrixMultiply/CublasKernel.o: Utilities/TensorOperations/GpuMatrixMultiply/CublasKernel.h Utilities/TensorOperations/GpuMatrixMultiply/CublasKernel.cpp
	mkdir -p build/Utilities/TensorOperations/GpuMatrixMultiply
	$(Gpp) -c -O3 -std=c++11 Utilities/TensorOperations/GpuMatrixMultiply/CublasKernel.cpp $(CUDA) $(INCLUDE_DIRS) -o build/Utilities/TensorOperations/GpuMatrixMultiply/CublasKernel.o

build/Utilities/TensorOperations/GpuMatrixMultiply/CublasMatrixMultiply.o: Utilities/TensorOperations/GpuMatrixMultiply/CublasMatrixMultiply.h Utilities/TensorOperations/GpuMatrixMultiply/CublasMatrixMultiply.cpp
	mkdir -p build/Utilities/TensorOperations/GpuMatrixMultiply
	$(Gpp) -c -O3 -std=c++11 Utilities/TensorOperations/GpuMatrixMultiply/CublasMatrixMultiply.cpp $(CUDA) $(INCLUDE_DIRS) -o build/Utilities/TensorOperations/GpuMatrixMultiply/CublasMatrixMultiply.o



# Test Framework


#FIXME: put back
# && $(MAKE) test
build/test/googletest/libgtest.a: ./googletest/README.md
	rm -rf build/test/googletest
	mkdir -p build
	mkdir -p build/test
	mkdir -p build/test/googletest
	cd googletest && cmake -Dgtest_build_samples=ON -Dgtest_build_tests=ON ./ && $(MAKE)
	cd build/test/googletest && ln -s ../../../googletest/lib/libgtest.a ./ && ln -s ../../../googletest/lib/libgtest_main.a ./ && ln -s ../../../googletest/lib/libgmock.a ./ && ln -s ../../../googletest/lib/libgmock_main.a ./ \
        && mkdir -p include && cd include && ln -s ../../../../googletest/googletest/include/gtest ./ && ln -s ../../../../googletest/googlemock/include/gmock ./


# Tests

build/test/Utilities/WorkQueue/WorkQueueUnorderedTest: build/test/googletest/libgtest.a test/Utilities/WorkQueue/WorkQueueUnorderedTest.cpp $(MLDEV)
	mkdir -p build/test/Utilities/WorkQueue/
	$(Gpp) -ggdb -o build/test/Utilities/WorkQueue/WorkQueueUnorderedTest -O3 -std=c++11 -pthread -IUtilities/WorkQueue/ test/Utilities/WorkQueue/WorkQueueUnorderedTest.cpp $(MLDEV_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/Utilities/WorkQueue/WorkQueueTest: build/test/googletest/libgtest.a test/Utilities/WorkQueue/WorkQueueTest.cpp $(MLDEV)
	mkdir -p build/test/Utilities/WorkQueue/
	$(Gpp) -ggdb -o build/test/Utilities/WorkQueue/WorkQueueTest -O3 -std=c++11 -pthread -IUtilities/WorkQueue/ test/Utilities/WorkQueue/WorkQueueTest.cpp $(MLDEV_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/Utilities/TensorOperations/GpuMatrixTranspose/gpuMatrixTransposeTest: build/test/googletest/libgtest.a test/Utilities/TensorOperations/GpuMatrixTranspose/gpuMatrixTransposeTest.cpp $(MLDEV)
	mkdir -p build/test/Utilities/TensorOperations/GpuMatrixTranspose
	$(Gpp) -fopenmp -o build/test/Utilities/TensorOperations/GpuMatrixTranspose/gpuMatrixTransposeTest test/Utilities/TensorOperations/GpuMatrixTranspose/gpuMatrixTransposeTest.cpp -O3 -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(MLDEV_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiplyTest: build/test/googletest/libgtest.a test/Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiplyTest.cpp $(MLDEV)
	mkdir -p build/test/Utilities/TensorOperations/GpuMatrixMultiply
	$(Gpp) -fopenmp -o build/test/Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiplyTest test/Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiplyTest.cpp -O3 -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(MLDEV_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/Utilities/TensorOperations/GpuConvolution/GpuConvolutionTest: build/test/googletest/libgtest.a test/Utilities/TensorOperations/GpuConvolution/GpuConvolutionTest.cpp $(MLDEV)
	mkdir -p build/test/Utilities/TensorOperations/GpuConvolution
	$(Gpp) -fopenmp -o build/test/Utilities/TensorOperations/GpuConvolution/GpuConvolutionTest test/Utilities/TensorOperations/GpuConvolution/GpuConvolutionTest.cpp -O3 -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(MLDEV_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/DeepLearning/Implementation/Layers/NeuralNetwork/Convolution2dTest: build/test/googletest/libgtest.a test/DeepLearning/Implementation/Layers/NeuralNetwork/Convolution2dTest.cpp $(MLDEV)
	mkdir -p build/test/DeepLearning/Implementation/Layers/NeuralNetwork
	$(Gpp) -ggdb -fopenmp -o build/test/DeepLearning/Implementation/Layers/NeuralNetwork/Convolution2dTest test/DeepLearning/Implementation/Layers/NeuralNetwork/Convolution2dTest.cpp -O3 -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(MLDEV_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/Utilities/TensorOperations/GpuMatrixMultiply/gpuMatrixMultiplyTest: build/test/googletest/libgtest.a test/Utilities/TensorOperations/GpuMatrixMultiply/gpuMatrixMultiplyTest.cpp $(MLDEV)
	mkdir -p build/test/Utilities/TensorOperations/GpuMatrixMultiply
	$(Gpp) -ggdb -o build/test/Utilities/TensorOperations/GpuMatrixMultiply/gpuMatrixMultiplyTest test/Utilities/TensorOperations/GpuMatrixMultiply/gpuMatrixMultiplyTest.cpp -O3 -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(MLDEV_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/Utilities/ComputeTopology/machineEvaluatorTest: build/test/googletest/libgtest.a test/Utilities/ComputeTopology/machineEvaluatorTest.cpp $(MLDEV)
	mkdir -p build/test/Utilities/ComputeTopology
	$(Gpp) -ggdb -o build/test/Utilities/ComputeTopology/machineEvaluatorTest test/Utilities/ComputeTopology/machineEvaluatorTest.cpp -O3 -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(MLDEV_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/Utilities/Common/OptionalTest: build/test/googletest/libgtest.a test/Utilities/Common/OptionalTest.cpp $(MLDEV)
	mkdir -p build/test/Utilities/Common
	$(Gpp) -ggdb -o build/test/Utilities/Common/OptionalTest test/Utilities/Common/OptionalTest.cpp -O3 -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(MLDEV_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/DeepLearning/Implementation/Tensor/tensorTest: build/test/googletest/libgtest.a test/DeepLearning/Implementation/Tensor/tensorTest.cpp $(MLDEV)
	mkdir -p build/test/DeepLearning/Implementation/Tensor
	$(Gpp) -ggdb -o build/test/DeepLearning/Implementation/Tensor/tensorTest test/DeepLearning/Implementation/Tensor/tensorTest.cpp -O3 -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(MLDEV_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/Utilities/TensorOperations/TypeConversions/TypeConverterTest: build/test/googletest/libgtest.a test/Utilities/TensorOperations/TypeConversions/TypeConverterTest.cpp $(MLDEV)
	mkdir -p build/test/Utilities/TensorOperations/TypeConversions
	$(Gpp) -ggdb -o build/test/Utilities/TensorOperations/TypeConversions/TypeConverterTest test/Utilities/TensorOperations/TypeConversions/TypeConverterTest.cpp -O3 -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(MLDEV_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/Utilities/TensorOperations/Arithmetic/ArithmeticTest: build/test/googletest/libgtest.a test/Utilities/TensorOperations/Arithmetic/ArithmeticTest.cpp $(MLDEV)
	mkdir -p build/test/Utilities/TensorOperations/Arithmetic/
	$(Gpp) -ggdb -o build/test/Utilities/TensorOperations/Arithmetic/ArithmeticTest -O3 -std=c++11 -pthread test/Utilities/TensorOperations/Arithmetic/ArithmeticTest.cpp $(CUDA_INCLUDE_DIRS) $(MLDEV_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/Utilities/TensorOperations/Misc/MiscTest: build/test/googletest/libgtest.a test/Utilities/TensorOperations/Misc/MiscTest.cpp $(MLDEV)
	mkdir -p build/test/Utilities/TensorOperations/Misc/
	$(Gpp) -ggdb -o build/test/Utilities/TensorOperations/Misc/MiscTest -O3 -std=c++11 -pthread test/Utilities/TensorOperations/Misc/MiscTest.cpp $(CUDA_INCLUDE_DIRS) $(MLDEV_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/DeepLearning/Implementation/Layers/Utility/UtilityLayerTest: build/test/googletest/libgtest.a test/DeepLearning/Implementation/Layers/Utility/UtilityLayerTest.cpp $(MLDEV)
	mkdir -p build/test/DeepLearning/Implementation/Layers/Utility
	$(Gpp) -ggdb -o build/test/DeepLearning/Implementation/Layers/Utility/UtilityLayerTest test/DeepLearning/Implementation/Layers/Utility/UtilityLayerTest.cpp -O3 -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(MLDEV_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/DeepLearning/Implementation/Layers/Activations/ActivationsLayerTest: build/test/googletest/libgtest.a test/DeepLearning/Implementation/Layers/Activations/ActivationsLayerTest.cpp $(MLDEV)
	mkdir -p build/test/DeepLearning/Implementation/Layers/Activations
	$(Gpp) -ggdb -o build/test/DeepLearning/Implementation/Layers/Activations/ActivationsLayerTest test/DeepLearning/Implementation/Layers/Activations/ActivationsLayerTest.cpp -O3 -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(MLDEV_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/DeepLearning/Implementation/Layers/NeuralNetwork/DropOutTest: build/test/googletest/libgtest.a test/DeepLearning/Implementation/Layers/NeuralNetwork/DropOutTest.cpp $(MLDEV)
	mkdir -p build/test/DeepLearning/Implementation/Layers/NeuralNetwork
	$(Gpp) -ggdb -o build/test/DeepLearning/Implementation/Layers/NeuralNetwork/DropOutTest test/DeepLearning/Implementation/Layers/NeuralNetwork/DropOutTest.cpp -O3 -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(MLDEV_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/DeepLearning/Implementation/Layers/Loss/CategoricalCrossEntropyLossTest: build/test/googletest/libgtest.a test/DeepLearning/Implementation/Layers/Loss/CategoricalCrossEntropyLossTest.cpp $(MLDEV)
	mkdir -p build/test/DeepLearning/Implementation/Layers/Loss
	$(Gpp) -ggdb -o build/test/DeepLearning/Implementation/Layers/Loss/CategoricalCrossEntropyLossTest test/DeepLearning/Implementation/Layers/Loss/CategoricalCrossEntropyLossTest.cpp -O3 -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(MLDEV_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/Utilities/TensorOperations/DeepLearning/CrossEntropyLossTest: build/test/googletest/libgtest.a test/Utilities/TensorOperations/DeepLearning/CrossEntropyLossTest.cpp $(MLDEV)
	mkdir -p build/test/Utilities/TensorOperations/DeepLearning/
	$(Gpp) -ggdb -o build/test/Utilities/TensorOperations/DeepLearning/CrossEntropyLossTest -O3 -std=c++11 -pthread test/Utilities/TensorOperations/DeepLearning/CrossEntropyLossTest.cpp $(CUDA_INCLUDE_DIRS) $(MLDEV_LIBS) $(TEST_COMPILE_DEPENDENCIES)







































WholeBuild: InitialSetup Utilities DeepLearning Clustering

Utilities: gpuMatrixTranspose.o
DeepLearning:
Clustering:

BuildTestBase: WholeBuild googletest
BuildTests: BuildTestsUtilities
BuildTestsUtilities: BuildTestBase BuildUtilityTestBase BuildWorkQueueTests

RunTests: BuildTests RunTestsUtilities
RunTestsUtilities: BuildTestsUtilities RunWorkQueueTests

BuildUtilityTestBase: googletest
	mkdir -p build/test/Utilities

# Utilities

# FIXME: separate into build and test
# FIXME: output files go into the biuld directory
#gpuMatrixMultiply: Utilities/TensorOperations/GpuMatrixMultiply/gpuMatrixMultiply.h gpuMatrixTransposeKernels.o makefile

workQueue: Utilities/WorkQueue/WorkQueue.h

BuildWorkQueueTests: BuildUtilityTestBase workQueue test/Utilities/WorkQueue/WorkQueueUnorderedTest.cpp test/Utilities/WorkQueue/WorkQueueTest.cpp
	mkdir -p build/test/Utilities/WorkQueue/
	g++ -o build/test/Utilities/WorkQueue/WorkQueueTest -O3 -std=c++11 -pthread -IUtilities/WorkQueue/ test/Utilities/WorkQueue/WorkQueueTest.cpp $(TEST_COMPILE_DEPENDENCIES)
	g++ -o build/test/Utilities/WorkQueue/WorkQueueUnorderedTest -O3 -std=c++11 -pthread -IUtilities/WorkQueue/ test/Utilities/WorkQueue/WorkQueueUnorderedTest.cpp $(TEST_COMPILE_DEPENDENCIES)

RunWorkQueueTests: BuildWorkQueueTests
	build/test/Utilities/WorkQueue/WorkQueueTest
	build/test/Utilities/WorkQueue/WorkQueueUnorderedTest























# Old stuff, some of it to bring into real makefile above


article2Vec: article2Vec.cu article2Vec.cpp article2Vec.h article2VecMonitor.cpp  extractDataSet.cpp
	g++ extractDataSet.cpp -std=c++11 -O3 -o extractDataSet
	nvcc --cudart static -gencode arch=compute_52,code=compute_52 -gencode arch=compute_52,code=sm_52 -c -I/usr/local/cuda/include -Xptxas -O3,-v article2Vec.cu -o article2Vec.o
	g++ article2Vec.cpp -std=c++11 -O3 -o article2Vec -fopenmp article2Vec.o -L/usr/local/cuda/lib64 -lcudart -lcublas -lcusolver
	g++ article2VecMonitor.cpp -std=c++11 -O3 -o article2VecMonitor

createRTree: createRTree.cpp mappedFileObjects.h
	g++ -I /usr/local/boost_1_69_0 -std=c++11 createRTree.cpp -O3 -pthread -o createRTree

addArticleIndexMap: addArticleIndexMap.cpp 
	g++ -I /usr/local/boost_1_69_0 -std=c++11 addArticleIndexMap.cpp -O3 -pthread -o addArticleIndexMap

cluster: cluster.cpp clusterKernels clusterTests.h cluster.h
	g++ -I /usr/local/boost_1_69_0 -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -std=c++11 clusterKernels.o cluster.cpp -O3 -pthread -o cluster -lcusolver -lcudart
	
clusterKMeans: clusterKMeans.cpp clusterKMeans.h
	g++ -I /usr/local/boost_1_69_0 -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -std=c++11 clusterKMeans.cpp -O3 -pthread -o clusterKMeans -lcudart
	

MappedFileDemo: MappedFileDemo.cpp
	g++ -I /usr/local/boost_1_69_0 -std=c++11 -pthread MappedFileDemo.cpp -O0 -g -o MappedFileDemo

clusterKernels: clusterKernels.cu clusterKernels.h
	nvcc --cudart static -std=c++11 -gencode arch=compute_52,code=compute_52 -gencode arch=compute_52,code=sm_52 -c -I/usr/local/cuda/include -Xptxas -O3,-v clusterKernels.cu
	#g++ -o clusterKernels -std=c++11 clusterKernels.o -O3 -L/usr/local/cuda/lib64 -lcusolver -lcudart

optimalMatrixMultiplyBad: optimalMatrixMultiplyBad.cpp optimalMatrixMultiplyKernels.cu makefile
	nvcc --cudart static -std=c++11 -gencode=arch=compute_52,code=compute_52 -gencode=arch=compute_75,code=compute_75 -g -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_75,code=sm_75 -c -I/usr/local/cuda/include -Xptxas -O3,-v optimalMatrixMultiplyBad.cpp
	nvcc --cudart static -std=c++11 -gencode=arch=compute_52,code=compute_52 -gencode=arch=compute_75,code=compute_75 -g -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_75,code=sm_75 -c -I/usr/local/cuda/include -Xptxas -O3,-v optimalMatrixMultiplyKernels.cu
	g++ -o optimalMatrixMultiplyBad -std=c++11 optimalMatrixMultiplyBad.o optimalMatrixMultiplyKernels.o -O3 -L/usr/local/cuda/lib64 -lcublas -lcublasLt -lcusolver -lcudart


cublasTest: cublasTest.cpp makefile
	nvcc --cudart static -std=c++11 -gencode=arch=compute_52,code=compute_52 -gencode=arch=compute_75,code=compute_75 -g -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_75,code=sm_75 -c -I/usr/local/cuda/include -Xptxas -O3,-v cublasTest.cpp
	g++ -o cublasTest -std=c++11 cublasTest.o -O3 -L/usr/local/cuda/lib64 -lcublas -lcublasLt -lcusolver -lcudart

cublasTestDebug: cublasTest.cpp
	nvcc --cudart static -std=c++11 -gencode=arch=compute_52,code=compute_52 -gencode=arch=compute_75,code=compute_75 -g -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_75,code=sm_75 -c -I/usr/local/cuda/include -Xptxas -O3,-v cublasTest.cpp
	g++ -o cublasTestDebug -std=c++11 cublasTest.o -O0 -L/usr/local/cuda/lib64 -lcublas -lcublasLt -ggdb -lcusolver -lcudart

simpleCublasTestcase: simpleCublasTestcase.cpp
	nvcc --cudart static -std=c++11 -gencode arch=compute_52,code=compute_52 -g -gencode arch=compute_52,code=sm_52 -c -I/usr/local/cuda/include -Xptxas -O3,-v simpleCublasTestcase.cpp
	g++ -o simpleCublasTestcase -std=c++11 simpleCublasTestcase.o -O0 -L/usr/local/cuda/lib64 -lcublas -lcublasLt -ggdb -lcusolver -lcudart


outerProduct: outerProduct.cu
	nvcc --cudart static -std=c++11 -gencode arch=compute_52,code=compute_52 -g -gencode arch=compute_52,code=sm_52 -c -I/usr/local/cuda/include -Xptxas -O3,-v outerProduct.cu
	g++ -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -std=c++11 outerProduct.o -O3 -pthread -o outerProduct -lcudart

gpuRTreeKernels: GpuRTreeKernels.cu GpuRTreeKernels.h
	nvcc --cudart static -std=c++11 -gencode arch=compute_52,code=compute_52 -g -gencode arch=compute_52,code=sm_52 -c -I/usr/local/cuda/include -Xptxas -O3,-v GpuRTreeKernels.cu
#	nvcc --cudart static -std=c++11 -gencode arch=compute_52,code=compute_52 -g -gencode arch=compute_52,code=sm_52 -I/usr/local/cuda/include -Xptxas -O3,-v --device-c GpuRTreeKernels.cu -o GpuRTreeKernels.o

gpuRTree: GpuRTree.cpp GpuRTreeNode.h GpuRTreeNode.cpp GpuRTreeTests.h gpuRTreeKernels GpuRTreeTests.h NearestNeighborExecutor.h NearestNeighborExecutor.cpp makefile
	g++ -c GpuRTree.cpp -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -std=c++11 -pthread -O3 -lcudart
	g++ -c GpuRTreeNode.cpp -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -std=c++11 -pthread -O3 -lcudart
	g++ -c NearestNeighborExecutor.cpp -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -std=c++11 -pthread -O3 -lcudart
	g++ GpuRTree.o -I/usr/local/cuda/include -L/usr/local/cuda/lib64 GpuRTreeNode.o NearestNeighborExecutor.o GpuRTreeKernels.o -std=c++11 -pthread -O3 -o gpuRTree -lcudart
#	g++ GpuRTree.cpp NearestNeighborExecutor.cpp -I/usr/local/cuda/include -L/usr/local/cuda/lib64 GpuRTreeKernels.o -std=c++11 -pthread -O3 -o gpuRTree -lcudart
#	nvcc --cudart static -std=c++11 -gencode arch=compute_52,code=compute_52 -g -gencode arch=compute_52,code=sm_52 -I/usr/local/cuda/include -Xptxas -O3,-v --device-c GpuRTree.cu -o gpuRTree.o
#	nvcc --cudart static -std=c++11 -gencode arch=compute_52,code=compute_52 -g -gencode arch=compute_52,code=sm_52 -I/usr/local/cuda/include -Xptxas -O3,-v  GpuRTreeKernels.o GpuRTree.o -o gpuRTree

gpuRTreeDebug: GpuRTree.cpp GpuRTreeNode.h GpuRTreeTests.h gpuRTreeKernels GpuRTreeTests.h NearestNeighborExecutor.h NearestNeighborExecutor.cpp makefile
	g++ -c GpuRTree.cpp -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -std=c++11 -pthread -O3 -ggdb -lcudart
	g++ -c NearestNeighborExecutor.cpp -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -std=c++11 -pthread -O3 -ggdb -lcudart
	g++ GpuRTree.o -I/usr/local/cuda/include -L/usr/local/cuda/lib64 NearestNeighborExecutor.o GpuRTreeKernels.o -std=c++11 -pthread -O3 -ggdb -o gpuRTreeDebug -lcudart

buildRTree: buildRTree.cpp
	g++ buildRTree.cpp -std=c++11 -pthread -O3 -o buildRTree
	
buildRTreeDebug: buildRTree.cpp
	g++ buildRTree.cpp -std=c++11 -pthread -O0 -ggdb -o buildRTreeDebug
	

#nvcc -c -I/usr/local/cuda/include -Xptxas -O3,-v article2Vec.cu && g++ -fopenmp -o article2Vec article2Vec.o -L/usr/local/cuda/lib64 -lcudart -lcublas -lcusolver -O3



#ar r article2Vec.a article2Vec.o
#ranlib article2Vec.a
#g++ article2Vec.cpp -std=c++11 -O3 -o article2Vec -fopenmp article2Vec.a -L/usr/local/cuda/lib64 -lcudart -lcublas -lcusolver


##g++ -std=c++11 -O3 -o article2Vec -fopenmp article2Vec.o -L/usr/local/cuda/lib64 -lcudart -lcublas -lcusolver










