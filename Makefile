CUDA_INCLUDE_DIRS = -I /usr/local/cuda/include -I /usr/include
CUDA_LIBRARIES = -L /usr/local/cuda/lib64 -l cublas -l cublasLt -l cusolver -l cudart -l cufile -L /usr/lib/x86_64-linux-gnu -l cudnn -l boost_filesystem -lX11
CUDA = $(CUDA_INCLUDE_DIRS) $(CUDA_LIBRARIES)
# https://en.wikipedia.org/wiki/CUDA#GPUs_supported
COMPUTE_CAPABILITIES_MOBILE_DEVICES = -gencode=arch=compute_62,code=compute_62 -gencode=arch=compute_62,code=sm_62 \
                                      -gencode=arch=compute_72,code=compute_72 -gencode=arch=compute_72,code=sm_72 \
                                      -gencode=arch=compute_87,code=compute_87 -gencode=arch=compute_87,code=sm_87
#COMPUTE_CAPABILITIES = -gencode=arch=compute_52,code=compute_52 -gencode=arch=compute_52,code=sm_52 \
#                       -gencode=arch=compute_60,code=compute_60 -gencode=arch=compute_60,code=sm_60 \
#                       -gencode=arch=compute_61,code=compute_61 -gencode=arch=compute_61,code=sm_61 \
#                       -gencode=arch=compute_70,code=compute_70 -gencode=arch=compute_70,code=sm_70 \
#                       -gencode=arch=compute_75,code=compute_75 -gencode=arch=compute_75,code=sm_75
COMPUTE_CAPABILITIES = -gencode=arch=compute_75,code=compute_75 -gencode=arch=compute_75,code=sm_75 \
                       #-gencode=arch=compute_89,code=compute_89 -gencode=arch=compute_89,code=sm_89


#COMPUTE_CAPABILITIES_WITH_TENSOR_CORES = -gencode=arch=compute_70,code=compute_70 -gencode=arch=compute_70,code=sm_70 \
#                       -gencode=arch=compute_75,code=compute_75 -gencode=arch=compute_75,code=sm_75
COMPUTE_CAPABILITIES_WITH_TENSOR_CORES = -gencode=arch=compute_75,code=compute_75 -gencode=arch=compute_75,code=sm_75 \
										 #-gencode=arch=compute_89,code=compute_89 -gencode=arch=compute_89,code=sm_89


BOOST_INCLUDE_DIR = -I /usr/local/boost -ldl -lcurses

GRAPHICS_MAGICK_INCLUDE_DIR = -I /usr/local/include/GraphicsMagick/
GRAPHICS_MAGICK = `GraphicsMagick++-config --cppflags --cxxflags --ldflags --libs`

THOR_LIBS = $(CUDA) -I./ -L./ -lThor

INCLUDE_HOME_DIR = -I ./
INCLUDE_DIRS = $(INCLUDE_HOME_DIR) $(CUDA_INCLUDE_DIRS) $(BOOST_INCLUDE_DIR) $(GRAPHICS_MAGICK_INCLUDE_DIR)

INCLUDE_DIRS_TEST = $(INCLUDE_DIRS) -I build/test/googletest/include
LIB_DIRS_TEST = -L build/test/googletest
LIBS_TEST = -lgtest -lgtest_main -pthread
TEST_COMPILE_DEPENDENCIES = $(INCLUDE_DIRS_TEST) $(LIB_DIRS_TEST) $(LIBS_TEST) $(CUDA)

DEBUG = -ggdb -O0
NVCC_DEBUG = -g -Xptxas -O0

RELEASE = false

ifeq ($(RELEASE),true)
	Gpp = g++ -fPIC -Wall -Werror -fopenmp -O3 -Wl,--no-as-needed -DTHOR_RELEASE -DGDK_NVDIRECT -D_FILE_OFFSET_BITS=64
	Nvcc = nvcc --Werror all-warnings -O3 -Xptxas -O3,-v -Xcompiler -fPIC -DTHOR_RELEASE -DGDK_NVDIRECT -D_FILE_OFFSET_BITS=64
else
	Gpp = g++ -fPIC -Wall -Werror -fopenmp -D_GLIBCXX_DEBUG -ggdb -O0 -Wl,--no-as-needed -DTHOR_DEBUG -DGDK_NVDIRECT -D_FILE_OFFSET_BITS=64
	Nvcc = nvcc --Werror all-warnings -Xcompiler -fPIC -D_GLIBCXX_DEBUG -g -DTHOR_DEBUG -DGDK_NVDIRECT -D_FILE_OFFSET_BITS=64
endif

RUN_ALL_TESTS = build/test/DeepLearning/Api/Network/NetworkTest && \
                build/test/DeepLearning/Implementation/Layers/NeuralNetwork/Convolution2dTest && \
                build/test/Utilities/TensorOperations/GpuConvolution/GpuConvolutionTest && \
				build/test/Utilities/TensorOperations/Loss/CrossEntropyLossTest && \
				build/test/DeepLearning/Implementation/Layers/Loss/CategoricalCrossEntropyTest && \
				build/test/DeepLearning/Implementation/Layers/Loss/BinaryCrossEntropyTest && \
				build/test/DeepLearning/Implementation/Layers/Loss/MeanSquaredErrorTest && \
				build/test/DeepLearning/Implementation/Layers/Loss/MeanAbsoluteErrorTest && \
				build/test/DeepLearning/Implementation/Layers/Loss/MeanAbsolutePercentageErrorTest && \
				build/test/Utilities/TensorOperations/Misc/ComputeCategoricalAccuracyTest && \
				build/test/Utilities/TensorOperations/Misc/ComputeBinaryAccuracyTest && \
				build/test/DeepLearning/Implementation/Layers/Metric/CategoricalAccuracyTest && \
				build/test/DeepLearning/Implementation/Layers/Metric/BinaryAccuracyTest && \
				build/test/DeepLearning/Api/Layers/Learning/FullyConnectedTest && \
                build/test/DeepLearning/Api/Layers/Learning/Convolution2dTest && \
                build/test/DeepLearning/Api/Layers/Activations/ActivationsTest && \
                build/test/DeepLearning/Api/Layers/Loss/CategoricalCrossEntropyTest && \
                build/test/DeepLearning/Api/Layers/Loss/BinaryCrossEntropyTest && \
                build/test/DeepLearning/Api/Layers/Metric/CategoricalAccuracyTest && \
                build/test/DeepLearning/Api/Layers/Metric/BinaryAccuracyTest && \
                build/test/DeepLearning/Api/Layers/Utility/UtilityLayerTests && \
                build/test/DeepLearning/Implementation/Layers/NeuralNetwork/BatchNormalizationTest && \
                build/test/DeepLearning/Implementation/Layers/NeuralNetwork/FullyConnectedTest && \
                build/test/DeepLearning/Implementation/Layers/Loss/LossShaperTest && \
                build/test/DeepLearning/Implementation/Layers/NeuralNetwork/PoolingTest && \
                build/test/Utilities/TensorOperations/Arithmetic/ArithmeticTest && \
                build/test/Utilities/Common/OptionalTest && \
                build/test/Utilities/Random/FullPeriodRandomTest && \
                build/test/Utilities/WorkQueue/AsyncQueueTest && \
                build/test/Utilities/WorkQueue/AsyncTensorQueueTest && \
                build/test/DeepLearning/Implementation/Layers/NeuralNetwork/DropOutTest && \
                build/test/Utilities/TensorOperations/Misc/MiscTest && \
                build/test/Utilities/TensorOperations/Misc/BatchReduceTest && \
                build/test/Utilities/Loaders/ShardedRawDatasetCreatorTest && \
                build/test/DeepLearning/Implementation/Layers/Utility/UtilityLayerTest && \
                build/test/DeepLearning/Implementation/Layers/Activations/ActivationsLayerTest && \
                build/test/Utilities/TensorOperations/TypeConversions/TypeConverterTest && \
                build/test/DeepLearning/Implementation/Tensor/TensorTest && \
                build/test/DeepLearning/Implementation/Tensor/TensorTrigonometricKernelsTest && \
                build/test/DeepLearning/Implementation/Tensor/TensorHyperbolicTrigonometricKernelsTest && \
                build/test/Utilities/ComputeTopology/machineEvaluatorTest && \
                build/test/Utilities/TensorOperations/GpuMatrixTranspose/gpuMatrixTransposeTest && \
                build/test/Utilities/TensorOperations/GpuMatrixMultiply/CublasMatrixMultiplyTest && \
                build/test/DeepLearning/Api/Layers/Loss/LossShaperTest && \
                build/test/DeepLearning/Api/Layers/Loss/MeanSquaredErrorTest && \
                build/test/DeepLearning/Api/Layers/Loss/MeanAbsoluteErrorTest && \
                build/test/DeepLearning/Api/Layers/Loss/MeanAbsolutePercentageErrorTest && \
                build/test/DeepLearning/Api/Optimizers/SgdTest && \
                build/test/DeepLearning/Api/Optimizers/AdamTest && \
                build/test/Utilities/TensorOperations/Optimizers/AdamTest && \
                build/test/Utilities/WorkQueue/WorkQueueUnorderedTest && \
                build/test/Utilities/WorkQueue/WorkQueueTest \

				# FIXME: put back
                # build/test/DeepLearning/Implementation/SimpleNetworkTest && \
                # build/test/DeepLearning/Implementation/Layers/Optimizers/SgdTest && \
                # build/test/DeepLearning/Implementation/Layers/Optimizers/AdamTest && \

ALL_TESTS = build/test/DeepLearning/Implementation/Layers/Loss/CategoricalCrossEntropyTest \
			build/test/DeepLearning/Implementation/Layers/Loss/BinaryCrossEntropyTest \
			build/test/DeepLearning/Api/Layers/Loss/CategoricalCrossEntropyTest \
			build/test/DeepLearning/Api/Layers/Loss/BinaryCrossEntropyTest \
			build/test/DeepLearning/Api/Layers/Metric/CategoricalAccuracyTest \
			build/test/DeepLearning/Api/Layers/Metric/BinaryAccuracyTest \
			build/test/Utilities/Random/FullPeriodRandomTest \
            build/test/Utilities/WorkQueue/AsyncQueueTest \
            build/test/Utilities/WorkQueue/AsyncTensorQueueTest \
            build/test/DeepLearning/Api/Layers/Learning/FullyConnectedTest \
            build/test/DeepLearning/Implementation/Layers/NeuralNetwork/BatchNormalizationTest \
            build/test/DeepLearning/Api/Layers/Learning/Convolution2dTest \
            build/test/DeepLearning/Api/Layers/Activations/ActivationsTest \
            build/test/DeepLearning/Api/Network/NetworkTest \
            build/test/DeepLearning/Api/Layers/Utility/UtilityLayerTests \
            build/test/Utilities/TensorOperations/Arithmetic/ArithmeticTest \
            build/test/Utilities/TensorOperations/Misc/MiscTest \
            build/test/Utilities/TensorOperations/Misc/BatchReduceTest \
            build/test/Utilities/Loaders/ShardedRawDatasetCreatorTest \
            build/test/DeepLearning/Implementation/Layers/Utility/UtilityLayerTest \
            build/test/DeepLearning/Implementation/Layers/NeuralNetwork/DropOutTest \
            build/test/DeepLearning/Implementation/Layers/Activations/ActivationsLayerTest \
            build/test/Utilities/TensorOperations/TypeConversions/TypeConverterTest \
            build/test/DeepLearning/Implementation/Tensor/TensorTest \
			build/test/DeepLearning/Implementation/Tensor/TensorTrigonometricKernelsTest \
			build/test/DeepLearning/Implementation/Tensor/TensorHyperbolicTrigonometricKernelsTest \
            build/test/Utilities/WorkQueue/WorkQueueUnorderedTest \
            build/test/Utilities/WorkQueue/WorkQueueTest \
            build/test/Utilities/TensorOperations/GpuMatrixMultiply/CublasMatrixMultiplyTest \
            build/test/Utilities/TensorOperations/GpuConvolution/GpuConvolutionTest \
            build/test/DeepLearning/Implementation/Layers/NeuralNetwork/FullyConnectedTest \
            build/test/DeepLearning/Implementation/Layers/Loss/LossShaperTest \
            build/test/DeepLearning/Implementation/Layers/Loss/MeanSquaredErrorTest \
            build/test/DeepLearning/Implementation/Layers/Loss/MeanAbsoluteErrorTest \
            build/test/DeepLearning/Implementation/Layers/Loss/MeanAbsolutePercentageErrorTest \
            build/test/DeepLearning/Implementation/Layers/NeuralNetwork/PoolingTest \
            build/test/DeepLearning/Implementation/Layers/NeuralNetwork/Convolution2dTest \
            build/test/Utilities/TensorOperations/GpuMatrixTranspose/gpuMatrixTransposeTest \
            build/test/Utilities/ComputeTopology/machineEvaluatorTest \
            build/test/Utilities/Common/OptionalTest \
            build/test/DeepLearning/Api/Visualizers/ConsoleVisualizerTest \
            build/test/DeepLearning/Api/Optimizers/SgdTest \
            build/test/DeepLearning/Api/Optimizers/AdamTest \
            build/test/DeepLearning/Implementation/Layers/Optimizers/SgdTest \
            build/test/DeepLearning/Implementation/Layers/Optimizers/AdamTest \
            build/test/Utilities/TensorOperations/Optimizers/AdamTest \
            build/test/DeepLearning/Api/Layers/Loss/MeanSquaredErrorTest \
            build/test/DeepLearning/Api/Layers/Loss/MeanAbsoluteErrorTest \
            build/test/DeepLearning/Api/Layers/Loss/MeanAbsolutePercentageErrorTest \
            build/test/DeepLearning/Api/Layers/Loss/LossShaperTest \
			build/test/Utilities/TensorOperations/Loss/CrossEntropyLossTest \
			build/test/Utilities/TensorOperations/Misc/ComputeCategoricalAccuracyTest \
			build/test/Utilities/TensorOperations/Misc/ComputeBinaryAccuracyTest \
			build/test/DeepLearning/Implementation/Layers/Metric/CategoricalAccuracyTest \
			build/test/DeepLearning/Implementation/Layers/Metric/BinaryAccuracyTest \


			# FIXME: put back
			# build/test/DeepLearning/Implementation/SimpleNetworkTest \


ALL_OBJECT_FILES = build/Utilities/TensorOperations/GpuMatrixTranspose/gpuMatrixTransposeKernels.o \
                   build/Utilities/TensorOperations/TypeConversions/TypeConverterKernels.o \
				   build/DeepLearning/Implementation/Layers/Activation/Softmax.o \
                   build/Utilities/TensorOperations/Activation/Tanh.o \
                   build/Utilities/TensorOperations/Activation/Relu.o \
                   build/Utilities/TensorOperations/Activation/Swish.o \
                   build/Utilities/TensorOperations/Activation/Selu.o \
                   build/Utilities/TensorOperations/Activation/HardSigmoid.o \
                   build/Utilities/TensorOperations/Activation/Gelu.o \
                   build/Utilities/TensorOperations/Activation/SoftSign.o \
                   build/Utilities/TensorOperations/Activation/SoftPlus.o \
                   build/Utilities/TensorOperations/Activation/Exponential.o \
                   build/Utilities/TensorOperations/Activation/Elu.o \
                   build/Utilities/TensorOperations/Activation/Sigmoid.o \
                   build/Utilities/TensorOperations/Arithmetic/Average.o \
                   build/Utilities/TensorOperations/Arithmetic/Sum.o \
                   build/Utilities/TensorOperations/Loss/MeanSquaredError.o \
                   build/DeepLearning/Api/Layers/Loss/MeanSquaredError.o \
                   build/Utilities/TensorOperations/Loss/MeanAbsoluteError.o \
                   build/DeepLearning/Api/Layers/Loss/MeanAbsoluteError.o \
                   build/Utilities/TensorOperations/Loss/MeanAbsolutePercentageError.o \
                   build/DeepLearning/Api/Layers/Loss/MeanAbsolutePercentageError.o \
                   build/DeepLearning/Api/Layers/Loss/CategoricalCrossEntropy.o \
                   build/DeepLearning/Api/Layers/Loss/BinaryCrossEntropy.o \
                   build/Utilities/TensorOperations/Misc/BatchReduce.o \
                   build/Utilities/TensorOperations/DeepLearning/Add1dBias.o \
                   build/Utilities/TensorOperations/Arithmetic/SumScale.o \
                   build/Utilities/TensorOperations/Arithmetic/SumManyToOne.o \
                   build/Utilities/TensorOperations/Arithmetic/Exponentiation.o \
                   build/Utilities/TensorOperations/Arithmetic/ElementwiseSubtract.o \
                   build/Utilities/TensorOperations/Arithmetic/MultiplyByScalar.o \
                   build/Utilities/TensorOperations/Loss/CrossEntropyLoss.o \
                   build/Utilities/TensorOperations/Loss/CategoricalCrossEntropyLoss.o \
                   build/Utilities/TensorOperations/Loss/BinaryCrossEntropyLoss.o \
                   build/Utilities/TensorOperations/Misc/Map.o \
                   build/Utilities/TensorOperations/Misc/Split.o \
                   build/Utilities/TensorOperations/Misc/Pad.o \
                   build/Utilities/TensorOperations/Misc/Extract.o \
                   build/Utilities/TensorOperations/Misc/Concatenate.o \
                   build/Utilities/TensorOperations/Optimizers/Adam.o \
                   build/Utilities/TensorOperations/GpuConvolution/GpuConvolution.o \
                   build/Utilities/TensorOperations/GpuConvolution/GpuConvolutionKernels.o \
                   build/DeepLearning/Implementation/Layers/NeuralNetwork/FullyConnected.o \
                   build/DeepLearning/Implementation/Layers/Loss/LossShaper.o \
                   build/DeepLearning/Implementation/Layers/Loss/MeanSquaredError.o \
                   build/DeepLearning/Implementation/Layers/Loss/MeanAbsoluteError.o \
                   build/DeepLearning/Implementation/Layers/Loss/MeanAbsolutePercentageError.o \
                   build/DeepLearning/Implementation/Layers/Loss/CrossEntropy.o \
                   build/Utilities/TensorOperations/GpuMatrixMultiply/CublasKernel.o \
                   build/Utilities/Common/ReferenceCounted.o \
                   build/Utilities/Common/ThreadJoinQueue.o \
                   build/Utilities/Common/CudnnHelper.o \
                   build/Utilities/TensorOperations/GpuMatrixMultiply/CublasMatrixMultiply.o \
                   build/Utilities/ComputeTopology/MachineEvaluator.o \
                   build/DeepLearning/Implementation/Tensor/Tensor.o \
                   build/DeepLearning/Implementation/Tensor/TensorArithmeticKernels.o \
                   build/DeepLearning/Implementation/Tensor/TensorMathKernels.o \
                   build/DeepLearning/Implementation/Tensor/TensorTrigonometryKernels.o \
                   build/DeepLearning/Implementation/Tensor/TensorHyperbolicTrigonometryKernels.o \
                   build/DeepLearning/Implementation/Layers/Layer.o \
                   build/Utilities/TensorOperations/TypeConversions/TypeConverter.o \
                   build/DeepLearning/Implementation/Layers/NeuralNetwork/Pooling.o \
                   build/DeepLearning/Implementation/Layers/NeuralNetwork/BatchNormalization.o \
                   build/DeepLearning/Implementation/Layers/Optimizers/Optimizer.o \
                   build/DeepLearning/Implementation/Layers/Optimizers/Sgd.o \
                   build/DeepLearning/Implementation/Layers/Optimizers/Adam.o \
                   build/DeepLearning/Implementation/Initializers/Initializer.o \
                   build/DeepLearning/Implementation/Initializers/UniformRandom.o \
                   build/DeepLearning/Implementation/Initializers/Glorot.o \
                   build/DeepLearning/Api/Visualizers/ConsoleVisualizer.o \
                   build/DeepLearning/Api/HyperparameterControllers/HyperparameterController.o \
                   build/DeepLearning/Api/Executors/LocalExecutor.o \
                   build/DeepLearning/Api/Network/Network.o \
                   build/DeepLearning/Api/Optimizers/Optimizer.o \
                   build/DeepLearning/Api/Optimizers/Sgd.o \
                   build/DeepLearning/Api/Optimizers/Adam.o \
                   build/Utilities/Common/Stream.o \
                   build/Utilities/Common/Event.o \
                   build/Utilities/Loaders/Shard.o \
                   build/Utilities/Loaders/ShardedRawDatasetCreator.o \
                   build/Utilities/Loaders/ImageLoader.o \
                   build/Utilities/TensorOperations/Misc/BatchReduce.o \
                   build/Utilities/Loaders/ImageProcessor.o \
                   build/Utilities/Loaders/NoOpDataProcessor.o \
                   build/Utilities/Loaders/BatchAssembler.o \
                   build/DeepLearning/Api/Loaders/LocalBatchLoader.o \
                   build/Utilities/WorkQueue/AsyncTensorQueue.o \
                   build/DeepLearning/Api/Tensor/Tensor.o \
                   build/DeepLearning/Api/Layers/Layer.o \
                   build/DeepLearning/Api/Initializers/Initializer.o \
                   build/DeepLearning/Api/Layers/Learning/FullyConnected.o \
                   build/DeepLearning/Api/Layers/Learning/Convolution2d.o \
                   build/DeepLearning/Api/Layers/Utility/BatchNormalization.o \
                   build/DeepLearning/Api/Layers/Utility/DropOut.o \
                   build/DeepLearning/Api/Layers/Utility/NetworkOutput.o \
                   build/DeepLearning/Api/Layers/Learning/Inception.o \
                   build/Utilities/TensorOperations/Misc/ComputeCategoricalAccuracy.o \
                   build/Utilities/TensorOperations/Misc/ComputeBinaryAccuracy.o \
                   build/DeepLearning/Api/ExampleNetworks/AlexNet.o \
                   build/DeepLearning/Api/ExampleNetworks/DeepFullyConnected.o \
                   build/DeepLearning/Api/ExampleNetworks/FewLayerFullyConnected.o \
                   build/DeepLearning/Api/ExampleNetworks/SingleLayerFullyConnected.o \
                   build/DeepLearning/Api/ExampleNetworks/SingleLayerConvolution2d.o \
                   build/DeepLearning/Api/ExampleNetworks/InceptionV3.o \



ALL_DEMOS =	build/Demos/AlexNetDemo \
			build/Demos/FewLayerFullyConnectedDemo \
			build/Demos/SingleLayerFullyConnectedDemo \
			build/Demos/SingleLayerConvolution2dDemo

# FIXME: .so
ML_DEV = libThor.a Thor.h




# Overall make targets

all: $(ML_DEV)
	$(MAKE) $(ALL_TESTS)
	$(MAKE) $(ALL_DEMOS)
	$(RUN_ALL_TESTS)
	@echo ""
	@echo ""
	@echo "Tests Passed"
	@echo "Build Succeeded"
	@echo ""


test: $(ML_DEV)
	$(MAKE) $(ALL_TESTS)
	$(RUN_ALL_TESTS)
	@echo ""
	@echo ""
	@echo "Tests Passed"
	@echo "Build Succeeded"
	@echo ""


build: $(ML_DEV)
	@echo ""
	@echo ""
	@echo "Build Succeeded, no tests run"
	@echo ""


clean:
	rm -rf build
	rm -f libThor.a
	rm -f Thor.h
	cd googletest && cmake ./
	$(MAKE) clean -C googletest


softclean:
	rm -rf build
	rm -f libThor.a
	rm -f Thor.h


formatAll:
	./formatAll


# Library

libThor.a: Thor.h
	git config core.hooksPath .githooks
	$(MAKE) $(ALL_OBJECT_FILES)
	ar rcs libThor.a $(ALL_OBJECT_FILES)

.PHONY: build/headerlist.txt

build/headerlist.txt:
	mkdir -p build
	find ./ -name "*.h" | grep -v ./googletest/ | grep -v ./test/ > build/headerlist.txt

build/buildTools/createMasterHeader: buildTools/createMasterHeader.cpp
	mkdir -p build
	mkdir -p build/buildTools
	$(Gpp) -o build/buildTools/createMasterHeader -std=c++11 buildTools/createMasterHeader.cpp

Thor.h: build/headerlist.txt build/buildTools/createMasterHeader
	build/buildTools/createMasterHeader build/headerlist.txt


# Object Files

build/Utilities/TensorOperations/GpuMatrixTranspose/gpuMatrixTransposeKernels.o: Utilities/TensorOperations/GpuMatrixTranspose/gpuMatrixTranspose.h Utilities/TensorOperations/GpuMatrixTranspose/gpuMatrixTransposeKernels.cu
	mkdir -p build/Utilities/TensorOperations/GpuMatrixTranspose
	$(Nvcc) -ccbin g++ -o build/Utilities/TensorOperations/GpuMatrixTranspose/gpuMatrixTransposeKernels.o -c -std=c++11 $(COMPUTE_CAPABILITIES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/GpuMatrixTranspose/gpuMatrixTransposeKernels.cu

build/Utilities/TensorOperations/TypeConversions/TypeConverterKernels.o: Utilities/TensorOperations/TypeConversions/TypeConverter.h Utilities/TensorOperations/TypeConversions/TypeConverterKernels.cu
	mkdir -p build/Utilities/TensorOperations/TypeConversions
	$(Nvcc) -ccbin g++ -o build/Utilities/TensorOperations/TypeConversions/TypeConverterKernels.o -c -std=c++11 $(COMPUTE_CAPABILITIES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/TypeConversions/TypeConverterKernels.cu

build/DeepLearning/Implementation/Tensor/TensorArithmeticKernels.o: DeepLearning/Implementation/Tensor/Tensor.h DeepLearning/Implementation/Tensor/TensorArithmeticKernels.cu
	mkdir -p build/DeepLearning/Implementation/Tensor
	$(Nvcc) -ccbin g++ -o build/DeepLearning/Implementation/Tensor/TensorArithmeticKernels.o -c -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v DeepLearning/Implementation/Tensor/TensorArithmeticKernels.cu

build/DeepLearning/Implementation/Tensor/TensorMathKernels.o: DeepLearning/Implementation/Tensor/Tensor.h DeepLearning/Implementation/Tensor/TensorMathKernels.cu
	mkdir -p build/DeepLearning/Implementation/Tensor
	$(Nvcc) -ccbin g++ -o build/DeepLearning/Implementation/Tensor/TensorMathKernels.o -c -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v DeepLearning/Implementation/Tensor/TensorMathKernels.cu

build/DeepLearning/Implementation/Tensor/TensorTrigonometryKernels.o: DeepLearning/Implementation/Tensor/Tensor.h DeepLearning/Implementation/Tensor/TensorTrigonometryKernels.cu
	mkdir -p build/DeepLearning/Implementation/Tensor
	$(Nvcc) -ccbin g++ -o build/DeepLearning/Implementation/Tensor/TensorTrigonometryKernels.o -c -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v DeepLearning/Implementation/Tensor/TensorTrigonometryKernels.cu

build/DeepLearning/Implementation/Tensor/TensorHyperbolicTrigonometryKernels.o: DeepLearning/Implementation/Tensor/Tensor.h DeepLearning/Implementation/Tensor/TensorHyperbolicTrigonometryKernels.cu
	mkdir -p build/DeepLearning/Implementation/Tensor
	$(Nvcc) -ccbin g++ -o build/DeepLearning/Implementation/Tensor/TensorHyperbolicTrigonometryKernels.o -c -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v DeepLearning/Implementation/Tensor/TensorHyperbolicTrigonometryKernels.cu

build/Utilities/TensorOperations/Activation/Tanh.o: Utilities/TensorOperations/Activation/Tanh.h Utilities/TensorOperations/Activation/Tanh.cu
	mkdir -p build/Utilities/TensorOperations/Activation
	$(Nvcc) -ccbin g++ -o build/Utilities/TensorOperations/Activation/Tanh.o -c --maxrregcount 128 -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/Activation/Tanh.cu

build/Utilities/TensorOperations/Activation/Relu.o: Utilities/TensorOperations/Activation/Relu.h Utilities/TensorOperations/Activation/Relu.cu
	mkdir -p build/Utilities/TensorOperations/Activation
	$(Nvcc) -ccbin g++ -o build/Utilities/TensorOperations/Activation/Relu.o -c --maxrregcount 128 -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/Activation/Relu.cu

build/Utilities/TensorOperations/Activation/Swish.o: Utilities/TensorOperations/Activation/Swish.h Utilities/TensorOperations/Activation/Swish.cu
	mkdir -p build/Utilities/TensorOperations/Activation
	$(Nvcc) -ccbin g++ -o build/Utilities/TensorOperations/Activation/Swish.o -c --maxrregcount 128 -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/Activation/Swish.cu

build/Utilities/TensorOperations/Activation/Selu.o: Utilities/TensorOperations/Activation/Selu.h Utilities/TensorOperations/Activation/Selu.cu
	mkdir -p build/Utilities/TensorOperations/Activation
	$(Nvcc) -ccbin g++ -o build/Utilities/TensorOperations/Activation/Selu.o -c --maxrregcount 128 -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/Activation/Selu.cu

build/Utilities/TensorOperations/Activation/HardSigmoid.o: Utilities/TensorOperations/Activation/HardSigmoid.h Utilities/TensorOperations/Activation/HardSigmoid.cu
	mkdir -p build/Utilities/TensorOperations/Activation
	$(Nvcc) -ccbin g++ -o build/Utilities/TensorOperations/Activation/HardSigmoid.o -c --maxrregcount 128 -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/Activation/HardSigmoid.cu

build/Utilities/TensorOperations/Activation/Gelu.o: Utilities/TensorOperations/Activation/Gelu.h Utilities/TensorOperations/Activation/Gelu.cu
	mkdir -p build/Utilities/TensorOperations/Activation
	$(Nvcc) -ccbin g++ -o build/Utilities/TensorOperations/Activation/Gelu.o -c --maxrregcount 128 -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/Activation/Gelu.cu

build/Utilities/TensorOperations/Activation/SoftSign.o: Utilities/TensorOperations/Activation/SoftSign.h Utilities/TensorOperations/Activation/SoftSign.cu
	mkdir -p build/Utilities/TensorOperations/Activation
	$(Nvcc) -ccbin g++ -o build/Utilities/TensorOperations/Activation/SoftSign.o -c --maxrregcount 128 -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/Activation/SoftSign.cu

build/Utilities/TensorOperations/Activation/SoftPlus.o: Utilities/TensorOperations/Activation/SoftPlus.h Utilities/TensorOperations/Activation/SoftPlus.cu
	mkdir -p build/Utilities/TensorOperations/Activation
	$(Nvcc) -ccbin g++ -o build/Utilities/TensorOperations/Activation/SoftPlus.o -c --maxrregcount 128 -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/Activation/SoftPlus.cu

build/Utilities/TensorOperations/Activation/Exponential.o: Utilities/TensorOperations/Activation/Exponential.h Utilities/TensorOperations/Activation/Exponential.cu
	mkdir -p build/Utilities/TensorOperations/Activation
	$(Nvcc) -ccbin g++ -o build/Utilities/TensorOperations/Activation/Exponential.o -c --maxrregcount 128 -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/Activation/Exponential.cu

build/Utilities/TensorOperations/Activation/Elu.o: Utilities/TensorOperations/Activation/Elu.h Utilities/TensorOperations/Activation/Elu.cu
	mkdir -p build/Utilities/TensorOperations/Activation
	$(Nvcc) -ccbin g++ -o build/Utilities/TensorOperations/Activation/Elu.o -c --maxrregcount 128 -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/Activation/Elu.cu

build/Utilities/TensorOperations/Activation/Sigmoid.o: Utilities/TensorOperations/Activation/Sigmoid.h Utilities/TensorOperations/Activation/Sigmoid.cu
	mkdir -p build/Utilities/TensorOperations/Activation
	$(Nvcc) -ccbin g++ -o build/Utilities/TensorOperations/Activation/Sigmoid.o -c --maxrregcount 128 -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/Activation/Sigmoid.cu

build/Utilities/TensorOperations/Arithmetic/Average.o: Utilities/TensorOperations/Arithmetic/Average.h Utilities/TensorOperations/Arithmetic/Average.cu
	mkdir -p build/Utilities/TensorOperations/Arithmetic
	$(Nvcc) -ccbin g++ -o build/Utilities/TensorOperations/Arithmetic/Average.o -c --maxrregcount 128 -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/Arithmetic/Average.cu

build/Utilities/TensorOperations/Arithmetic/Sum.o: Utilities/TensorOperations/Arithmetic/Sum.h Utilities/TensorOperations/Arithmetic/Sum.cu
	mkdir -p build/Utilities/TensorOperations/Arithmetic
	$(Nvcc) -ccbin g++ -o build/Utilities/TensorOperations/Arithmetic/Sum.o -c --maxrregcount 128 -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/Arithmetic/Sum.cu

build/Utilities/TensorOperations/Loss/MeanSquaredError.o: Utilities/TensorOperations/Loss/MeanSquaredError.h Utilities/TensorOperations/Loss/MeanSquaredError.cu
	mkdir -p build/Utilities/TensorOperations/Loss
	$(Nvcc) -ccbin g++ -o build/Utilities/TensorOperations/Loss/MeanSquaredError.o -c --maxrregcount 128 -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/Loss/MeanSquaredError.cu

build/Utilities/TensorOperations/Loss/MeanAbsoluteError.o: Utilities/TensorOperations/Loss/MeanAbsoluteError.h Utilities/TensorOperations/Loss/MeanAbsoluteError.cu
	mkdir -p build/Utilities/TensorOperations/Loss
	$(Nvcc) -ccbin g++ -o build/Utilities/TensorOperations/Loss/MeanAbsoluteError.o -c --maxrregcount 128 -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/Loss/MeanAbsoluteError.cu

build/Utilities/TensorOperations/Loss/MeanAbsolutePercentageError.o: Utilities/TensorOperations/Loss/MeanAbsolutePercentageError.h Utilities/TensorOperations/Loss/MeanAbsolutePercentageError.cu
	mkdir -p build/Utilities/TensorOperations/Loss
	$(Nvcc) -ccbin g++ -o build/Utilities/TensorOperations/Loss/MeanAbsolutePercentageError.o -c --maxrregcount 128 -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/Loss/MeanAbsolutePercentageError.cu

build/Utilities/TensorOperations/DeepLearning/Add1dBias.o: Utilities/TensorOperations/DeepLearning/Add1dBias.h Utilities/TensorOperations/DeepLearning/Add1dBias.cu
	mkdir -p build/Utilities/TensorOperations/DeepLearning
	$(Nvcc) -ccbin g++ -o build/Utilities/TensorOperations/DeepLearning/Add1dBias.o -c --maxrregcount 128 -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/DeepLearning/Add1dBias.cu

build/Utilities/TensorOperations/Arithmetic/SumScale.o: Utilities/TensorOperations/Arithmetic/SumScale.h Utilities/TensorOperations/Arithmetic/SumScale.cu
	mkdir -p build/Utilities/TensorOperations/Arithmetic
	$(Nvcc) -ccbin g++ -o build/Utilities/TensorOperations/Arithmetic/SumScale.o -c --maxrregcount 128 -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/Arithmetic/SumScale.cu

build/Utilities/TensorOperations/Arithmetic/SumManyToOne.o: Utilities/TensorOperations/Arithmetic/SumManyToOne.h Utilities/TensorOperations/Arithmetic/SumManyToOne.cu
	mkdir -p build/Utilities/TensorOperations/Arithmetic
	$(Nvcc) -ccbin g++ -o build/Utilities/TensorOperations/Arithmetic/SumManyToOne.o -c --maxrregcount 128 -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/Arithmetic/SumManyToOne.cu

build/Utilities/TensorOperations/Misc/ComputeCategoricalAccuracy.o: Utilities/TensorOperations/Misc/ComputeCategoricalAccuracy.h Utilities/TensorOperations/Misc/ComputeCategoricalAccuracy.cu
	mkdir -p build/Utilities/TensorOperations/Misc
	$(Nvcc) -ccbin g++ -o build/Utilities/TensorOperations/Misc/ComputeCategoricalAccuracy.o -c --maxrregcount 128 -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/Misc/ComputeCategoricalAccuracy.cu

build/Utilities/TensorOperations/Misc/ComputeBinaryAccuracy.o: Utilities/TensorOperations/Misc/ComputeBinaryAccuracy.h Utilities/TensorOperations/Misc/ComputeBinaryAccuracy.cu
	mkdir -p build/Utilities/TensorOperations/Misc
	$(Nvcc) -ccbin g++ -o build/Utilities/TensorOperations/Misc/ComputeBinaryAccuracy.o -c --maxrregcount 128 -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/Misc/ComputeBinaryAccuracy.cu

build/Utilities/TensorOperations/Arithmetic/ElementwiseSubtract.o: Utilities/TensorOperations/Arithmetic/ElementwiseSubtract.h Utilities/TensorOperations/Arithmetic/ElementwiseSubtract.cu
	mkdir -p build/Utilities/TensorOperations/Arithmetic
	$(Nvcc) -ccbin g++ -o build/Utilities/TensorOperations/Arithmetic/ElementwiseSubtract.o -c --maxrregcount 128 -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/Arithmetic/ElementwiseSubtract.cu

build/Utilities/TensorOperations/Arithmetic/MultiplyByScalar.o: Utilities/TensorOperations/Arithmetic/MultiplyByScalar.h Utilities/TensorOperations/Arithmetic/MultiplyByScalar.cu
	mkdir -p build/Utilities/TensorOperations/Arithmetic
	$(Nvcc) -ccbin g++ -o build/Utilities/TensorOperations/Arithmetic/MultiplyByScalar.o -c --maxrregcount 128 -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/Arithmetic/MultiplyByScalar.cu

build/Utilities/TensorOperations/Loss/CrossEntropyLoss.o: Utilities/TensorOperations/Loss/CrossEntropyLoss.h Utilities/TensorOperations/Loss/CrossEntropyLoss.cu
	mkdir -p build/Utilities/TensorOperations/Loss
	$(Nvcc) -ccbin g++ -o build/Utilities/TensorOperations/Loss/CrossEntropyLoss.o -c --maxrregcount 128 -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/Loss/CrossEntropyLoss.cu

build/Utilities/TensorOperations/Loss/CategoricalCrossEntropyLoss.o: Utilities/TensorOperations/Loss/CategoricalCrossEntropyLoss.h Utilities/TensorOperations/Loss/CategoricalCrossEntropyLoss.cu
	mkdir -p build/Utilities/TensorOperations/Loss
	$(Nvcc) -ccbin g++ -o build/Utilities/TensorOperations/Loss/CategoricalCrossEntropyLoss.o -c --maxrregcount 128 -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/Loss/CategoricalCrossEntropyLoss.cu

build/Utilities/TensorOperations/Loss/BinaryCrossEntropyLoss.o: Utilities/TensorOperations/Loss/BinaryCrossEntropyLoss.h Utilities/TensorOperations/Loss/BinaryCrossEntropyLoss.cu
	mkdir -p build/Utilities/TensorOperations/Loss
	$(Nvcc) -ccbin g++ -o build/Utilities/TensorOperations/Loss/BinaryCrossEntropyLoss.o -c --maxrregcount 128 -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/Loss/BinaryCrossEntropyLoss.cu

build/Utilities/TensorOperations/Arithmetic/Exponentiation.o: Utilities/TensorOperations/Arithmetic/Exponentiation.h Utilities/TensorOperations/Arithmetic/Exponentiation.cu
	mkdir -p build/Utilities/TensorOperations/Arithmetic
	$(Nvcc) -ccbin g++ -o build/Utilities/TensorOperations/Arithmetic/Exponentiation.o -c --maxrregcount 128 -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/Arithmetic/Exponentiation.cu

build/Utilities/TensorOperations/Misc/Map.o: Utilities/TensorOperations/Misc/Map.h Utilities/TensorOperations/Misc/Map.cu
	mkdir -p build/Utilities/TensorOperations/Misc
	$(Nvcc) -ccbin g++ -o build/Utilities/TensorOperations/Misc/Map.o -c --maxrregcount 64 -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/Misc/Map.cu

build/Utilities/TensorOperations/Misc/Split.o: Utilities/TensorOperations/Misc/Split.h Utilities/TensorOperations/Misc/Split.cu
	mkdir -p build/Utilities/TensorOperations/Misc
	$(Nvcc) -ccbin g++ -o build/Utilities/TensorOperations/Misc/Split.o -c --maxrregcount 64 -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/Misc/Split.cu

build/Utilities/TensorOperations/Misc/Pad.o: Utilities/TensorOperations/Misc/Pad.h Utilities/TensorOperations/Misc/Pad.cu
	mkdir -p build/Utilities/TensorOperations/Misc
	$(Nvcc) -ccbin g++ -o build/Utilities/TensorOperations/Misc/Pad.o -c --maxrregcount 64 -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/Misc/Pad.cu

build/Utilities/TensorOperations/Optimizers/Adam.o: Utilities/TensorOperations/Optimizers/Adam.h Utilities/TensorOperations/Optimizers/Adam.cu
	mkdir -p build/Utilities/TensorOperations/Optimizers
	$(Nvcc) -ccbin g++ -o build/Utilities/TensorOperations/Optimizers/Adam.o -c --maxrregcount 64 -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/Optimizers/Adam.cu

build/Utilities/TensorOperations/Misc/Extract.o: Utilities/TensorOperations/Misc/Extract.h Utilities/TensorOperations/Misc/Extract.cu
	mkdir -p build/Utilities/TensorOperations/Misc
	$(Nvcc) -ccbin g++ -o build/Utilities/TensorOperations/Misc/Extract.o -c --maxrregcount 64 -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/Misc/Extract.cu

build/Utilities/TensorOperations/Misc/Concatenate.o: Utilities/TensorOperations/Misc/Concatenate.h Utilities/TensorOperations/Misc/Concatenate.cu
	mkdir -p build/Utilities/TensorOperations/Misc
	$(Nvcc) -ccbin g++ -o build/Utilities/TensorOperations/Misc/Concatenate.o -c --maxrregcount 64 -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/Misc/Concatenate.cu

build/Utilities/TensorOperations/TypeConversions/TypeConverter.o: Utilities/TensorOperations/TypeConversions/TypeConverter.h Utilities/TensorOperations/TypeConversions/TypeConverter.cpp
	mkdir -p build/Utilities/TensorOperations/TypeConversions
	$(Gpp) -c -std=c++11 Utilities/TensorOperations/TypeConversions/TypeConverter.cpp $(CUDA) $(INCLUDE_DIRS) -o build/Utilities/TensorOperations/TypeConversions/TypeConverter.o

build/Utilities/ComputeTopology/MachineEvaluator.o: Utilities/ComputeTopology/MachineEvaluator.h Utilities/ComputeTopology/MachineEvaluator.cpp
	mkdir -p build/Utilities/ComputeTopology
	$(Gpp) -c -std=c++11 Utilities/ComputeTopology/MachineEvaluator.cpp $(CUDA) $(INCLUDE_DIRS) -o build/Utilities/ComputeTopology/MachineEvaluator.o

build/DeepLearning/Implementation/Tensor/Tensor.o: DeepLearning/Implementation/Tensor/Tensor.h DeepLearning/Implementation/Tensor/Tensor.cpp DeepLearning/Implementation/Tensor/TensorDescriptor.h DeepLearning/Implementation/Tensor/TensorPlacement.h
	mkdir -p build/DeepLearning/Implementation/Tensor
	$(Gpp) -c -I./ -std=c++11 DeepLearning/Implementation/Tensor/Tensor.cpp $(CUDA) $(INCLUDE_DIRS) -o build/DeepLearning/Implementation/Tensor/Tensor.o

build/DeepLearning/Implementation/Layers/Layer.o:  DeepLearning/Implementation/Layers/Layer.h DeepLearning/Implementation/Layers/NeuralNetwork/DropOut.h
	mkdir -p build/DeepLearning/Implementation/Layers/Layer
	$(Gpp) -c -I./ -std=c++11 DeepLearning/Implementation/Layers/Layer.cpp $(CUDA) $(INCLUDE_DIRS) -o build/DeepLearning/Implementation/Layers/Layer.o

build/Utilities/TensorOperations/GpuConvolution/GpuConvolution.o: Utilities/TensorOperations/GpuConvolution/GpuConvolution.h Utilities/TensorOperations/GpuConvolution/GpuConvolution.cpp
	mkdir -p build/Utilities/TensorOperations/GpuConvolution
	$(Gpp) -c -std=c++11 Utilities/TensorOperations/GpuConvolution/GpuConvolution.cpp $(CUDA) $(INCLUDE_DIRS) -o build/Utilities/TensorOperations/GpuConvolution/GpuConvolution.o

build/Utilities/TensorOperations/GpuConvolution/GpuConvolutionKernels.o: Utilities/TensorOperations/GpuConvolution/GpuConvolution.h Utilities/TensorOperations/GpuConvolution/GpuConvolutionKernels.cu
	mkdir -p build/Utilities/TensorOperations/GpuConvolution
	$(Nvcc) -ccbin g++ -o build/Utilities/TensorOperations/GpuConvolution/GpuConvolutionKernels.o -c --maxrregcount 64 -std=c++11 $(COMPUTE_CAPABILITIES_WITH_TENSOR_CORES) $(INCLUDE_DIRS) -Xptxas -O3,-v Utilities/TensorOperations/GpuConvolution/GpuConvolutionKernels.cu

build/DeepLearning/Implementation/Layers/NeuralNetwork/FullyConnected.o: DeepLearning/Implementation/Layers/NeuralNetwork/FullyConnected.h DeepLearning/Implementation/Layers/NeuralNetwork/FullyConnected.cpp
	mkdir -p build/DeepLearning/Implementation/Layers/NeuralNetwork
	$(Gpp) -c -std=c++11 DeepLearning/Implementation/Layers/NeuralNetwork/FullyConnected.cpp $(CUDA) $(INCLUDE_DIRS) -o build/DeepLearning/Implementation/Layers/NeuralNetwork/FullyConnected.o

build/DeepLearning/Implementation/Layers/Loss/LossShaper.o: DeepLearning/Implementation/Layers/Loss/LossShaper.h DeepLearning/Implementation/Layers/Loss/LossShaper.cpp
	mkdir -p build/DeepLearning/Implementation/Layers/Loss
	$(Gpp) -c -std=c++11 DeepLearning/Implementation/Layers/Loss/LossShaper.cpp $(CUDA) $(INCLUDE_DIRS) -o build/DeepLearning/Implementation/Layers/Loss/LossShaper.o

build/DeepLearning/Implementation/Layers/Loss/MeanSquaredError.o: DeepLearning/Implementation/Layers/Loss/MeanSquaredError.h DeepLearning/Implementation/Layers/Loss/MeanSquaredError.cpp
	mkdir -p build/DeepLearning/Implementation/Layers/Loss
	$(Gpp) -c -std=c++11 DeepLearning/Implementation/Layers/Loss/MeanSquaredError.cpp $(CUDA) $(INCLUDE_DIRS) -o build/DeepLearning/Implementation/Layers/Loss/MeanSquaredError.o

build/DeepLearning/Implementation/Layers/Loss/MeanAbsoluteError.o: DeepLearning/Implementation/Layers/Loss/MeanAbsoluteError.h DeepLearning/Implementation/Layers/Loss/MeanAbsoluteError.cpp
	mkdir -p build/DeepLearning/Implementation/Layers/Loss
	$(Gpp) -c -std=c++11 DeepLearning/Implementation/Layers/Loss/MeanAbsoluteError.cpp $(CUDA) $(INCLUDE_DIRS) -o build/DeepLearning/Implementation/Layers/Loss/MeanAbsoluteError.o

build/DeepLearning/Implementation/Layers/Loss/MeanAbsolutePercentageError.o: DeepLearning/Implementation/Layers/Loss/MeanAbsolutePercentageError.h DeepLearning/Implementation/Layers/Loss/MeanAbsolutePercentageError.cpp
	mkdir -p build/DeepLearning/Implementation/Layers/Loss
	$(Gpp) -c -std=c++11 DeepLearning/Implementation/Layers/Loss/MeanAbsolutePercentageError.cpp $(CUDA) $(INCLUDE_DIRS) -o build/DeepLearning/Implementation/Layers/Loss/MeanAbsolutePercentageError.o

build/DeepLearning/Implementation/Layers/Loss/CrossEntropy.o: DeepLearning/Implementation/Layers/Loss/CrossEntropy.h DeepLearning/Implementation/Layers/Loss/CrossEntropy.cpp
	mkdir -p build/DeepLearning/Implementation/Layers/Loss
	$(Gpp) -c -std=c++11 DeepLearning/Implementation/Layers/Loss/CrossEntropy.cpp $(CUDA) $(INCLUDE_DIRS) -o build/DeepLearning/Implementation/Layers/Loss/CrossEntropy.o

build/Utilities/TensorOperations/GpuMatrixMultiply/CublasKernel.o: Utilities/TensorOperations/GpuMatrixMultiply/CublasKernel.h Utilities/TensorOperations/GpuMatrixMultiply/CublasKernel.cpp
	mkdir -p build/Utilities/TensorOperations/GpuMatrixMultiply
	$(Gpp) -c -std=c++11 Utilities/TensorOperations/GpuMatrixMultiply/CublasKernel.cpp $(CUDA) $(INCLUDE_DIRS) -o build/Utilities/TensorOperations/GpuMatrixMultiply/CublasKernel.o

build/Utilities/Common/ReferenceCounted.o: Utilities/Common/ReferenceCounted.h Utilities/Common/ReferenceCounted.cpp
	mkdir -p build/Utilities/Common
	$(Gpp) -c -std=c++11 Utilities/Common/ReferenceCounted.cpp $(CUDA) $(INCLUDE_DIRS) -o build/Utilities/Common/ReferenceCounted.o

build/Utilities/Common/ThreadJoinQueue.o: Utilities/Common/ThreadJoinQueue.h Utilities/Common/ThreadJoinQueue.cpp
	mkdir -p build/Utilities/Common
	$(Gpp) -c -std=c++11 Utilities/Common/ThreadJoinQueue.cpp $(CUDA) $(INCLUDE_DIRS) -o build/Utilities/Common/ThreadJoinQueue.o

build/Utilities/Common/CudnnHelper.o: Utilities/Common/CudnnHelper.h Utilities/Common/CudnnHelper.cpp
	mkdir -p build/Utilities/Common
	$(Gpp) -c -std=c++11 Utilities/Common/CudnnHelper.cpp $(CUDA) $(INCLUDE_DIRS) -o build/Utilities/Common/CudnnHelper.o

build/Utilities/TensorOperations/GpuMatrixMultiply/CublasMatrixMultiply.o: Utilities/TensorOperations/GpuMatrixMultiply/CublasMatrixMultiply.h Utilities/TensorOperations/GpuMatrixMultiply/CublasMatrixMultiply.cpp
	mkdir -p build/Utilities/TensorOperations/GpuMatrixMultiply
	$(Gpp) -c -std=c++11 Utilities/TensorOperations/GpuMatrixMultiply/CublasMatrixMultiply.cpp $(CUDA) $(INCLUDE_DIRS) -o build/Utilities/TensorOperations/GpuMatrixMultiply/CublasMatrixMultiply.o

build/DeepLearning/Implementation/Layers/NeuralNetwork/Pooling.o: DeepLearning/Implementation/Layers/NeuralNetwork/Pooling.h DeepLearning/Implementation/Layers/NeuralNetwork/Pooling.cpp
	mkdir -p build/DeepLearning/Implementation/Layers/NeuralNetwork
	$(Gpp) -c -std=c++11 DeepLearning/Implementation/Layers/NeuralNetwork/Pooling.cpp $(CUDA) $(INCLUDE_DIRS) -o build/DeepLearning/Implementation/Layers/NeuralNetwork/Pooling.o

build/DeepLearning/Implementation/Layers/NeuralNetwork/BatchNormalization.o: DeepLearning/Implementation/Layers/NeuralNetwork/BatchNormalization.h DeepLearning/Implementation/Layers/NeuralNetwork/BatchNormalization.cpp
	mkdir -p build/DeepLearning/Implementation/Layers/NeuralNetwork
	$(Gpp) -c -std=c++11 DeepLearning/Implementation/Layers/NeuralNetwork/BatchNormalization.cpp $(CUDA) $(INCLUDE_DIRS) -o build/DeepLearning/Implementation/Layers/NeuralNetwork/BatchNormalization.o

build/DeepLearning/Implementation/Layers/Optimizers/Optimizer.o: DeepLearning/Implementation/Layers/Optimizers/Optimizer.h DeepLearning/Implementation/Layers/Optimizers/Optimizer.cpp
	mkdir -p build/DeepLearning/Implementation/Layers/Optimizers
	$(Gpp) -c -std=c++11 DeepLearning/Implementation/Layers/Optimizers/Optimizer.cpp $(CUDA) $(INCLUDE_DIRS) -o build/DeepLearning/Implementation/Layers/Optimizers/Optimizer.o

build/DeepLearning/Implementation/Layers/Optimizers/Sgd.o: DeepLearning/Implementation/Layers/Optimizers/Sgd.h DeepLearning/Implementation/Layers/Optimizers/Sgd.cpp
	mkdir -p build/DeepLearning/Implementation/Layers/Optimizers
	$(Gpp) -c -std=c++11 DeepLearning/Implementation/Layers/Optimizers/Sgd.cpp $(CUDA) $(INCLUDE_DIRS) -o build/DeepLearning/Implementation/Layers/Optimizers/Sgd.o

build/DeepLearning/Implementation/Layers/Optimizers/Adam.o: DeepLearning/Implementation/Layers/Optimizers/Adam.h DeepLearning/Implementation/Layers/Optimizers/Adam.cpp
	mkdir -p build/DeepLearning/Implementation/Layers/Optimizers
	$(Gpp) -c -std=c++11 DeepLearning/Implementation/Layers/Optimizers/Adam.cpp $(CUDA) $(INCLUDE_DIRS) -o build/DeepLearning/Implementation/Layers/Optimizers/Adam.o

build/DeepLearning/Implementation/Layers/Activation/Softmax.o: DeepLearning/Implementation/Layers/Activation/Softmax.h DeepLearning/Implementation/Layers/Activation/Softmax.cpp
	mkdir -p build/DeepLearning/Implementation/Layers/Activation
	$(Gpp) -c -std=c++11 DeepLearning/Implementation/Layers/Activation/Softmax.cpp $(CUDA) $(INCLUDE_DIRS) -o build/DeepLearning/Implementation/Layers/Activation/Softmax.o

build/DeepLearning/Implementation/Initializers/Initializer.o: DeepLearning/Implementation/Initializers/Initializer.h DeepLearning/Implementation/Initializers/Initializer.cpp
	mkdir -p build/DeepLearning/Implementation/Initializers
	$(Gpp) -c -std=c++11 DeepLearning/Implementation/Initializers/Initializer.cpp $(CUDA) $(INCLUDE_DIRS) -o build/DeepLearning/Implementation/Initializers/Initializer.o

build/DeepLearning/Implementation/Initializers/UniformRandom.o: DeepLearning/Implementation/Initializers/UniformRandom.h DeepLearning/Implementation/Initializers/UniformRandom.cpp
	mkdir -p build/DeepLearning/Implementation/Initializers
	$(Gpp) -c -std=c++11 DeepLearning/Implementation/Initializers/UniformRandom.cpp $(CUDA) $(INCLUDE_DIRS) -o build/DeepLearning/Implementation/Initializers/UniformRandom.o

build/DeepLearning/Implementation/Initializers/Glorot.o: DeepLearning/Implementation/Initializers/Glorot.h DeepLearning/Implementation/Initializers/Glorot.cpp
	mkdir -p build/DeepLearning/Implementation/Initializers
	$(Gpp) -c -std=c++11 DeepLearning/Implementation/Initializers/Glorot.cpp $(CUDA) $(INCLUDE_DIRS) -o build/DeepLearning/Implementation/Initializers/Glorot.o

build/DeepLearning/Api/Visualizers/ConsoleVisualizer.o: DeepLearning/Api/Visualizers/ConsoleVisualizer.h DeepLearning/Api/Visualizers/ConsoleVisualizer.cpp
	mkdir -p build/DeepLearning/Api/Visualizers
	$(Gpp) -c -std=c++11 DeepLearning/Api/Visualizers/ConsoleVisualizer.cpp $(CUDA) $(INCLUDE_DIRS) -o build/DeepLearning/Api/Visualizers/ConsoleVisualizer.o

build/DeepLearning/Api/HyperparameterControllers/HyperparameterController.o: DeepLearning/Api/HyperparameterControllers/HyperparameterController.h DeepLearning/Api/HyperparameterControllers/HyperparameterController.cpp
	mkdir -p build/DeepLearning/Api/HyperparameterControllers
	$(Gpp) -c -std=c++11 DeepLearning/Api/HyperparameterControllers/HyperparameterController.cpp $(CUDA) $(INCLUDE_DIRS) -o build/DeepLearning/Api/HyperparameterControllers/HyperparameterController.o

build/DeepLearning/Api/Executors/LocalExecutor.o: DeepLearning/Api/Executors/LocalExecutor.h DeepLearning/Api/Executors/LocalExecutor.cpp
	mkdir -p build/DeepLearning/Api/Executors
	$(Gpp) -c -std=c++11 DeepLearning/Api/Executors/LocalExecutor.cpp $(CUDA) $(INCLUDE_DIRS) -o build/DeepLearning/Api/Executors/LocalExecutor.o

build/DeepLearning/Api/Network/Network.o: DeepLearning/Api/Network/Network.h DeepLearning/Api/Network/Network.cpp
	mkdir -p build/DeepLearning/Api/Network
	$(Gpp) -c -std=c++11 DeepLearning/Api/Network/Network.cpp $(CUDA) $(INCLUDE_DIRS) -o build/DeepLearning/Api/Network/Network.o

build/DeepLearning/Api/Optimizers/Optimizer.o: DeepLearning/Api/Optimizers/Optimizer.h DeepLearning/Api/Optimizers/Optimizer.cpp
	mkdir -p build/DeepLearning/Api/Optimizers/
	$(Gpp) -c -std=c++11 DeepLearning/Api/Optimizers/Optimizer.cpp $(CUDA) $(INCLUDE_DIRS) -o build/DeepLearning/Api/Optimizers/Optimizer.o

build/DeepLearning/Api/Optimizers/Sgd.o: DeepLearning/Api/Optimizers/Sgd.h DeepLearning/Api/Optimizers/Sgd.cpp
	mkdir -p build/DeepLearning/Api/Optimizers/
	$(Gpp) -c -std=c++11 DeepLearning/Api/Optimizers/Sgd.cpp $(CUDA) $(INCLUDE_DIRS) -o build/DeepLearning/Api/Optimizers/Sgd.o

build/DeepLearning/Api/Optimizers/Adam.o: DeepLearning/Api/Optimizers/Adam.h DeepLearning/Api/Optimizers/Adam.cpp
	mkdir -p build/DeepLearning/Api/Optimizers/
	$(Gpp) -c -std=c++11 DeepLearning/Api/Optimizers/Adam.cpp $(CUDA) $(INCLUDE_DIRS) -o build/DeepLearning/Api/Optimizers/Adam.o

build/Utilities/Common/Stream.o: Utilities/Common/Stream.h Utilities/Common/Stream.cpp
	mkdir -p build/Utilities/Common
	$(Gpp) -c -std=c++11 Utilities/Common/Stream.cpp $(CUDA) $(INCLUDE_DIRS) -o build/Utilities/Common/Stream.o

build/Utilities/Common/Event.o: Utilities/Common/Event.h Utilities/Common/Event.cpp
	mkdir -p build/Utilities/Common
	$(Gpp) -c -std=c++11 Utilities/Common/Event.cpp $(CUDA) $(INCLUDE_DIRS) -o build/Utilities/Common/Event.o

build/Utilities/Loaders/Shard.o: Utilities/Loaders/Shard.h Utilities/Loaders/Shard.cpp
	mkdir -p build/Utilities/Loaders
	$(Gpp) -Wno-error=maybe-uninitialized -c -std=c++11 Utilities/Loaders/Shard.cpp $(CUDA) $(INCLUDE_DIRS) -o build/Utilities/Loaders/Shard.o $(GRAPHICS_MAGICK)

build/Utilities/Loaders/ShardedRawDatasetCreator.o: Utilities/Loaders/ShardedRawDatasetCreator.h Utilities/Loaders/ShardedRawDatasetCreator.cpp
	mkdir -p build/Utilities/Loaders
	$(Gpp) -c -std=c++11 Utilities/Loaders/ShardedRawDatasetCreator.cpp $(CUDA) $(INCLUDE_DIRS) -o build/Utilities/Loaders/ShardedRawDatasetCreator.o $(GRAPHICS_MAGICK)

build/Utilities/Loaders/ImageLoader.o: Utilities/Loaders/ImageLoader.h Utilities/Loaders/ImageLoader.cpp
	mkdir -p build/Utilities/Loaders
	$(Gpp) -c -std=c++11 Utilities/Loaders/ImageLoader.cpp $(CUDA) $(INCLUDE_DIRS) -o build/Utilities/Loaders/ImageLoader.o

build/Utilities/TensorOperations/Misc/BatchReduce.o: Utilities/TensorOperations/Misc/BatchReduce.h Utilities/TensorOperations/Misc/BatchReduce.cpp
	mkdir -p build/Utilities/TensorOperations/Misc
	$(Gpp) -c -std=c++11 Utilities/TensorOperations/Misc/BatchReduce.cpp $(CUDA) $(INCLUDE_DIRS) -o build/Utilities/TensorOperations/Misc/BatchReduce.o

build/DeepLearning/Api/Layers/Loss/MeanSquaredError.o: DeepLearning/Api/Layers/Loss/MeanSquaredError.h DeepLearning/Api/Layers/Loss/MeanSquaredError.cpp
	mkdir -p build/DeepLearning/Api/Layers/Loss
	$(Gpp) -c -std=c++11 DeepLearning/Api/Layers/Loss/MeanSquaredError.cpp $(CUDA) $(INCLUDE_DIRS) -o build/DeepLearning/Api/Layers/Loss/MeanSquaredError.o

build/DeepLearning/Api/Layers/Loss/MeanAbsoluteError.o: DeepLearning/Api/Layers/Loss/MeanAbsoluteError.h DeepLearning/Api/Layers/Loss/MeanAbsoluteError.cpp
	mkdir -p build/DeepLearning/Api/Layers/Loss
	$(Gpp) -c -std=c++11 DeepLearning/Api/Layers/Loss/MeanAbsoluteError.cpp $(CUDA) $(INCLUDE_DIRS) -o build/DeepLearning/Api/Layers/Loss/MeanAbsoluteError.o

build/DeepLearning/Api/Layers/Loss/MeanAbsolutePercentageError.o: DeepLearning/Api/Layers/Loss/MeanAbsolutePercentageError.h DeepLearning/Api/Layers/Loss/MeanAbsolutePercentageError.cpp
	mkdir -p build/DeepLearning/Api/Layers/Loss
	$(Gpp) -c -std=c++11 DeepLearning/Api/Layers/Loss/MeanAbsolutePercentageError.cpp $(CUDA) $(INCLUDE_DIRS) -o build/DeepLearning/Api/Layers/Loss/MeanAbsolutePercentageError.o

build/DeepLearning/Api/Layers/Loss/CategoricalCrossEntropy.o: DeepLearning/Api/Layers/Loss/CategoricalCrossEntropy.h DeepLearning/Api/Layers/Loss/CategoricalCrossEntropy.cpp
	mkdir -p build/DeepLearning/Api/Layers/Loss
	$(Gpp) -c -std=c++11 DeepLearning/Api/Layers/Loss/CategoricalCrossEntropy.cpp $(CUDA) $(INCLUDE_DIRS) -o build/DeepLearning/Api/Layers/Loss/CategoricalCrossEntropy.o

build/DeepLearning/Api/Layers/Loss/BinaryCrossEntropy.o: DeepLearning/Api/Layers/Loss/BinaryCrossEntropy.h DeepLearning/Api/Layers/Loss/BinaryCrossEntropy.cpp
	mkdir -p build/DeepLearning/Api/Layers/Loss
	$(Gpp) -c -std=c++11 DeepLearning/Api/Layers/Loss/BinaryCrossEntropy.cpp $(CUDA) $(INCLUDE_DIRS) -o build/DeepLearning/Api/Layers/Loss/BinaryCrossEntropy.o

build/Utilities/Loaders/ImageProcessor.o: Utilities/Loaders/ImageProcessor.h Utilities/Loaders/ImageProcessor.cpp
	mkdir -p build/Utilities/Loaders
	$(Gpp) -c -std=c++11 Utilities/Loaders/ImageProcessor.cpp $(CUDA) $(INCLUDE_DIRS) -o build/Utilities/Loaders/ImageProcessor.o

build/Utilities/Loaders/NoOpDataProcessor.o: Utilities/Loaders/NoOpDataProcessor.h Utilities/Loaders/NoOpDataProcessor.cpp
	mkdir -p build/Utilities/Loaders
	$(Gpp) -c -std=c++11 Utilities/Loaders/NoOpDataProcessor.cpp $(CUDA) $(INCLUDE_DIRS) -o build/Utilities/Loaders/NoOpDataProcessor.o

build/Utilities/Loaders/BatchAssembler.o: Utilities/Loaders/BatchAssembler.h Utilities/Loaders/BatchAssembler.cpp
	mkdir -p build/Utilities/Loaders
	$(Gpp) -c -std=c++11 Utilities/Loaders/BatchAssembler.cpp $(CUDA) $(INCLUDE_DIRS) -o build/Utilities/Loaders/BatchAssembler.o

build/DeepLearning/Api/Loaders/LocalBatchLoader.o: DeepLearning/Api/Loaders/LocalBatchLoader.h DeepLearning/Api/Loaders/LocalBatchLoader.cpp
	mkdir -p build/DeepLearning/Api/Loaders/
	$(Gpp) -c -std=c++11 DeepLearning/Api/Loaders/LocalBatchLoader.cpp $(CUDA) $(INCLUDE_DIRS) -o build/DeepLearning/Api/Loaders/LocalBatchLoader.o

build/Utilities/WorkQueue/AsyncTensorQueue.o: Utilities/WorkQueue/AsyncTensorQueue.h Utilities/WorkQueue/AsyncTensorQueue.cpp
	mkdir -p build/Utilities/WorkQueue
	$(Gpp) -c -std=c++11 Utilities/WorkQueue/AsyncTensorQueue.cpp $(CUDA) $(INCLUDE_DIRS) -o build/Utilities/WorkQueue/AsyncTensorQueue.o

build/DeepLearning/Api/Tensor/Tensor.o: DeepLearning/Api/Tensor/Tensor.h DeepLearning/Api/Tensor/Tensor.cpp
	mkdir -p build/DeepLearning/Api/Tensor
	$(Gpp) -c -std=c++11 DeepLearning/Api/Tensor/Tensor.cpp $(CUDA) $(INCLUDE_DIRS) -o build/DeepLearning/Api/Tensor/Tensor.o

build/DeepLearning/Api/Layers/Layer.o: DeepLearning/Api/Layers/Layer.h DeepLearning/Api/Layers/Layer.cpp
	mkdir -p build/DeepLearning/Api/Layers
	$(Gpp) -c -std=c++11 DeepLearning/Api/Layers/Layer.cpp $(CUDA) $(INCLUDE_DIRS) -o build/DeepLearning/Api/Layers/Layer.o

build/DeepLearning/Api/Initializers/Initializer.o: DeepLearning/Api/Initializers/Initializer.h DeepLearning/Api/Initializers/Initializer.cpp
	mkdir -p build/DeepLearning/Api/Initializers
	$(Gpp) -c -std=c++11 DeepLearning/Api/Initializers/Initializer.cpp $(CUDA) $(INCLUDE_DIRS) -o build/DeepLearning/Api/Initializers/Initializer.o

build/DeepLearning/Api/Layers/Learning/FullyConnected.o: DeepLearning/Api/Layers/Learning/FullyConnected.h DeepLearning/Api/Layers/Learning/FullyConnected.cpp
	mkdir -p build/DeepLearning/Api/Layers/Learning
	$(Gpp) -c -std=c++11 DeepLearning/Api/Layers/Learning/FullyConnected.cpp $(CUDA) $(INCLUDE_DIRS) -o build/DeepLearning/Api/Layers/Learning/FullyConnected.o

build/DeepLearning/Api/Layers/Learning/Convolution2d.o: DeepLearning/Api/Layers/Learning/Convolution2d.h DeepLearning/Api/Layers/Learning/Convolution2d.cpp
	mkdir -p build/DeepLearning/Api/Layers/Learning
	$(Gpp) -c -std=c++11 DeepLearning/Api/Layers/Learning/Convolution2d.cpp $(CUDA) $(INCLUDE_DIRS) -o build/DeepLearning/Api/Layers/Learning/Convolution2d.o

build/DeepLearning/Api/Layers/Utility/BatchNormalization.o: DeepLearning/Api/Layers/Utility/BatchNormalization.h DeepLearning/Api/Layers/Utility/BatchNormalization.cpp
	mkdir -p build/DeepLearning/Api/Layers/Utility
	$(Gpp) -c -std=c++11 DeepLearning/Api/Layers/Utility/BatchNormalization.cpp $(CUDA) $(INCLUDE_DIRS) -o build/DeepLearning/Api/Layers/Utility/BatchNormalization.o

build/DeepLearning/Api/Layers/Utility/DropOut.o: DeepLearning/Api/Layers/Utility/DropOut.h DeepLearning/Api/Layers/Utility/DropOut.cpp
	mkdir -p build/DeepLearning/Api/Layers/Utility
	$(Gpp) -c -std=c++11 DeepLearning/Api/Layers/Utility/DropOut.cpp $(CUDA) $(INCLUDE_DIRS) -o build/DeepLearning/Api/Layers/Utility/DropOut.o

build/DeepLearning/Api/Layers/Utility/NetworkOutput.o: DeepLearning/Api/Layers/Utility/NetworkOutput.h DeepLearning/Api/Layers/Utility/NetworkOutput.cpp
	mkdir -p build/DeepLearning/Api/Layers/Utility
	$(Gpp) -c -std=c++11 DeepLearning/Api/Layers/Utility/NetworkOutput.cpp $(CUDA) $(INCLUDE_DIRS) -o build/DeepLearning/Api/Layers/Utility/NetworkOutput.o

build/DeepLearning/Api/Layers/Learning/Inception.o: DeepLearning/Api/Layers/Learning/Inception.h DeepLearning/Api/Layers/Learning/Inception.cpp
	mkdir -p build/DeepLearning/Api/Layers/Learning
	$(Gpp) -c -std=c++11 DeepLearning/Api/Layers/Learning/Inception.cpp $(CUDA) $(INCLUDE_DIRS) -o build/DeepLearning/Api/Layers/Learning/Inception.o

build/DeepLearning/Api/ExampleNetworks/AlexNet.o: DeepLearning/Api/ExampleNetworks/AlexNet.h DeepLearning/Api/ExampleNetworks/AlexNet.cpp
	mkdir -p build/DeepLearning/Api/ExampleNetworks
	$(Gpp) -c -std=c++11 DeepLearning/Api/ExampleNetworks/AlexNet.cpp $(CUDA) $(INCLUDE_DIRS) -o build/DeepLearning/Api/ExampleNetworks/AlexNet.o

build/DeepLearning/Api/ExampleNetworks/DeepFullyConnected.o: DeepLearning/Api/ExampleNetworks/DeepFullyConnected.h DeepLearning/Api/ExampleNetworks/DeepFullyConnected.cpp
	mkdir -p build/DeepLearning/Api/ExampleNetworks
	$(Gpp) -c -std=c++11 DeepLearning/Api/ExampleNetworks/DeepFullyConnected.cpp $(CUDA) $(INCLUDE_DIRS) -o build/DeepLearning/Api/ExampleNetworks/DeepFullyConnected.o

build/DeepLearning/Api/ExampleNetworks/FewLayerFullyConnected.o: DeepLearning/Api/ExampleNetworks/FewLayerFullyConnected.h DeepLearning/Api/ExampleNetworks/FewLayerFullyConnected.cpp
	mkdir -p build/DeepLearning/Api/ExampleNetworks
	$(Gpp) -c -std=c++11 DeepLearning/Api/ExampleNetworks/FewLayerFullyConnected.cpp $(CUDA) $(INCLUDE_DIRS) -o build/DeepLearning/Api/ExampleNetworks/FewLayerFullyConnected.o

build/DeepLearning/Api/ExampleNetworks/SingleLayerFullyConnected.o: DeepLearning/Api/ExampleNetworks/SingleLayerFullyConnected.h DeepLearning/Api/ExampleNetworks/SingleLayerFullyConnected.cpp
	mkdir -p build/DeepLearning/Api/ExampleNetworks
	$(Gpp) -c -std=c++11 DeepLearning/Api/ExampleNetworks/SingleLayerFullyConnected.cpp $(CUDA) $(INCLUDE_DIRS) -o build/DeepLearning/Api/ExampleNetworks/SingleLayerFullyConnected.o

build/DeepLearning/Api/ExampleNetworks/SingleLayerConvolution2d.o: DeepLearning/Api/ExampleNetworks/SingleLayerConvolution2d.h DeepLearning/Api/ExampleNetworks/SingleLayerConvolution2d.cpp
	mkdir -p build/DeepLearning/Api/ExampleNetworks
	$(Gpp) -c -std=c++11 DeepLearning/Api/ExampleNetworks/SingleLayerConvolution2d.cpp $(CUDA) $(INCLUDE_DIRS) -o build/DeepLearning/Api/ExampleNetworks/SingleLayerConvolution2d.o

build/DeepLearning/Api/ExampleNetworks/InceptionV3.o: DeepLearning/Api/ExampleNetworks/InceptionV3.h DeepLearning/Api/ExampleNetworks/InceptionV3.cpp
	mkdir -p build/DeepLearning/Api/ExampleNetworks
	$(Gpp) -c -std=c++11 DeepLearning/Api/ExampleNetworks/InceptionV3.cpp $(CUDA) $(INCLUDE_DIRS) -o build/DeepLearning/Api/ExampleNetworks/InceptionV3.o

build/Demos/AlexNetDemo: build/test/googletest/libgtest.a Demos/AlexNetDemo.cpp $(THOR)
	mkdir -p build/Demos
	$(Gpp) -o build/Demos/AlexNetDemo -std=c++11 -pthread Demos/AlexNetDemo.cpp $(CUDA) $(INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/Demos/KernelProfilerScaffold: build/test/googletest/libgtest.a Demos/KernelProfilerScaffold.cpp $(THOR)
	mkdir -p build/Demos
	$(Gpp) -o build/Demos/KernelProfilerScaffold -std=c++11 -pthread Demos/KernelProfilerScaffold.cpp $(CUDA) $(INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/Demos/FewLayerFullyConnectedDemo: build/test/googletest/libgtest.a Demos/FewLayerFullyConnectedDemo.cpp $(THOR)
	mkdir -p build/Demos
	$(Gpp) -o build/Demos/FewLayerFullyConnectedDemo -std=c++11 -pthread Demos/FewLayerFullyConnectedDemo.cpp $(CUDA) $(INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/Demos/SingleLayerFullyConnectedDemo: build/test/googletest/libgtest.a Demos/SingleLayerFullyConnectedDemo.cpp $(THOR)
	mkdir -p build/Demos
	$(Gpp) -o build/Demos/SingleLayerFullyConnectedDemo -std=c++11 -pthread Demos/SingleLayerFullyConnectedDemo.cpp $(CUDA) $(INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/Demos/SingleLayerConvolution2dDemo: build/test/googletest/libgtest.a Demos/SingleLayerConvolution2dDemo.cpp $(THOR)
	mkdir -p build/Demos
	$(Gpp) -o build/Demos/SingleLayerConvolution2dDemo -std=c++11 -pthread Demos/SingleLayerConvolution2dDemo.cpp $(CUDA) $(INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)


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

build/test/Utilities/WorkQueue/WorkQueueUnorderedTest: build/test/googletest/libgtest.a test/Utilities/WorkQueue/WorkQueueUnorderedTest.cpp $(THOR)
	mkdir -p build/test/Utilities/WorkQueue/
	$(Gpp) $(DEBUG) -o build/test/Utilities/WorkQueue/WorkQueueUnorderedTest -std=c++11 -pthread -IUtilities/WorkQueue/ test/Utilities/WorkQueue/WorkQueueUnorderedTest.cpp $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/Utilities/WorkQueue/WorkQueueTest: build/test/googletest/libgtest.a test/Utilities/WorkQueue/WorkQueueTest.cpp $(THOR)
	mkdir -p build/test/Utilities/WorkQueue/
	$(Gpp) $(DEBUG) -o build/test/Utilities/WorkQueue/WorkQueueTest -std=c++11 -pthread -IUtilities/WorkQueue/ test/Utilities/WorkQueue/WorkQueueTest.cpp $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/Utilities/TensorOperations/GpuMatrixTranspose/gpuMatrixTransposeTest: build/test/googletest/libgtest.a test/Utilities/TensorOperations/GpuMatrixTranspose/gpuMatrixTransposeTest.cpp $(THOR)
	mkdir -p build/test/Utilities/TensorOperations/GpuMatrixTranspose
	$(Gpp) -o build/test/Utilities/TensorOperations/GpuMatrixTranspose/gpuMatrixTransposeTest test/Utilities/TensorOperations/GpuMatrixTranspose/gpuMatrixTransposeTest.cpp -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/Utilities/TensorOperations/GpuMatrixMultiply/CublasMatrixMultiplyTest: build/test/googletest/libgtest.a test/Utilities/TensorOperations/GpuMatrixMultiply/CublasMatrixMultiplyTest.cpp $(THOR)
	mkdir -p build/test/Utilities/TensorOperations/GpuMatrixMultiply
	$(Gpp) -o build/test/Utilities/TensorOperations/GpuMatrixMultiply/CublasMatrixMultiplyTest test/Utilities/TensorOperations/GpuMatrixMultiply/CublasMatrixMultiplyTest.cpp -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/Utilities/TensorOperations/GpuConvolution/GpuConvolutionTest: build/test/googletest/libgtest.a test/Utilities/TensorOperations/GpuConvolution/GpuConvolutionTest.cpp $(THOR)
	mkdir -p build/test/Utilities/TensorOperations/GpuConvolution
	$(Gpp) -o build/test/Utilities/TensorOperations/GpuConvolution/GpuConvolutionTest test/Utilities/TensorOperations/GpuConvolution/GpuConvolutionTest.cpp -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/DeepLearning/Implementation/Layers/NeuralNetwork/Convolution2dTest: build/test/googletest/libgtest.a test/DeepLearning/Implementation/Layers/NeuralNetwork/Convolution2dTest.cpp $(THOR)
	mkdir -p build/test/DeepLearning/Implementation/Layers/NeuralNetwork
	$(Gpp) $(DEBUG) -o build/test/DeepLearning/Implementation/Layers/NeuralNetwork/Convolution2dTest test/DeepLearning/Implementation/Layers/NeuralNetwork/Convolution2dTest.cpp -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/DeepLearning/Implementation/Layers/NeuralNetwork/FullyConnectedTest: build/test/googletest/libgtest.a test/DeepLearning/Implementation/Layers/NeuralNetwork/FullyConnectedTest.cpp $(THOR)
	mkdir -p build/test/DeepLearning/Implementation/Layers/NeuralNetwork
	$(Gpp) -g $(DEBUG) -o build/test/DeepLearning/Implementation/Layers/NeuralNetwork/FullyConnectedTest test/DeepLearning/Implementation/Layers/NeuralNetwork/FullyConnectedTest.cpp -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/DeepLearning/Implementation/Layers/Loss/LossShaperTest: build/test/googletest/libgtest.a test/DeepLearning/Implementation/Layers/Loss/LossShaperTest.cpp $(THOR)
	mkdir -p build/test/DeepLearning/Implementation/Layers/Loss
	$(Gpp) -g $(DEBUG) -o build/test/DeepLearning/Implementation/Layers/Loss/LossShaperTest test/DeepLearning/Implementation/Layers/Loss/LossShaperTest.cpp -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/DeepLearning/Implementation/Layers/Loss/MeanSquaredErrorTest: build/test/googletest/libgtest.a test/DeepLearning/Implementation/Layers/Loss/MeanSquaredErrorTest.cpp $(THOR)
	mkdir -p build/test/DeepLearning/Implementation/Layers/Loss
	$(Gpp) -g $(DEBUG) -o build/test/DeepLearning/Implementation/Layers/Loss/MeanSquaredErrorTest test/DeepLearning/Implementation/Layers/Loss/MeanSquaredErrorTest.cpp -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/DeepLearning/Implementation/Layers/Loss/MeanAbsoluteErrorTest: build/test/googletest/libgtest.a test/DeepLearning/Implementation/Layers/Loss/MeanAbsoluteErrorTest.cpp $(THOR)
	mkdir -p build/test/DeepLearning/Implementation/Layers/Loss
	$(Gpp) -g $(DEBUG) -o build/test/DeepLearning/Implementation/Layers/Loss/MeanAbsoluteErrorTest test/DeepLearning/Implementation/Layers/Loss/MeanAbsoluteErrorTest.cpp -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/DeepLearning/Implementation/Layers/Loss/MeanAbsolutePercentageErrorTest: build/test/googletest/libgtest.a test/DeepLearning/Implementation/Layers/Loss/MeanAbsolutePercentageErrorTest.cpp $(THOR)
	mkdir -p build/test/DeepLearning/Implementation/Layers/Loss
	$(Gpp) -g $(DEBUG) -o build/test/DeepLearning/Implementation/Layers/Loss/MeanAbsolutePercentageErrorTest test/DeepLearning/Implementation/Layers/Loss/MeanAbsolutePercentageErrorTest.cpp -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/DeepLearning/Implementation/Layers/NeuralNetwork/PoolingTest: build/test/googletest/libgtest.a test/DeepLearning/Implementation/Layers/NeuralNetwork/PoolingTest.cpp $(THOR)
	mkdir -p build/test/DeepLearning/Implementation/Layers/NeuralNetwork
	$(Gpp) -g $(DEBUG) -o build/test/DeepLearning/Implementation/Layers/NeuralNetwork/PoolingTest test/DeepLearning/Implementation/Layers/NeuralNetwork/PoolingTest.cpp -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/Utilities/ComputeTopology/machineEvaluatorTest: build/test/googletest/libgtest.a test/Utilities/ComputeTopology/machineEvaluatorTest.cpp $(THOR)
	mkdir -p build/test/Utilities/ComputeTopology
	$(Gpp) $(DEBUG) -o build/test/Utilities/ComputeTopology/machineEvaluatorTest test/Utilities/ComputeTopology/machineEvaluatorTest.cpp -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/Utilities/Common/OptionalTest: build/test/googletest/libgtest.a test/Utilities/Common/OptionalTest.cpp $(THOR)
	mkdir -p build/test/Utilities/Common
	$(Gpp) $(DEBUG) -o build/test/Utilities/Common/OptionalTest test/Utilities/Common/OptionalTest.cpp -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/Utilities/Random/FullPeriodRandomTest: build/test/googletest/libgtest.a test/Utilities/Random/FullPeriodRandomTest.cpp $(THOR)
	mkdir -p build/test/Utilities/Random
	$(Gpp) $(DEBUG) -o build/test/Utilities/Random/FullPeriodRandomTest test/Utilities/Random/FullPeriodRandomTest.cpp -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/Utilities/WorkQueue/AsyncQueueTest: test/Utilities/WorkQueue/AsyncQueueTest.cpp build/test/googletest/libgtest.a $(THOR)
	mkdir -p build/test/Utilities/WorkQueue
	$(Gpp) $(DEBUG) -o build/test/Utilities/WorkQueue/AsyncQueueTest test/Utilities/WorkQueue/AsyncQueueTest.cpp -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/Utilities/WorkQueue/AsyncTensorQueueTest: test/Utilities/WorkQueue/AsyncTensorQueueTest.cpp build/test/googletest/libgtest.a $(THOR)
	mkdir -p build/test/Utilities/WorkQueue
	$(Gpp) $(DEBUG) -o build/test/Utilities/WorkQueue/AsyncTensorQueueTest test/Utilities/WorkQueue/AsyncTensorQueueTest.cpp -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/DeepLearning/Implementation/Tensor/TensorTest: build/test/googletest/libgtest.a test/DeepLearning/Implementation/Tensor/TensorTest.cpp $(THOR)
	mkdir -p build/test/DeepLearning/Implementation/Tensor
	$(Gpp) $(DEBUG) -o build/test/DeepLearning/Implementation/Tensor/TensorTest test/DeepLearning/Implementation/Tensor/TensorTest.cpp -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/DeepLearning/Implementation/Tensor/TensorTrigonometricKernelsTest: build/test/googletest/libgtest.a test/DeepLearning/Implementation/Tensor/TensorTrigonometricKernelsTest.cpp $(THOR)
	mkdir -p build/test/DeepLearning/Implementation/Tensor
	$(Gpp) $(DEBUG) -o build/test/DeepLearning/Implementation/Tensor/TensorTrigonometricKernelsTest test/DeepLearning/Implementation/Tensor/TensorTrigonometricKernelsTest.cpp -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/DeepLearning/Implementation/Tensor/TensorHyperbolicTrigonometricKernelsTest: build/test/googletest/libgtest.a test/DeepLearning/Implementation/Tensor/TensorHyperbolicTrigonometricKernelsTest.cpp $(THOR)
	mkdir -p build/test/DeepLearning/Implementation/Tensor
	$(Gpp) $(DEBUG) -o build/test/DeepLearning/Implementation/Tensor/TensorHyperbolicTrigonometricKernelsTest test/DeepLearning/Implementation/Tensor/TensorHyperbolicTrigonometricKernelsTest.cpp -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/Utilities/TensorOperations/TypeConversions/TypeConverterTest: build/test/googletest/libgtest.a test/Utilities/TensorOperations/TypeConversions/TypeConverterTest.cpp $(THOR)
	mkdir -p build/test/Utilities/TensorOperations/TypeConversions
	$(Gpp) $(DEBUG) -o build/test/Utilities/TensorOperations/TypeConversions/TypeConverterTest test/Utilities/TensorOperations/TypeConversions/TypeConverterTest.cpp -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/Utilities/TensorOperations/Arithmetic/ArithmeticTest: build/test/googletest/libgtest.a test/Utilities/TensorOperations/Arithmetic/ArithmeticTest.cpp $(THOR)
	mkdir -p build/test/Utilities/TensorOperations/Arithmetic/
	$(Gpp) $(DEBUG) -o build/test/Utilities/TensorOperations/Arithmetic/ArithmeticTest -std=c++11 -pthread test/Utilities/TensorOperations/Arithmetic/ArithmeticTest.cpp $(CUDA_INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/Utilities/TensorOperations/Misc/MiscTest: build/test/googletest/libgtest.a test/Utilities/TensorOperations/Misc/MiscTest.cpp $(THOR)
	mkdir -p build/test/Utilities/TensorOperations/Misc/
	$(Gpp) $(DEBUG) -o build/test/Utilities/TensorOperations/Misc/MiscTest -std=c++11 -pthread test/Utilities/TensorOperations/Misc/MiscTest.cpp $(CUDA_INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/Utilities/TensorOperations/Misc/ComputeCategoricalAccuracyTest: build/test/googletest/libgtest.a test/Utilities/TensorOperations/Misc/ComputeCategoricalAccuracyTest.cpp $(THOR)
	mkdir -p build/test/Utilities/TensorOperations/Misc/
	$(Gpp) $(DEBUG) -o build/test/Utilities/TensorOperations/Misc/ComputeCategoricalAccuracyTest -std=c++11 -pthread test/Utilities/TensorOperations/Misc/ComputeCategoricalAccuracyTest.cpp $(CUDA_INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/Utilities/TensorOperations/Misc/ComputeBinaryAccuracyTest: build/test/googletest/libgtest.a test/Utilities/TensorOperations/Misc/ComputeBinaryAccuracyTest.cpp $(THOR)
	mkdir -p build/test/Utilities/TensorOperations/Misc/
	$(Gpp) $(DEBUG) -o build/test/Utilities/TensorOperations/Misc/ComputeBinaryAccuracyTest -std=c++11 -pthread test/Utilities/TensorOperations/Misc/ComputeBinaryAccuracyTest.cpp $(CUDA_INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/Utilities/TensorOperations/Misc/BatchReduceTest: build/test/googletest/libgtest.a test/Utilities/TensorOperations/Misc/BatchReduceTest.cpp $(THOR)
	mkdir -p build/test/Utilities/TensorOperations/Misc/
	$(Gpp) $(DEBUG) -o build/test/Utilities/TensorOperations/Misc/BatchReduceTest -std=c++11 -pthread test/Utilities/TensorOperations/Misc/BatchReduceTest.cpp $(CUDA_INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/Utilities/Loaders/ShardedRawDatasetCreatorTest: build/test/googletest/libgtest.a test/Utilities/Loaders/ShardedRawDatasetCreatorTest.cpp $(THOR)
	mkdir -p build/test/Utilities/Loaders/
	$(Gpp) $(DEBUG) -o build/test/Utilities/Loaders/ShardedRawDatasetCreatorTest -std=c++11 -pthread test/Utilities/Loaders/ShardedRawDatasetCreatorTest.cpp $(CUDA_INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES) $(GRAPHICS_MAGICK)

build/test/DeepLearning/Implementation/Layers/Utility/UtilityLayerTest: build/test/googletest/libgtest.a test/DeepLearning/Implementation/Layers/Utility/UtilityLayerTest.cpp $(THOR)
	mkdir -p build/test/DeepLearning/Implementation/Layers/Utility
	$(Gpp) $(DEBUG) -o build/test/DeepLearning/Implementation/Layers/Utility/UtilityLayerTest test/DeepLearning/Implementation/Layers/Utility/UtilityLayerTest.cpp -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/DeepLearning/Implementation/Layers/Activations/ActivationsLayerTest: build/test/googletest/libgtest.a test/DeepLearning/Implementation/Layers/Activations/ActivationsLayerTest.cpp $(THOR)
	mkdir -p build/test/DeepLearning/Implementation/Layers/Activations
	$(Gpp) $(DEBUG) -o build/test/DeepLearning/Implementation/Layers/Activations/ActivationsLayerTest test/DeepLearning/Implementation/Layers/Activations/ActivationsLayerTest.cpp -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/DeepLearning/Implementation/Layers/NeuralNetwork/DropOutTest: build/test/googletest/libgtest.a test/DeepLearning/Implementation/Layers/NeuralNetwork/DropOutTest.cpp $(THOR)
	mkdir -p build/test/DeepLearning/Implementation/Layers/NeuralNetwork
	$(Gpp) $(DEBUG) -o build/test/DeepLearning/Implementation/Layers/NeuralNetwork/DropOutTest test/DeepLearning/Implementation/Layers/NeuralNetwork/DropOutTest.cpp -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/DeepLearning/Implementation/Layers/Loss/CategoricalCrossEntropyTest: build/test/googletest/libgtest.a test/DeepLearning/Implementation/Layers/Loss/CategoricalCrossEntropyTest.cpp $(THOR)
	mkdir -p build/test/DeepLearning/Implementation/Layers/Loss
	$(Gpp) $(DEBUG) -o build/test/DeepLearning/Implementation/Layers/Loss/CategoricalCrossEntropyTest test/DeepLearning/Implementation/Layers/Loss/CategoricalCrossEntropyTest.cpp -O0 -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/DeepLearning/Implementation/Layers/Loss/BinaryCrossEntropyTest: build/test/googletest/libgtest.a test/DeepLearning/Implementation/Layers/Loss/BinaryCrossEntropyTest.cpp $(THOR)
	mkdir -p build/test/DeepLearning/Implementation/Layers/Loss
	$(Gpp) $(DEBUG) -o build/test/DeepLearning/Implementation/Layers/Loss/BinaryCrossEntropyTest test/DeepLearning/Implementation/Layers/Loss/BinaryCrossEntropyTest.cpp -O0 -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/DeepLearning/Implementation/Layers/Metric/CategoricalAccuracyTest: build/test/googletest/libgtest.a test/DeepLearning/Implementation/Layers/Metric/CategoricalAccuracyTest.cpp $(THOR)
	mkdir -p build/test/DeepLearning/Implementation/Layers/Metric
	$(Gpp) $(DEBUG) -o build/test/DeepLearning/Implementation/Layers/Metric/CategoricalAccuracyTest test/DeepLearning/Implementation/Layers/Metric/CategoricalAccuracyTest.cpp -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/DeepLearning/Implementation/Layers/Metric/BinaryAccuracyTest: build/test/googletest/libgtest.a test/DeepLearning/Implementation/Layers/Metric/BinaryAccuracyTest.cpp $(THOR)
	mkdir -p build/test/DeepLearning/Implementation/Layers/Metric
	$(Gpp) $(DEBUG) -o build/test/DeepLearning/Implementation/Layers/Metric/BinaryAccuracyTest test/DeepLearning/Implementation/Layers/Metric/BinaryAccuracyTest.cpp -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/DeepLearning/Api/Visualizers/ConsoleVisualizerTest: build/test/googletest/libgtest.a test/DeepLearning/Api/Visualizers/ConsoleVisualizerTest.cpp $(THOR)
	mkdir -p build/test/DeepLearning/Api/Visualizers/
	$(Gpp) $(DEBUG) -o build/test/DeepLearning/Api/Visualizers/ConsoleVisualizerTest test/DeepLearning/Api/Visualizers/ConsoleVisualizerTest.cpp -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/Utilities/TensorOperations/Loss/CrossEntropyLossTest: build/test/googletest/libgtest.a test/Utilities/TensorOperations/Loss/CrossEntropyLossTest.cpp $(THOR)
	mkdir -p build/test/Utilities/TensorOperations/Loss/
	$(Gpp) $(DEBUG) -o build/test/Utilities/TensorOperations/Loss/CrossEntropyLossTest -std=c++11 -pthread test/Utilities/TensorOperations/Loss/CrossEntropyLossTest.cpp $(CUDA_INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/DeepLearning/Implementation/SimpleNetworkTest: build/test/googletest/libgtest.a test/DeepLearning/Implementation/SimpleNetworkTest.cpp $(THOR)
	mkdir -p build/test/DeepLearning/Implementation
	$(Gpp) $(DEBUG) -o build/test/DeepLearning/Implementation/SimpleNetworkTest test/DeepLearning/Implementation/SimpleNetworkTest.cpp -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/DeepLearning/Implementation/Layers/NeuralNetwork/BatchNormalizationTest: build/test/googletest/libgtest.a test/DeepLearning/Implementation/Layers/NeuralNetwork/BatchNormalizationTest.cpp $(THOR)
	mkdir -p build/test/DeepLearning/Implementation/Layers/NeuralNetwork
	$(Gpp) $(DEBUG) -o build/test/DeepLearning/Implementation/Layers/NeuralNetwork/BatchNormalizationTest test/DeepLearning/Implementation/Layers/NeuralNetwork/BatchNormalizationTest.cpp -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/DeepLearning/Api/Layers/Activations/ActivationsTest: build/test/googletest/libgtest.a test/DeepLearning/Api/Layers/Activations/ActivationsTest.cpp $(THOR)
	mkdir -p build/test/DeepLearning/Api/Layers/Activations
	$(Gpp) $(DEBUG) -o build/test/DeepLearning/Api/Layers/Activations/ActivationsTest test/DeepLearning/Api/Layers/Activations/ActivationsTest.cpp -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/DeepLearning/Api/Network/NetworkTest: build/test/googletest/libgtest.a test/DeepLearning/Api/Network/NetworkTest.cpp $(THOR)
	mkdir -p build/test/DeepLearning/Api/Network
	$(Gpp) $(DEBUG) -o build/test/DeepLearning/Api/Network/NetworkTest test/DeepLearning/Api/Network/NetworkTest.cpp -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/DeepLearning/Api/Layers/Loss/CategoricalCrossEntropyTest: build/test/googletest/libgtest.a test/DeepLearning/Api/Layers/Loss/CategoricalCrossEntropyTest.cpp $(THOR)
	mkdir -p build/test/DeepLearning/Api/Layers/Loss
	$(Gpp) $(DEBUG) -o build/test/DeepLearning/Api/Layers/Loss/CategoricalCrossEntropyTest test/DeepLearning/Api/Layers/Loss/CategoricalCrossEntropyTest.cpp -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/DeepLearning/Api/Layers/Loss/BinaryCrossEntropyTest: build/test/googletest/libgtest.a test/DeepLearning/Api/Layers/Loss/BinaryCrossEntropyTest.cpp $(THOR)
	mkdir -p build/test/DeepLearning/Api/Layers/Loss
	$(Gpp) $(DEBUG) -o build/test/DeepLearning/Api/Layers/Loss/BinaryCrossEntropyTest test/DeepLearning/Api/Layers/Loss/BinaryCrossEntropyTest.cpp -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/DeepLearning/Api/Layers/Metric/CategoricalAccuracyTest: build/test/googletest/libgtest.a test/DeepLearning/Api/Layers/Metric/CategoricalAccuracyTest.cpp $(THOR)
	mkdir -p build/test/DeepLearning/Api/Layers/Metric
	$(Gpp) $(DEBUG) -o build/test/DeepLearning/Api/Layers/Metric/CategoricalAccuracyTest test/DeepLearning/Api/Layers/Metric/CategoricalAccuracyTest.cpp -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/DeepLearning/Api/Layers/Metric/BinaryAccuracyTest: build/test/googletest/libgtest.a test/DeepLearning/Api/Layers/Metric/BinaryAccuracyTest.cpp $(THOR)
	mkdir -p build/test/DeepLearning/Api/Layers/Metric
	$(Gpp) $(DEBUG) -o build/test/DeepLearning/Api/Layers/Metric/BinaryAccuracyTest test/DeepLearning/Api/Layers/Metric/BinaryAccuracyTest.cpp -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/DeepLearning/Api/Layers/Loss/MeanSquaredErrorTest: build/test/googletest/libgtest.a test/DeepLearning/Api/Layers/Loss/MeanSquaredErrorTest.cpp $(THOR)
	mkdir -p build/test/DeepLearning/Api/Layers/Loss
	$(Gpp) $(DEBUG) -o build/test/DeepLearning/Api/Layers/Loss/MeanSquaredErrorTest test/DeepLearning/Api/Layers/Loss/MeanSquaredErrorTest.cpp -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/DeepLearning/Api/Layers/Loss/MeanAbsoluteErrorTest: build/test/googletest/libgtest.a test/DeepLearning/Api/Layers/Loss/MeanAbsoluteErrorTest.cpp $(THOR)
	mkdir -p build/test/DeepLearning/Api/Layers/Loss
	$(Gpp) $(DEBUG) -o build/test/DeepLearning/Api/Layers/Loss/MeanAbsoluteErrorTest test/DeepLearning/Api/Layers/Loss/MeanAbsoluteErrorTest.cpp -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/DeepLearning/Api/Layers/Loss/MeanAbsolutePercentageErrorTest: build/test/googletest/libgtest.a test/DeepLearning/Api/Layers/Loss/MeanAbsolutePercentageErrorTest.cpp $(THOR)
	mkdir -p build/test/DeepLearning/Api/Layers/Loss
	$(Gpp) $(DEBUG) -o build/test/DeepLearning/Api/Layers/Loss/MeanAbsolutePercentageErrorTest test/DeepLearning/Api/Layers/Loss/MeanAbsolutePercentageErrorTest.cpp -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/DeepLearning/Api/Layers/Loss/LossShaperTest: build/test/googletest/libgtest.a test/DeepLearning/Api/Layers/Loss/LossShaperTest.cpp $(THOR)
	mkdir -p build/test/DeepLearning/Api/Layers/Loss
	$(Gpp) $(DEBUG) -o build/test/DeepLearning/Api/Layers/Loss/LossShaperTest test/DeepLearning/Api/Layers/Loss/LossShaperTest.cpp -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/DeepLearning/Api/Layers/Utility/UtilityLayerTests: build/test/googletest/libgtest.a test/DeepLearning/Api/Layers/Utility/UtilityLayerTests.cpp $(THOR)
	mkdir -p build/test/DeepLearning/Api/Layers/Utility
	$(Gpp) $(DEBUG) -o build/test/DeepLearning/Api/Layers/Utility/UtilityLayerTests test/DeepLearning/Api/Layers/Utility/UtilityLayerTests.cpp -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/DeepLearning/Api/Layers/Learning/FullyConnectedTest: build/test/googletest/libgtest.a test/DeepLearning/Api/Layers/Learning/FullyConnectedTest.cpp $(THOR)
	mkdir -p build/test/DeepLearning/Api/Layers/Learning
	$(Gpp) $(DEBUG) -o build/test/DeepLearning/Api/Layers/Learning/FullyConnectedTest test/DeepLearning/Api/Layers/Learning/FullyConnectedTest.cpp -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/DeepLearning/Api/Layers/Learning/Convolution2dTest: build/test/googletest/libgtest.a test/DeepLearning/Api/Layers/Learning/Convolution2dTest.cpp $(THOR)
	mkdir -p build/test/DeepLearning/Api/Layers/Learning
	$(Gpp) $(DEBUG) -o build/test/DeepLearning/Api/Layers/Learning/Convolution2dTest test/DeepLearning/Api/Layers/Learning/Convolution2dTest.cpp -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/DeepLearning/Api/ExampleNetworks/PerformanceTests/AlexNetPerformanceTest: DeepLearning/Api/ExampleNetworks/PerformanceTests/AlexNetPerformanceTest.cpp DeepLearning/Api/Executors/HackathonExecutor.h $(THOR)
	mkdir -p build/DeepLearning/Api/ExampleNetworks/PerformanceTests
	$(Gpp) DeepLearning/Api/ExampleNetworks/PerformanceTests/AlexNetPerformanceTest.cpp -std=c++11 -pthread $(THOR_LIBS) $(CUDA) $(INCLUDE_DIRS) -o build/DeepLearning/Api/ExampleNetworks/PerformanceTests/AlexNetPerformanceTest

build/DeepLearning/Api/ExampleNetworks/PerformanceTests/InceptionV3PerformanceTest: DeepLearning/Api/ExampleNetworks/PerformanceTests/InceptionV3PerformanceTest.cpp DeepLearning/Api/Executors/HackathonExecutor.h $(THOR)
	mkdir -p build/DeepLearning/Api/ExampleNetworks/PerformanceTests
	$(Gpp) $(DEBUG) DeepLearning/Api/ExampleNetworks/PerformanceTests/InceptionV3PerformanceTest.cpp -std=c++11 -pthread $(THOR_LIBS) $(CUDA) $(INCLUDE_DIRS) -o build/DeepLearning/Api/ExampleNetworks/PerformanceTests/InceptionV3PerformanceTest

build/test/DeepLearning/Api/Optimizers/SgdTest: build/test/googletest/libgtest.a test/DeepLearning/Api/Optimizers/SgdTest.cpp $(THOR)
	mkdir -p build/test/DeepLearning/Api/Optimizers
	$(Gpp) $(DEBUG) -o build/test/DeepLearning/Api/Optimizers/SgdTest test/DeepLearning/Api/Optimizers/SgdTest.cpp -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/DeepLearning/Api/Optimizers/AdamTest: build/test/googletest/libgtest.a test/DeepLearning/Api/Optimizers/AdamTest.cpp $(THOR)
	mkdir -p build/test/DeepLearning/Api/Optimizers
	$(Gpp) $(DEBUG) -o build/test/DeepLearning/Api/Optimizers/AdamTest test/DeepLearning/Api/Optimizers/AdamTest.cpp -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/DeepLearning/Implementation/Layers/Optimizers/SgdTest: build/test/googletest/libgtest.a test/DeepLearning/Implementation/Layers/Optimizers/SgdTest.cpp test/Utilities/TensorOperations/GpuMatrixMultiply/MatrixMultiplyTestHelper.h $(THOR)
	mkdir -p build/test/DeepLearning/Implementation/Layers/Optimizers
	$(Gpp) $(DEBUG) -o build/test/DeepLearning/Implementation/Layers/Optimizers/SgdTest test/DeepLearning/Implementation/Layers/Optimizers/SgdTest.cpp -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/DeepLearning/Implementation/Layers/Optimizers/AdamTest: build/test/googletest/libgtest.a test/DeepLearning/Implementation/Layers/Optimizers/AdamTest.cpp $(THOR)
	mkdir -p build/test/DeepLearning/Implementation/Layers/Optimizers
	$(Gpp) $(DEBUG) -o build/test/DeepLearning/Implementation/Layers/Optimizers/AdamTest test/DeepLearning/Implementation/Layers/Optimizers/AdamTest.cpp -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)

build/test/Utilities/TensorOperations/Optimizers/AdamTest: build/test/googletest/libgtest.a test/Utilities/TensorOperations/Optimizers $(THOR)
	mkdir -p build/test/Utilities/TensorOperations/Optimizers
	$(Gpp) $(DEBUG) -o build/test/Utilities/TensorOperations/Optimizers/AdamTest test/Utilities/TensorOperations/Optimizers/AdamTest.cpp -std=c++11 -pthread $(CUDA_INCLUDE_DIRS) $(THOR_LIBS) $(TEST_COMPILE_DEPENDENCIES)





























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
	g++ -o build/test/Utilities/WorkQueue/WorkQueueTest -std=c++11 -pthread -IUtilities/WorkQueue/ test/Utilities/WorkQueue/WorkQueueTest.cpp $(TEST_COMPILE_DEPENDENCIES)
	g++ -o build/test/Utilities/WorkQueue/WorkQueueUnorderedTest -std=c++11 -pthread -IUtilities/WorkQueue/ test/Utilities/WorkQueue/WorkQueueUnorderedTest.cpp $(TEST_COMPILE_DEPENDENCIES)

RunWorkQueueTests: BuildWorkQueueTests
	build/test/Utilities/WorkQueue/WorkQueueTest
	build/test/Utilities/WorkQueue/WorkQueueUnorderedTest























# Old stuff, some of it to bring into real makefile above


article2Vec: article2Vec.cu article2Vec.cpp article2Vec.h article2VecMonitor.cpp  extractDataSet.cpp
	g++ extractDataSet.cpp -std=c++11 -O3 -o extractDataSet
	nvcc -gencode arch=compute_52,code=compute_52 -gencode arch=compute_52,code=sm_52 -c -I/usr/local/cuda/include -Xptxas -O3,-v article2Vec.cu -o article2Vec.o
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
	nvcc -std=c++11 -gencode arch=compute_52,code=compute_52 -gencode arch=compute_52,code=sm_52 -c -I/usr/local/cuda/include -Xptxas -O3,-v clusterKernels.cu
	#g++ -o clusterKernels -std=c++11 clusterKernels.o -O3 -L/usr/local/cuda/lib64 -lcusolver -lcudart

optimalMatrixMultiplyBad: optimalMatrixMultiplyBad.cpp optimalMatrixMultiplyKernels.cu makefile
	nvcc -std=c++11 -gencode=arch=compute_52,code=compute_52 -gencode=arch=compute_75,code=compute_75 -g -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_75,code=sm_75 -c -I/usr/local/cuda/include -Xptxas -O3,-v optimalMatrixMultiplyBad.cpp
	nvcc -std=c++11 -gencode=arch=compute_52,code=compute_52 -gencode=arch=compute_75,code=compute_75 -g -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_75,code=sm_75 -c -I/usr/local/cuda/include -Xptxas -O3,-v optimalMatrixMultiplyKernels.cu
	g++ -o optimalMatrixMultiplyBad -std=c++11 optimalMatrixMultiplyBad.o optimalMatrixMultiplyKernels.o -O3 -L/usr/local/cuda/lib64 -lcublas -lcublasLt -lcusolver -lcudart


cublasTest: cublasTest.cpp makefile
	nvcc -std=c++11 -gencode=arch=compute_52,code=compute_52 -gencode=arch=compute_75,code=compute_75 -g -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_75,code=sm_75 -c -I/usr/local/cuda/include -Xptxas -O3,-v cublasTest.cpp
	g++ -o cublasTest -std=c++11 cublasTest.o -O3 -L/usr/local/cuda/lib64 -lcublas -lcublasLt -lcusolver -lcudart

cublasTestDebug: cublasTest.cpp
	nvcc -std=c++11 -gencode=arch=compute_52,code=compute_52 -gencode=arch=compute_75,code=compute_75 -g -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_75,code=sm_75 -c -I/usr/local/cuda/include -Xptxas -O3,-v cublasTest.cpp
	g++ -o cublasTestDebug -std=c++11 cublasTest.o -O0 -L/usr/local/cuda/lib64 -lcublas -lcublasLt -ggdb -lcusolver -lcudart

simpleCublasTestcase: simpleCublasTestcase.cpp
	nvcc -std=c++11 -gencode arch=compute_52,code=compute_52 -g -gencode arch=compute_52,code=sm_52 -c -I/usr/local/cuda/include -Xptxas -O3,-v simpleCublasTestcase.cpp
	g++ -o simpleCublasTestcase -std=c++11 simpleCublasTestcase.o -O0 -L/usr/local/cuda/lib64 -lcublas -lcublasLt -ggdb -lcusolver -lcudart


outerProduct: outerProduct.cu
	nvcc -std=c++11 -gencode arch=compute_52,code=compute_52 -g -gencode arch=compute_52,code=sm_52 -c -I/usr/local/cuda/include -Xptxas -O3,-v outerProduct.cu
	g++ -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -std=c++11 outerProduct.o -O3 -pthread -o outerProduct -lcudart

gpuRTreeKernels: GpuRTreeKernels.cu GpuRTreeKernels.h
	nvcc -std=c++11 -gencode arch=compute_52,code=compute_52 -g -gencode arch=compute_52,code=sm_52 -c -I/usr/local/cuda/include -Xptxas -O3,-v GpuRTreeKernels.cu
#	nvcc -std=c++11 -gencode arch=compute_52,code=compute_52 -g -gencode arch=compute_52,code=sm_52 -I/usr/local/cuda/include -Xptxas -O3,-v --device-c GpuRTreeKernels.cu -o GpuRTreeKernels.o

gpuRTree: GpuRTree.cpp GpuRTreeNode.h GpuRTreeNode.cpp GpuRTreeTests.h gpuRTreeKernels GpuRTreeTests.h NearestNeighborExecutor.h NearestNeighborExecutor.cpp makefile
	g++ -c GpuRTree.cpp -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -std=c++11 -pthread -O3 -lcudart
	g++ -c GpuRTreeNode.cpp -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -std=c++11 -pthread -O3 -lcudart
	g++ -c NearestNeighborExecutor.cpp -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -std=c++11 -pthread -O3 -lcudart
	g++ GpuRTree.o -I/usr/local/cuda/include -L/usr/local/cuda/lib64 GpuRTreeNode.o NearestNeighborExecutor.o GpuRTreeKernels.o -std=c++11 -pthread -O3 -o gpuRTree -lcudart
#	g++ GpuRTree.cpp NearestNeighborExecutor.cpp -I/usr/local/cuda/include -L/usr/local/cuda/lib64 GpuRTreeKernels.o -std=c++11 -pthread -O3 -o gpuRTree -lcudart
#	nvcc -std=c++11 -gencode arch=compute_52,code=compute_52 -g -gencode arch=compute_52,code=sm_52 -I/usr/local/cuda/include -Xptxas -O3,-v --device-c GpuRTree.cu -o gpuRTree.o
#	nvcc -std=c++11 -gencode arch=compute_52,code=compute_52 -g -gencode arch=compute_52,code=sm_52 -I/usr/local/cuda/include -Xptxas -O3,-v  GpuRTreeKernels.o GpuRTree.o -o gpuRTree

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










