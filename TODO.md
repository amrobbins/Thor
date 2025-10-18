# FIXME:
DeepLearning/Implementation/SimpleNetworkTest.cpp
DeepLearning/Implementation/Layers/Optimizers/AdamTest.cpp
DeepLearning/Implementation/Layers/Optimizers/SgdTest.cpp


# TODO:
1. ADAM Optimizer - test and get working - done - get test to pass consistently.
2. Alexnet learning rate parity with Tensorflow on imagenet.
3. Create a kanban board with current priority items
4. Save and load trained models - native save() and load() functions, also saveAsKeras() and loadFromKeras()
   1. Weights saved in safetensors format. model saved in json, with SHA for integrity. Encryption optional.
      2. For python API safetensors is a native API, for c++ api, bind with python to call from c++ side.
6. Overhaul the build system, possibly using CMake
5. Determine the optimal number of stamps per GPU. Note that this may be different for training and inference, will depend on batch size,
      There may be additional inference-only stamps.
9. Multi-Gpu support
    1. Multiple stamps with accumulation.
        1. Note that the current implementation updates the layers as they are added to the network so that their inputs and outputs remain correct after flattening multi-layers. However only single layers are added to the network and so it may be stamped as many times as desired.
    2. Defer kernel selection due to mem requirements of workspace until logic that determines number of stamps
6. Support to deploy inference server (python: gunicorn running uvicorn workers, c++: something else) using trained model to run inference requests from a single server 
4. Mux and Demux -> 1 to N controlled by input tensor / N to 1 controlled by input tensor.
5. reuse workspaces based on dependency graph, per stamp.
   6. Meaning if I have an FC layer and it needs 100 MB workspace for forward pass and a 75 MB workspace for backward pass, I only need the 100 MB workspace.
   7. Also if I have FC layer 1 and it needs 100 MB workspace and I have FC Layer 2 and it needs 50 MB workspace and these are connected sequentially, I only need the 100 MB workspace.  
      1. Use GPUDirect storage to save and load weights so that they don't need fit in CPU memory. Also for performance.
      1. Done. Check if tests pass.
2. Additional Losses
4. mish activation function
5. Kernel fusing step upon compiling network.
   6. Note that if the network is being constructed for inference purposes we don't need to maintain the ability to compute backward gradients.
5. Improve visualizer so that logging to the console is not cleared when run finishes.
   6. The existing one probably goes away, it didn't come out as desired.
7. CPU support for all layers
   1. https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-0/cblas-gemm-001.html
   2. https://oneapi-src.github.io/oneDNN/v0/index.html
8. Add functional support to Tensor to implement a full high performance linear algebra library, for example matrix inverse, cholesky decomposition, SVD, vector normalization, batched vector normalization (in general vector operations should have a batched version for high performance), cross product, ...
9. Fix PACKED_BOOLEAN: support for dimensions not of multiple of 8, so there will be some unused bits at the end of a row.
9. Python Bindings - using pybind11
11. Recurrent Neural Network support
    1. LSTM
    1. GRU
    1. Transformer support
       1. Attention 
       1. Multi-headed attention - implemented via CuDNN, possibly can be used for regular Attention also.
12. Graph Neural Network support
    1. Algorithms like random walk, etc.
    1. Investigate what can be built into the framework to support GNN's
13. Embedding support
14. Kmeans clustering using GpuRtree
15. DeepDPM implementation
15. Gaussian Mixture Model clustering using GpuRtree
16. Multi-server support, perhaps through an api to horizontally scale but make it look like vertical scaling. So in this case memory would have a CPU affinity.
    17. IO bandwidth would need to be better understood by the system.
    18. This may be an enterprise only solution.
16. NN architecture drawing export
    1. support shading units to show tensor values given a particular training example
    2. There should be options to show the tensor into and out of say the activation, batch norm, etc or to collapse it and only show the last one
17. Feature elimination using decision trees
18. k-fold cross validation
24. Other optimizers
24. Expand the list of supported clustering algorithms

--------------- below this line maybe not doing -------------------------

1. Java Bindings - likely using BridJ - maybe not but using some library.
1. GPU SVD support using nvidia library
1. GPU FFT support using nvidia library - including windowing i.e. Hanning window, Hamming window, etc.
1. PCA support

