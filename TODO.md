# TODO:
1. Additional Losses
2. Optimizers
3. Fix multi-connection api layer stamp logic - is this actually broken?
4. Save and load trained models - native save() and load() functions, also saveAsKeras() and loadFromKeras()
5. Improve visualizer so that logging to the console is not cleared when run finishes.
6. Overhaul the build system, possibly using 
7. CPU support for all layers
   1. https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-0/cblas-gemm-001.html
   2. https://oneapi-src.github.io/oneDNN/v0/index.html
8. Add functional support to Tensor to implement a full high performance linear algebra library, for example matrix inverse, cholesky decomposition, SVD, vector normalization, batched vector normalization (in general vector operations should have a batched version for high performance), cross product, ...
9. Fix PACKED_BOOLEAN: support for dimensions not of multiple of 8, so there will be some unused bits at the end of a row.
9. Python Bindings - using pybind11
9. Multi-Gpu support
   1. Multiple stamps with accumulation.
      1. Not that the current implementation updates the layers as they are added to the network so that their inputs and outputs remain correct after flattening multi-layers. However only single layers are added to the network and so it may be stamped as many times as desired.
   2. Defer kernel selection due to mem requirements of workspace until logic that determines number of stamps
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
15. Gaussian Mixture Model clustering using GpuRtree
16. NN architecture drawing export
    1. support shading units to show tensor values given a particular training example
    2. There should be options to show the tensor into and out of say the activation, batch norm, etc or to collapse it and only show the last one
17. Java Bindings - likely using BridJ
18. k-fold cross validation
19. Feature elimination using decision trees
20. GPU SVD support using nvidia library
21. GPU FFT support using nvidia library - including windowing i.e. Hanning window, Hamming window, etc.
22. PCA support
23. Streaming server support built in. Hmm, think about this, want to interface with kubernetes.
    1. What would the framework provide?
24. Expand the list of supported clustering algorithms
25. Expand the list of signal processing algorithms
26. Audio decoding / support
27. Video decoding / support
