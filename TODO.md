# TODO:
1. Losses
2. Fix multi-connection api layer stamp logic
3. Save and load trained models - must be compatible with keras.
4. Improve visualizer so that logging to the console is not cleared when run finishes.
5. Overhaul the build system, possibly using CMake
6. Python Bindings - using pybind11
7. Multi-Gpu support
   1. Multiple stamps with accumulation.
      1. Not that the current implementation updates the layers as they are added to the network so that their inputs and outputs remain correct after flattening multi-layers. However only single layers are added to the network and so it may be stamped as many times as desired.
   2. Defer kernel selection due to mem requirements of workspace until logic that determines number of stamps
8. Additional math function support including backward pass logic.
9. Recurrent Neural Network support
   1. LSTM
   1. GRU
   1. Transformer support
      1. Attention 
      1. Multi-headed attention - implemented via CuDNN, possibly can be used for regular Attention also.
10. Graph Neural Network support
    1. Algorithms like random walk, etc.
    1. Investigate what can be built into the framework to support GNN's
11. Embedding support
12. Kmeans clustering using GpuRtree
13. Gaussian Mixture Model clustering using GpuRtree
14. NN architecture drawing export
    1. support shading units to show tensor values given a particular training example
    2. There should be options to show the tensor into and out of say the activation, batch norm, etc or to collapse it and only show the last one
15. Java Bindings - likely using BridJ
16. k-fold cross validation
17. Feature elimination using decision trees
18. GPU SVD support using nvidia library
19. GPU FFT support using nvidia library - including windowing i.e. Hanning window, Hamming window, etc.
20. PCA support
21. Streaming server support built in. Hmm, think about this, want to interface with kubernetes.
22. What would the framework provide?
23. Expand the list of supported clustering algorithms
24. Expand the list of signal processing algorithms
25. Audio decoding / support
26. Video decoding / support
