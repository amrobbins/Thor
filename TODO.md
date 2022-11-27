# TODO:
1. Save and load trained models - must be compatible with keras.
2. Improve visualizer so that logging to the console is not cleared when run finishes.
3. Overhaul the build system, possibly using CMake
4. Python Bindings - using pybind11
5. Multi-Gpu support
   1. Multiple stamps with accumulation.
      1. Not that the current implementation updates the layers as they are added to the network so that their inputs and outputs remain correct after flattening multi-layers. However only single layers are added to the network and so it may be stamped as many times as desired.
   2. Defer kernel selection due to mem requirements of workspace until logic that determines number of stamps
6. Losses
7. Additional math function support including backward pass logic.
8. Recurrent Neural Network support
   1. LSTM
   1. GRU
   1. Transformer support
      1. Attention 
      1. Multi-headed attention - implemented via CuDNN, possibly can be used for regular Attention also.
9. Graph Neural Network support
   1. Algorithms like random walk, etc.
   1. Investigate what can be built into the framework to support GNN's
10. Embedding support
11. Kmeans clustering using GpuRtree
12. Gaussian Mixture Model clustering using GpuRtree
13. NN architecture drawing export
    1. support shading units to show tensor values given a particular training example
    2. There should be options to show the tensor into and out of say the activation, batch norm, etc or to collapse it and only show the last one
14. Java Bindings - likely using BridJ
15. k-fold cross validation
16. Feature elimination using decision trees
17. GPU SVD support using nvidia library
18. GPU FFT support using nvidia library - including windowing i.e. Hanning window, Hamming window, etc.
19. PCA support
20. Streaming server support built in. Hmm, think about this, want to interface with kubernetes.
21. What would the framework provide?
22. Expand the list of supported clustering algorithms
23. Expand the list of signal processing algorithms
24. Audio decoding / support
25. Video decoding / support
