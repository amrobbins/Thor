# TODO:
1. Save and load trained models - must be compatible with keras.
2. Improve visualizer so that logging to the console is not cleared when run finishes.
3. Overhaul the build system, possibly using CMake
4. Python Bindings - using pybind11
5. Losses
6. Additional math function support including backward pass logic.
7. Recurrent Neural Network support
   1. LSTM
   1. GRU
   1. Transformer support
      1. Attention 
      1. Multi-headed attention - implemented via CuDNN, possibly can be used for regular Attention also.
8. Graph Neural Network support
   1. Algorithms like random walk, etc.
   1. Investigate what can be built into the framework to support GNN's
9. Embedding support
10. Kmeans clustering using GpuRtree
11. Gaussian Mixture Model clustering using GpuRtree
12. NN architecture drawing export
    1. support shading units to show tensor values given a particular training example
    2. There should be options to show the tensor into and out of say the activation, batch norm, etc or to collapse it and only show the last one
13. Java Bindings - likely using BridJ
14. k-fold cross validation
15. Feature elimination using decision trees
16. GPU SVD support using nvidia library
17. GPU FFT support using nvidia library - including windowing i.e. Hanning window, Hamming window, etc.
18. PCA support
19. Streaming server support built in. Hmm, think about this, want to interface with kubernetes.
20. What would the framework provide?
21. Expand the list of supported clustering algorithms
22. Expand the list of signal processing algorithms
23. Audio decoding / support
24. Video decoding / support
