# TODO:
1. Save and load trained models - must be compatible with keras.
1. Improve visualizer so that logging to the console is not cleared when run finishes.
2. Overhaul the build system, possibly using CMake
3. Python Bindings - using pybind11
4. Losses
5. Additional math function support including backward pass logic.
6. Recurrent Neural Network support
   1. LSTM
   1. GRU
   1. Transformer support
      1. Attention 
      1. Multi-headed attention
7. Graph Neural Network support
   1. Algorithms like random walk, etc.
   1. Investigate what can be built into the framework to support GNN's
8. Embedding support
9. Kmeans clustering using GpuRtree
10. Gaussian Mixture Model clustering using GpuRtree
11. Java Bindings - likely using BridJ
12. k-fold cross validation
13. Feature elimination using decision trees
14. GPU SVD support using nvidia library
15. GPU FFT support using nvidia library - including windowing i.e. Hanning window, Hamming window, etc.
16. PCA support
17. Streaming server support built in. Hmm, think about this, want to interface with kubernetes.
   1. What would the framework provide?
18. Expand the list of supported clustering algorithms
19. Expand the list of signal processing algorithms
20. Audio decoding / support
21. Video decoding / support
