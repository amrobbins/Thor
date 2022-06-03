# TODO:
1. Save and load trained models - must be compatible with keras.
1. Improve visualizer so that logging to the console is not cleared when run finishes.
1. Python Bindings - using pybind11
1. Losses
1. Additional math function support including backward pass logic.
1. Recurrent Neural Network support
   1. LSTM
   1. GRU
   1. Transformer support
      1. Attention 
      1. Multi-headed attention
1. Graph Neural Network support
   1. Algorithms like random walk, etc.
   1. Investigate what can be built into the framework to support GNN's
1. Embedding support
1. Kmeans clustering using GpuRtree
1. Gaussian Mixture Model clustering using GpuRtree
1. Java Bindings - likely using BridJ
1. k-fold cross validation
1. Feature elimination using decision trees
1. GPU SVD support using nvidia library
1. GPU FFT support using nvidia library - including windowing i.e. Hanning window, Hamming window, etc.
1. PCA support
1. Streaming server support built in. Hmm, think about this, want to interface with kubernetes.
   1. What would the framework provide?
1. Expand the list of supported clustering algorithms
1. Expand the list of signal processing algorithms