<img src="logo.png" title="Thor" alt="Tensor Hyper-Parallel Optimized pRocessing" width="360" height="360">


# Thor
## (Tensor Hyper-parallel Optimized pRocessing)

Design Objectives:
1. Efficiency
2. Performance
3. Scaling
4. Ease of use
5. Full featured

## If TensorFlow is your Volkswagen Beetle, then Thor is your Ferrari.

This framework is for Linux, and is currently being developed using Ubunut 22.04.

Dependencies, with installation directions for Ubuntu:
  1. sudo apt-get update
  1. sudo apt-get install build-essential cmake clang-format  
  1. cuda 12  - can install with 'sudo apt-get install cuda'
  1. After installing cuda and resetting, make sure that nvcc is in your path by running 'which nvcc'
     1. If it is not in your path then add the following line to the end of your ~/.bashrc file:
     2. export PATH=$PATH:/usr/local/cuda-12.2/bin
     3. assuming cuda12.2 is installed there. Then close your terminal and open a new one and confirm that nvcc is on your path using 'which nvcc'
  2. cudnn 8.9 for cuda 12 - get it here: https://developer.nvidia.com/cudnn
     1. you will need to create an account and accept the terms
     2. download the local installer for the version of ubuntu you are using - I'm on 22.04
     3. follow this guide to set up public keys and install cudnn via the deb file: https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html
  3. clang-format  - sudo apt-get install clang-format
  4. boost 1.74 (earlier and later is probably fine.) sudo apt-get install libboost-all-dev
  6. ncurses - sudo apt-get install lib64ncurses-dev.
  7. X11 - sudo apt-get install libx11-dev 
  8. GraphicsMagick
     1. Ensure you have all of the image libraries installed:
        1. sudo apt-get install libpng-dev zlib1g-dev libgs-dev libjpeg-dev libtiff-dev libxml2-dev
     1. I built from source as documented below, but that may not be necessary, try this first:
        1. sudo apt install libgraphicsmagick1-dev
        1. If this works then you don't need to follow the build instructions below.
     1. If above doesn't work then follow this build from source example:
     1. download the latest version - http://www.graphicsmagick.org/README.html 
     1. tar -xf GraphicsMagick-1.3.41.tar.xz
     1. cd GraphicsMagick-1.3.41
     1. ./configure CC=gcc CXX=c++ CFLAGS=-O3 CPPFLAGS=-O3 CXXFLAGS=-O3 LDFLAGS='-L/usr/local/lib'
     1. make check
     1. sudo make install

Install:

```shell
bash install_google_test.sh
make -j all
```
