
## Thor is a work in progress Machine Learning framework that I am developing.

The goals of this project are:
  1. Create a optimally high performance AI suite
  2. Allow convenient workflows for development on a local machine and then easily transition to running in the cloud by switching your executor.
  3. Create a thoughtful and easy to use API
  4. The implementation layer will be C++ and Cuda, but an API layer will be supported in 
     1. Python
     2. Java
     3. C++
  5. Make the default settings the recommended configurations, so that beginners can quickly succeed.
  6. Keep up to date on the latest research and bring the promising pieces into this framework.
  7. Educate users about the different pieces and strategies in a straightforward and understandable way. Point users to the latest trends and research. Have a newsletter.


This framework is for Linux, and has been tested on Ubuntu 18.04 and 20.04.

Dependencies, with installation directions for Ubuntu:
  1. sudo apt-get update
  1. sudo apt-get install build-essential cmake
  1. cuda 12  - can install with 'sudo apt-get install cuda'
  2. cudnn 8.9 - get it here: https://developer.nvidia.com/cudnn
     1. you will need to create an account and accept the terms
     2. download the local installer for the version ubuntu you are using - I'm on 22.04
     3. follow this guide to set up public keys and install cudnn via the deb file: https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html
  3. clang-format  - sudo apt-get install clang-format
  4. boost 1.74 (earlier and later is probably fine.) sudo apt-get install libboost-all-dev
  6. ncurses - sudo apt-get install lib64ncurses-dev.
  7. X11 - sudo apt-get install libx11-dev 
  8. GraphicsMagick - http://www.graphicsmagick.org/README.html
     1. Example: download the latest version 
     2. tar -xf GraphicsMagick-1.3.41.tar.xz
     3. cd GraphicsMagick-1.3.41
     4. ./configure CC=gcc CXX=c++ CFLAGS=-O3 CPPFLAGS=-O3 CXXFLAGS=-O3 LDFLAGS='-L/usr/local/lib'
     5. make check
     6. sudo make install

Install:

```shell
git clone https://github.com/amrobbins/Thor.git
cd Thor
git clone https://github.com/google/googletest.git
cd googletest
mkdir build
cd build
cmake ..
make -j10
cd ../..
make -j10 all
```
