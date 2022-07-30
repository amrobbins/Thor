
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
  1. cuda 11  - can install with 'sudo apt-get install cuda'
  2. cudnn 8.2 - get it here: https://developer.nvidia.com/cudnn
  3. clang-format  - can install with 'sudo apt-get install clang-format'
  4. boost 1.72 (later is probably fine.) get it here: https://www.boost.org/users/download/#live
  5. GraphicsMagick - http://www.graphicsmagick.org/README.html
  6. ncurses

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
