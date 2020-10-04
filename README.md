
## Thor is a work in progress Machine Learning framework that I am developing.

The goals of this project are:
  1. Create a very high performance AI suite
  2. Allow convenient workflows for development on a local machine and then running for real on an AWS cluster, just switch from SingleMachineExecutor to your configured AwsExecutor.
  3. Create a thoughtful and simple API that's not full of spam
  4. Simplify development by using a compiled, strongly typed, stable programming language - C++.
  5. C++ is also better suited than java because it is faster, it is able to use all the resources of the machine without having to worry about the limits of a virtual machine, and it can directly interface with cuda.
  6. The implementation layer will be C++ but an API layer will be supported in 
     1. C++
     2. Java
     3. Python
  7. Make the defaults the recommended configurations, so that beginners can quickly succeed.
  8. Keep up to date on the latest research and bring the promising pieces into this framework.
  9. Educate the users about the different pieces and strategies, tell them for real don't over academize it. Point them to the latest trends and research. Have a newsletter.


This framework is for Linux, and has been tested on Ubuntu 18.04.

Dependencies, with installation directions for Ubuntu:
  1. cuda 11  - can install with 'sudo apt-get install cuda'
  2. cudnn 8.03 - uses /usr/include/cudnn.h, get it here: https://developer.nvidia.com/cudnn
  3. clang-format  - can install with 'sudo apt-get install clang-format'
  4. boost 1.72 (later is probably fine.) get it here: https://www.boost.org/users/download/#live

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
