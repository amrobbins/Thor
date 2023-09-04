git clone https://github.com/google/googletest.git
cd googletest
git checkout release-1.12.1
mkdir build
cd build
cmake ..
make -j
cd ../..
