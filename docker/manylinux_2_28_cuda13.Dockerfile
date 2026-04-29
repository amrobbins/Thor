FROM quay.io/pypa/manylinux_2_28_x86_64:latest

ARG CUDA_MAJOR=13
ARG CUDA_MINOR=2

RUN dnf install -y \
      dnf-plugins-core \
      ca-certificates \
      curl \
      file \
      findutils \
      patchelf \
      which \
      binutils \
      git \
      ninja-build \
      cmake \
      ncurses-libs \
      ncurses-devel \
      libgcc \
      libstdc++ \
      libgomp \
      liburing \
      liburing-devel \
    && dnf clean all

RUN dnf config-manager --add-repo \
      https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo \
    && dnf makecache

RUN dnf install -y \
      cuda-toolkit-${CUDA_MAJOR}-${CUDA_MINOR} \
      cuda-libraries-devel-${CUDA_MAJOR}-${CUDA_MINOR} \
      cuda-cudart-devel-${CUDA_MAJOR}-${CUDA_MINOR} \
      cuda-nvrtc-devel-${CUDA_MAJOR}-${CUDA_MINOR} \
      libcublas-devel-${CUDA_MAJOR}-${CUDA_MINOR} \
      libnvjitlink-devel-${CUDA_MAJOR}-${CUDA_MINOR} \
      libcudnn9-devel-cuda-${CUDA_MAJOR} \
    && dnf clean all \
    && rm -rf /var/cache/dnf

ENV CUDA_HOME=/usr/local/cuda-${CUDA_MAJOR}.${CUDA_MINOR}
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

RUN nvcc --version && \
    test -e "${CUDA_HOME}/lib64/libcudart.so" && \
    test -e "${CUDA_HOME}/lib64/libnvrtc.so" && \
    test -e "${CUDA_HOME}/lib64/libcublas.so" && \
    test -e "${CUDA_HOME}/lib64/libcublasLt.so" && \
    test -e "${CUDA_HOME}/lib64/libnvJitLink.so"

RUN /opt/python/cp312-cp312/bin/python -m pip install -U \
      pip \
      build \
      wheel \
      auditwheel \
      nanobind \
      pytest \
      numpy \
      ml_dtypes \
      twine

# manylinux_2_28 is AlmaLinux/RHEL8-like.
# GraphicsMagick++ is in EPEL for EL8.
RUN dnf install -y epel-release && \
    dnf install -y \
      GraphicsMagick \
      GraphicsMagick-devel \
      GraphicsMagick-c++ \
      GraphicsMagick-c++-devel \
    && dnf clean all \
    && rm -rf /var/cache/dnf

RUN dnf install -y boost-devel

RUN dnf install -y libarchive-devel

RUN dnf install -y json-devel

RUN dnf install -y ccache

# Mounted repo is owned by your host user, but container runs as root.
RUN git config --global --add safe.directory /io


# ~/Thor$ sudo docker build -f docker/manylinux_2_28_cuda13.Dockerfile -t thor-manylinux_2_28-cuda13 .

sudo docker run --rm -it --gpus all \
  --privileged \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -v "$PWD":/io \
  -w /io \
  thor-manylinux_2_28-cuda13 \
  bash

## Clean hard, so no local Ubuntu artifacts leak in.
#rm -rf build bindings/python/build bindings/python/dist bindings/python/wheelhouse
#find . -name '*.so' -path '*/build/*' -delete
#
## 1. native
#export CMAKE_GENERATOR=Ninja
#export CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release \
#    -DTHOR_USE_PROJECT_VENV=OFF \
#    -DTHOR_PYTHON_EXECUTABLE=/opt/python/cp312-cp312/bin/python \
#    -DTHOR_PYTHON_AUTO_INSTALL=OFF"
#rm -rf cmake-build-release
#cmake -S . -B cmake-build-release \
#    -G Ninja \
#    -DCMAKE_BUILD_TYPE=Release \
#    -DTHOR_USE_PROJECT_VENV=OFF \
#    -DTHOR_PYTHON_EXECUTABLE=/opt/python/cp312-cp312/bin/python \
#    -DTHOR_PYTHON_AUTO_INSTALL=OFF
#cmake --build cmake-build-release -j 32
#ctest --test-dir cmake-build-release --output-on-failure
#
## 2. wheel
#cd /io/bindings/python
#rm -rf build dist wheelhouse
#/opt/python/cp312-cp312/bin/python -m build --wheel
#export TWINE_USERNAME=__token__
#export TWINE_PASSWORD='pypi-redacted'
#/opt/python/cp312-cp312/bin/python -m twine check dist/*
#/opt/python/cp312-cp312/bin/python -m auditwheel show dist/*.whl
#/opt/python/cp312-cp312/bin/python -m auditwheel repair \
#  --plat manylinux_2_28_x86_64 \
#  --exclude libcublas.so.13 \
#  --exclude libcublasLt.so.13 \
#  --exclude libnvJitLink.so.13 \
#  --exclude libnvrtc.so.13 \
#  --exclude libcudart.so.13 \
#  --exclude libcudnn.so.9 \
#  --exclude libcusolver.so.12 \
#  --exclude libcusparse.so.12 \
#  -w wheelhouse \
#  dist/*.whl
#ls -ltr wheelhouse/
#/opt/python/cp312-cp312/bin/python -m auditwheel show wheelhouse/*.whl
#/opt/python/cp312-cp312/bin/python -m twine check wheelhouse/*
#
## Test the wheel
#/opt/python/cp312-cp312/bin/python -m venv /tmp/thor-wheel-test
#source /tmp/thor-wheel-test/bin/activate
#python -m pip install -U pip pytest numpy ml_dtypes
#python -m pip install wheelhouse/*.whl
#cd /tmp
#python - <<'PY'
#import thor
#print("imported thor from:", thor.__file__)
#print("thor version:", getattr(thor, "__version__", "<missing>"))
#PY
#python -m pytest -q /io/bindings/python/test
#deactivate
#
#cd /io/bindings/python
#/opt/python/cp312-cp312/bin/python -m auditwheel show wheelhouse/*.whl
#/opt/python/cp312-cp312/bin/python -m twine check wheelhouse/*
#/opt/python/cp312-cp312/bin/python -m twine upload wheelhouse/*
