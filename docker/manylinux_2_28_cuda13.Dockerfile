FROM quay.io/pypa/manylinux_2_28_x86_64:latest

ARG CUDA_MAJOR=13
ARG CUDA_MINOR=3

ENV PIP_ROOT_USER_ACTION=ignore
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

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
      libgcc \
      libstdc++ \
      libgomp \
      liburing \
      liburing-devel \
      openssl-devel \
    && dnf clean all

ARG NLOHMANN_JSON_VERSION=3.11.3

RUN curl -L \
      "https://github.com/nlohmann/json/releases/download/v${NLOHMANN_JSON_VERSION}/json.tar.xz" \
      -o /tmp/json.tar.xz && \
    mkdir -p /tmp/json-src && \
    tar -xf /tmp/json.tar.xz -C /tmp/json-src --strip-components=1 && \
    cmake -S /tmp/json-src -B /tmp/json-build \
      -DCMAKE_BUILD_TYPE=Release \
      -DJSON_BuildTests=OFF \
      -DCMAKE_INSTALL_PREFIX=/usr/local && \
    cmake --build /tmp/json-build --target install && \
    rm -rf /tmp/json-src /tmp/json-build /tmp/json.tar.xz

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
      twine \
      nvidia-cudnn-frontend==1.23.0

RUN dnf install -y libarchive-devel

RUN dnf install -y ccache

# Mounted repo is owned by your host user, but container runs as root.
RUN git config --global --add safe.directory /io


#~/Thor$ sudo docker build -f docker/manylinux_2_28_cuda13.Dockerfile -t thor-manylinux_2_28-cuda13 .
#
#sudo docker run --rm -it --gpus all \
#  --privileged \
#  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
#  -v "$PWD":/io \
#  -w /io \
#  thor-manylinux_2_28-cuda13 \
#  bash

## Clean hard, so no local Ubuntu artifacts leak in.
#rm -rf build bindings/python/build bindings/python/dist bindings/python/wheelhouse
#find . -name '*.so' -path '*/build/*' -delete
#
## 1. native
#/opt/python/cp312-cp312/bin/python -m pip install -U pip setuptools wheel
#/opt/python/cp312-cp312/bin/python -m pip install \
#  nvidia-cuda-runtime==13.3.29 \
#  nvidia-cuda-nvrtc==13.3.33 \
#  nvidia-nvjitlink==13.3.33 \
#  nvidia-cublas==13.5.1.27 \
#  nvidia-cusparse==12.8.1.7 \
#  nvidia-cusolver==12.2.2.18 \
#  nvidia-cuda-cccl==13.3.3.3.1 \
#  nvidia-cudnn-cu13==9.23.2.1 \
#  nvidia-cudnn-frontend==1.25.0
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
#cmake --build cmake-build-release -j 32 && ctest --test-dir cmake-build-release --output-on-failure
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
#python - <<'PY'
#from pathlib import Path
#import zipfile
#
#wheel = next(Path("wheelhouse").glob("*.whl"))
#with zipfile.ZipFile(wheel) as z:
#    metadata_name = next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))
#    metadata = z.read(metadata_name).decode()
#    print(metadata_name)
#    for line in metadata.splitlines():
#        if line.startswith("Requires-Dist: nvidia") or line.startswith("Requires-Dist: cuda-toolkit"):
#            print(line)
#
#    resolved = z.read("thor/_cuda_stack_resolved.py").decode()
#    print("\n--- resolved stack ---")
#    for line in resolved.splitlines():
#        if "CudaDistribution(" in line:
#            print(line.strip())
#PY
#
#/opt/python/cp312-cp312/bin/python -m venv /tmp/thor-wheel-test
#source /tmp/thor-wheel-test/bin/activate
#python -m pip install -U pip pytest
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
