<div align="center">
<img src="logo.png" title="Thor" alt="Tensor Hyper-Parallel Optimized pRocessing" width="360" height="360">
</div>

# Thor
## (Tensor Hyper-parallel Optimized pRocessing)

Design Objectives:
1. Efficiency
2. Performance
3. Scaling
4. Ease of use
5. Full featured

This framework is for Linux, and is currently being developed using Ubuntu 22.04 and Cuda 12.2, using an Nvidia GPU of compute capability >= 7.5.

## Set up Ubuntu 22.04 machine and build:

Cuda of at least 12 is supported. If your machine does not have cuda12, cudnn9 and oneAPI set up, then you can do it this way:

``shell
git clone https://github.com/amrobbins/Thor.git
``

```shell
bash Thor/MachineSetup/install_nvidia_driver.sh
sudo reboot
```

```shell
bash Thor/MachineSetup/install_cuda.sh
sudo reboot
```

```shell
bash Thor/MachineSetup/install_cudnn.sh
bash Thor/MachineSetup/install_oneAPI.sh
sudo reboot
```

If you have cuda12 and cudnn9 already, but don't have oneAPI:

```shell
bash Thor/MachineSetup/install_oneAPI.sh
sudo reboot
```

Now that the machine is set up:

```shell
bash Thor/MachineSetup/install_dependencies.sh
cd Thor
bash MachineSetup/install_google_test.sh
make -j all
```
