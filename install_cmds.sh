#!/bin/bash

## Step0: Prerequisites

### eanble nvidia_peermem on the host

sudo modprobe nvidia_peermem

lsmod | grep nvidia_peermem # should show nvidia_peermem module loaded

# NOTE: if we need to auto-load when the system boots, we can add the following line to /etc/modules:
# sudo echo "modprobe nvidia_peermem" >>/etc/rc.local


## step1: install GDRCopy

### build gdrcopy on the bare-metal host

wget https://github.com/NVIDIA/gdrcopy/archive/refs/tags/v2.4.4.tar.gz

tar -xvzf v2.4.4.tar.gz

cd gdrcopy-2.4.4/

make -j32

sudo make prefix=/opt/gdrcopy install


### kernel module installation with root privileges on host

pushd packages

sudo apt install build-essential devscripts debhelper fakeroot pkg-config dkms

CUDA=/usr/local/cuda ./build-deb-packages.sh

sudo dpkg -i gdrdrv-dkms_2.4.4_amd64.Ubuntu22_04.deb \
             libgdrapi_2.4.4_amd64.Ubuntu22_04.deb \
             gdrcopy-tests_2.4.4_amd64.Ubuntu22_04+cuda12.4.deb \
             gdrcopy_2.4.4_amd64.Ubuntu22_04.deb

popd

sudo ./insmod.sh  # Load kernel modules on the bare-metal system


### Container environment notes

# NOTE: keep kernel modules loaded (gdrdrv) on host
# and might need to restart the container

gdrcopy_copybw  # should show bandwidth test results as belows:

# GPU id:0; name: Tesla V100-SXM2-32GB; Bus id: 0000:06:00
# GPU id:1; name: Tesla V100-SXM2-32GB; Bus id: 0000:07:00
# GPU id:2; name: Tesla V100-SXM2-32GB; Bus id: 0000:0a:00
# GPU id:3; name: Tesla V100-SXM2-32GB; Bus id: 0000:0b:00
# GPU id:4; name: Tesla V100-SXM2-32GB; Bus id: 0000:85:00
# GPU id:5; name: Tesla V100-SXM2-32GB; Bus id: 0000:86:00
# GPU id:6; name: Tesla V100-SXM2-32GB; Bus id: 0000:89:00
# GPU id:7; name: Tesla V100-SXM2-32GB; Bus id: 0000:8a:00
# selecting device 0
# testing size: 131072
# rounded size: 131072
# gpu alloc fn: cuMemAlloc
# device ptr: 7f1153a00000
# map_d_ptr: 0x7f1172257000
# info.va: 7f1153a00000
# info.mapped_size: 131072
# info.page_size: 65536
# info.mapped: 1
# info.wc_mapping: 1
# page offset: 0
# user-space pointer:0x7f1172257000
# writing test, size=131072 offset=0 num_iters=10000
# write BW: 9638.54MB/s
# reading test, size=131072 offset=0 num_iters=100
# read BW: 530.135MB/s
# unmapping buffer
# unpinning buffer
# closing gdrdrv


## Step2: Enable IBGDA on Host

sudo vim /etc/modprobe.d/nvidia.conf

# insert below line and wq
options nvidia NVreg_EnableStreamMemOPs=1 NVreg_RegistryDwords="PeerMappingOverride=1;"


sudo update-initramfs -u
sudo reboot


## Step3: Build DeepEp-patched NVSHMEM

### get nvshmem src

cd /usr/local/

wget https://developer.nvidia.com/downloads/assets/secure/nvshmem/nvshmem_src_3.2.5-1.txz

mkdir nvshmem_src_3.2.5-1

tar -xvf nvshmem_src_3.2.5-1.txz -C nvshmem_src_3.2.5-1

cd nvshmem_src_3.2.5-1/nvshmem_src


### patch nvshmem src with deepep patch

git apply /path/to/deep_ep/dir/third-party/nvshmem.patch


### cmake nvshmem with IBGDA support

# NOTE: `-D MLX5_lib=/usr/lib/x86_64-linux-gnu/libmlx5.so.1` is custom due to error: 
# Please set them or make sure they are set and tested correctly in the CMake files:
# MLX5_lib
#     linked by target "nvshmem_transport_ibgda" in directory /usr/local/nvshmem_src_3.2.5-1/nvshmem_src/src

# NOTE: due to `include "mpi.h": no such file or directory` error
# we change the CMakeLists.txt as belows:

# => commend these lines:
# if(NVSHMEM_MPI_SUPPORT)
#   find_package(MPI REQUIRED)
# endif()

# => add these lines:
# find_package(MPI REQUIRED)
# include_directories(${MPI_INCLUDE_PATH})


CUDA_HOME=/usr/local/cuda \
GDRCOPY_HOME=/home/littsk/kato/gdrcopy \
NVSHMEM_SHMEM_SUPPORT=0 \
NVSHMEM_UCX_SUPPORT=0 \
NVSHMEM_USE_NCCL=0 \
NVSHMEM_IBGDA_SUPPORT=1 \
NVSHMEM_PMIX_SUPPORT=0 \
NVSHMEM_TIMEOUT_DEVICE_POLLING=0 \
NVSHMEM_USE_GDRCOPY=1 \
cmake -S . -B build/ -DCMAKE_INSTALL_PREFIX=/opt/nvshmem -D MLX5_lib=/usr/lib/x86_64-linux-gnu/libmlx5.so.1 \
-D CMAKE_CXX_FLAGS="-I/usr/local/mpi/include" \
-D CMAKE_C_FLAGS="-I/usr/local/mpi/include"


### build and install nvshmem

cd build

make -j32

make install


### install nvshmrun script

bash scripts/install_hydra.sh /usr/local/nvshmem_src_3.2.5-1/ /usr/local


## Step4: Post-installation configuration

vim ~/.bashrc

# in ~/.bashrc, add belows:
export NVSHMEM_DIR=/opt/nvshmem
export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:$LD_LIBRARY_PATH"
export PATH="${NVSHMEM_DIR}/bin:$PATH"

# after wq
source ~/.bashrc

nvshmem-info -a # Should display details of nvshmem


## Step5: install DeepEP

# NOTE: in setup.py, we should explicitly add `gencode` for sm90 only to nvcc flags:
# nvcc_flags = ['-O3', '-Xcompiler', '-O3', '-rdc=true', '--ptxas-options=--register-usage-level=10',
#                   '-gencode', 'arch=compute_90,code=sm_90']  # Explicitly specify sm_90

NVSHMEM_DIR=/opt/nvshmem pip install -e . --no-build-isolation --config-settings editable_mode=strict

