#!/bin/bash

# NOTE: if using ngc-pytorch container, deepep now should be installed at version >= 25.03

CUDA_VERSION=12.8
UBUNTU_VERSION=24_04
GDRCOPY_VERSION=2.5.1
NVSHMEM_VERSION=3.4.5
NVSHMEM_WRAPPER_DIR="/opt/nvshmem"

## Step0: Prerequisites

### eanble nvidia_peermem on the host

sudo modprobe nvidia_peermem

# should output something like `nvidia_peermem 16384  0`, 
# meaning nvidia_peermem module is loaded
lsmod | grep nvidia_peermem

# NOTE: if we need to auto-load when the system boots, 
# we can add the following line to /etc/modules in root privileges:
# sudo su; echo "modprobe nvidia_peermem" >> /etc/rc.local


## step1: install GDRCopy
# NOTE: due to issues including https://github.com/deepseek-ai/DeepEP/issues/11, https://github.com/deepseek-ai/DeepEP/issues/24
# and after issue https://github.com/deepseek-ai/DeepEP/pull/201
# we don't have to install gdrcopy as long as we turn on ibgda

### build gdrcopy on the bare-metal host

wget https://github.com/NVIDIA/gdrcopy/archive/refs/tags/v2.5.1.tar.gz

tar -xvzf v2.5.1.tar.gz

cd gdrcopy-2.5.1/

make -j32

sudo make prefix=/opt/gdrcopy install


### kernel module installation with root privileges on host

pushd packages

sudo apt update
sudo apt install build-essential devscripts debhelper fakeroot pkg-config dkms

CUDA=/usr/local/cuda ./build-deb-packages.sh

# NOTE: the following installation process might seems to be unsuccessful as below:
#   Errors were encountered while processing:
#   gdrdrv-dkms:amd64
#   gdrcopy:amd64
# but it is actually fine
sudo dpkg -i gdrdrv-dkms_2.5.1-1_amd64.Ubuntu24_04.deb \
             libgdrapi_2.5.1-1_amd64.Ubuntu24_04.deb \
             gdrcopy-tests_2.5.1-1_amd64.Ubuntu24_04+cuda12.8.deb \
             gdrcopy_2.5.1-1_amd64.Ubuntu24_04.deb

popd

sudo ./insmod.sh  # Load kernel modules on the bare-metal system

lsmod | grep gdrdrv # should show gdrdrv module loaded, like: `gdrdrv 28672 0`


### Container environment notes

export NVIDIA_SRC_DIR="/usr/src/nvidia-570.148.08/nvidia" # replace with your nvidia driver version

sudo apt update
sudo apt install build-essential devscripts debhelper fakeroot pkg-config dkms

# you might need to reinstall debs in /path/to/gdrcopy-2.5.1/packages on the container
# NOTE: the installation process might seems to be unsuccessful, but it is actually fine
# but it will affect apt-update, so we it's necessary, run:
# sudo dpkg --remove --force-remove-reinstreq gdrdrv-dkms gdrcopy
# to uninstall gdrcopy

cd /path/to/gdrcopy-2.5.1/packages

sudo dpkg -i gdrdrv-dkms_2.5.1-1_amd64.Ubuntu24_04.deb \
             libgdrapi_2.5.1-1_amd64.Ubuntu24_04.deb \
             gdrcopy-tests_2.5.1-1_amd64.Ubuntu24_04+cuda12.8.deb \
             gdrcopy_2.5.1-1_amd64.Ubuntu24_04.deb

# NOTE: keep kernel modules loaded (gdrdrv) on host and might need to restart the container

gdrcopy_copybw  # should show bandwidth test results as belows:

# GPU id:0; name: NVIDIA H100 80GB HBM3; Bus id: 0000:0f:00
# GPU id:1; name: NVIDIA H100 80GB HBM3; Bus id: 0000:10:00
# GPU id:2; name: NVIDIA H100 80GB HBM3; Bus id: 0000:41:00
# GPU id:3; name: NVIDIA H100 80GB HBM3; Bus id: 0000:44:00
# GPU id:4; name: NVIDIA H100 80GB HBM3; Bus id: 0000:a0:00
# GPU id:5; name: NVIDIA H100 80GB HBM3; Bus id: 0000:a1:00
# GPU id:6; name: NVIDIA H100 80GB HBM3; Bus id: 0000:b8:00
# GPU id:7; name: NVIDIA H100 80GB HBM3; Bus id: 0000:bb:00
# selecting device 0
# testing size: 131072
# rounded size: 131072
# gpu alloc fn: cuMemAlloc
# device ptr: 7e51ade00000
# map_d_ptr: 0x7e53cc118000
# info.va: 7e51ade00000
# info.mapped_size: 131072
# info.page_size: 65536
# info.mapped: 1
# info.wc_mapping: 1
# page offset: 0
# user-space pointer:0x7e53cc118000
# writing test, size=131072 offset=0 num_iters=10000
# write BW: 17326.2MB/s
# reading test, size=131072 offset=0 num_iters=100
# read BW: 665.056MB/s
# unmapping buffer
# unpinning buffer
# closing gdrdrv


## Step2: Enable IBGDA on Host

sudo vim /etc/modprobe.d/nvidia.conf

# insert below line and wq
options nvidia NVreg_EnableStreamMemOPs=1 NVreg_RegistryDwords="PeerMappingOverride=1;"

# update initramfs to apply the changes
sudo update-initramfs -u

# reboot
sudo reboot


# after reboot, check if the changes are applied
# you should see something like 
# `options nvidia NVreg_EnableStreamMemOPs=1 NVreg_RegistryDwords="PeerMappingOverride=1;"` 
# in the output
sudo modprobe -c | grep NVreg


## Step3: Build DeepEp-patched NVSHMEM in Container

cd /usr/local/

wget https://developer.download.nvidia.com/compute/nvshmem/3.4.5/local_installers/nvshmem-local-repo-ubuntu2404-3.4.5_3.4.5-1_amd64.deb

sudo dpkg -i nvshmem-local-repo-ubuntu2404-3.4.5_3.4.5-1_amd64.deb

# NOTE: the secret number * will be popped up in the stdout when the cmd above is successfully executed
sudo cp /var/nvshmem-local-repo-ubuntu2404-3.4.5/nvshmem-*-keyring.gpg /usr/share/keyrings/

sudo apt-get update

sudo apt-get -y install nvshmem-cuda-12

# Verify whether nvshmem is installed
dpkg -l | grep nvshmem
find /usr/include -name "nvshmem.h" # for include/
find /usr/include -name "nvshmemx.h" # for include/
find /usr/lib/x86_64-linux-gnu -name "libnvshmem_host.so*" # for lib/
find /usr/lib/x86_64-linux-gnu -name "libnvshmem_device.a*" # for lib/
dpkg -L libnvshmem3-dev-cuda-12 | grep bin # for bin/


# Put nvshmem include/,lib/ and bin/ together into a single home dir

sudo rm -rf "${NVSHMEM_WRAPPER_DIR}"

sudo mkdir -p "${NVSHMEM_WRAPPER_DIR}"

sudo ln -s /usr/include/nvshmem_12 "${NVSHMEM_WRAPPER_DIR}/include"

sudo ln -s /usr/lib/x86_64-linux-gnu/nvshmem/12 "${NVSHMEM_WRAPPER_DIR}/lib"

sudo ln -s /usr/bin/nvshmem_12 "${NVSHMEM_WRAPPER_DIR}/bin"

echo "Checking symlinks in ${NVSHMEM_WRAPPER_DIR}:"
ls -l "${NVSHMEM_WRAPPER_DIR}"/include
ls -l "${NVSHMEM_WRAPPER_DIR}"/lib
ls -l "${NVSHMEM_WRAPPER_DIR}"/bin


# Set path env variables to bashrc
vim ~/.bashrc

# in ~/.bashrc, add belows:
export NVSHMEM_DIR=/opt/nvshmem
export NVSHMEM_HOME=/opt/nvshmem
export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:$LD_LIBRARY_PATH"
export PATH="${NVSHMEM_DIR}/bin:$PATH"

# after wq
source ~/.bashrc

# Should display details of nvshmem / nvshmrun
nvshmem-info -a
nvshmrun -h


# If this doesn't work (e.g. encountering the unexpected low bandwidth like this issue: https://github.com/deepseek-ai/DeepEP/issues/336) 
# please fall back to the original install script for ubuntu22.04 after:
# dpkg -l | grep nvshmem
# apt purge nvshmem-cuda-12 libnvshmem3-cuda-12 libnvshmem3-dev-cuda-12 libnvshmem3-static-cuda-12
# dpkg --purge --force-remove-reinstreq nvshmem-local-repo-ubuntu2404-3.4.5
# rm -rf /opt/nvshmem
# though the authors've said it should work in this issue: https://github.com/deepseek-ai/DeepEP/issues/337


## Step5: install DeepEP

# NOTE: due to some sm_80-related error, in setup.py, we should explicitly add `gencode` for sm90 only to nvcc flags:
# nvcc_flags = ['-O3', '-Xcompiler', '-O3', '-rdc=true', '--ptxas-options=--register-usage-level=10',
#                   '-gencode', 'arch=compute_90,code=sm_90']  # Explicitly specify sm_90

pip uninstall -y deep_ep
pip install -e . -v --no-build-isolation --config-settings editable_mode=strict > logs/install.log 2>&1

