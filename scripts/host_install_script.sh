#!/bin/bash
# 设置错误时退出
set -e
# 更新软件包并安装依赖
apt-get update
apt-get install -y --no-install-recommends \
    wget \
    kmod \
    build-essential \
    linux-headers-$(uname -r) \
    dkms \
    libnuma-dev \
    pkg-config \
    devscripts \
    dkms \
    debhelper
# 解决没有nv头文件的问题
proxy="http://10.1.4.213:3128"
export https_proxy=$proxy
export http_proxy=$proxy
export ftp_proxy=$proxy
wget https://us.download.nvidia.com/XFree86/Linux-x86_64/565.57.01/NVIDIA-Linux-x86_64-565.57.01.run
chmod +x NVIDIA-Linux-x86_64-565.57.01.run
./NVIDIA-Linux-x86_64-565.57.01.run --extract-only
cd NVIDIA-Linux-x86_64-565.57.01
mkdir -p /usr/src/nvidia-565.57.01
cp -r kernel/* /usr/src/nvidia-565.57.01
cd ..
# 安装 gdrcopy
wget https://github.com/NVIDIA/gdrcopy/archive/refs/tags/v2.4.4.tar.gz
tar zxvf v2.4.4.tar.gz
cd gdrcopy-2.4.4/
make -j16
make prefix=/opt/gdrcopy install
cd packages
CUDA=/usr/local/cuda ./build-deb-packages.sh
dpkg -i gdrcopy_2.4.4_amd64.Ubuntu22_04.deb \
        libgdrapi_2.4.4_amd64.Ubuntu22_04.deb \
        gdrcopy-tests_2.4.4_amd64.Ubuntu22_04+cuda12.4.deb \
        gdrdrv-dkms_2.4.4_amd64.Ubuntu22_04.deb
cd ../..
# 安装 DeepEP 和 NVSHMEM
cd /tmp
git clone https://github.com/deepseek-ai/DeepEP
cd DeepEP
git checkout 0008c6755e173fa11
cd ..
wget -O nvshmem_src_3.2.5-1.txz https://developer.download.nvidia.com/compute/redist/nvshmem/3.2.5/source/nvshmem_src_3.2.5-1.txz
tar --auto-compress -xvf nvshmem_src_3.2.5-1.txz
cd nvshmem_src
git apply ../DeepEP/third-party/nvshmem.patch
# 配置 NVIDIA 驱动以启用 IBGDA
echo 'options nvidia NVreg_EnableStreamMemOPs=1 NVreg_RegistryDwords="PeerMappingOverride=1;"' > /etc/modprobe.d/nvidia.conf
update-initramfs -u
# 编译安装 NVSHMEM
CUDA_HOME=/usr/local/cuda \
GDRCOPY_HOME=/opt/gdrcopy \
NVSHMEM_SHMEM_SUPPORT=0 \
NVSHMEM_UCX_SUPPORT=0 \
NVSHMEM_USE_NCCL=0 \
NVSHMEM_IBGDA_SUPPORT=1 \
NVSHMEM_PMIX_SUPPORT=0 \
NVSHMEM_TIMEOUT_DEVICE_POLLING=0 \
NVSHMEM_USE_GDRCOPY=1 \
cmake -S . -B build/ -DCMAKE_INSTALL_PREFIX=/opt/nvshmem
cd build
make -j$(nproc)
make install
# 设置 NVSHMEM 环境变量
echo 'export NVSHMEM_DIR=/opt/nvshmem' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:${LD_LIBRARY_PATH}"' >> ~/.bashrc
echo 'export PATH="${NVSHMEM_DIR}/bin:${PATH}"' >> ~/.bashrc
echo "安装完成！请重启机器使IBDGA生效！如果需要在host机器上运行DeepEP，请运行 'source ~/.bashrc' 使环境变量生效"