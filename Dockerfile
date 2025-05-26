FROM nvcr.io/nvidia/pytorch:25.02-py3

RUN apt-get update && \
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

# 安装 gdrcopy
RUN wget https://github.com/NVIDIA/gdrcopy/archive/refs/tags/v2.4.4.tar.gz && \
    tar zxvf v2.4.4.tar.gz && \
    cd gdrcopy-2.4.4/ && \
    make -j16 && \
    make prefix=/opt/gdrcopy install && \
    cd packages && \ 
    CUDA=/usr/local/cuda ./build-deb-packages.sh && \
    dpkg -i gdrcopy_2.4.4_amd64.Ubuntu22_04.deb \
            libgdrapi_2.4.4_amd64.Ubuntu22_04.deb \
            gdrcopy-tests_2.4.4_amd64.Ubuntu22_04+cuda12.3.deb \
            gdrdrv-dkms_2.4.4_amd64.Ubuntu22_04.deb

# 安装 DeepEP版本的NVSHMEM（注意这是从github拉的master分支，如果DeepEP需要的shemem版本更新了，从nv仓库拉的shemem版本）
RUN mkdir -p /third_party && cd /third_party && \
    git clone https://github.com/deepseek-ai/DeepEP && \
    cd DeepEP && git checkout 0008c6755e1 && cd .. && \
    wget -O nvshmem_src_3.2.5-1.txz https://developer.download.nvidia.com/compute/redist/nvshmem/3.2.5/source/nvshmem_src_3.2.5-1.txz && \
    tar --auto-compress -xvf nvshmem_src_3.2.5-1.txz && \
    cd nvshmem_src && \
    git apply ../DeepEP/third-party/nvshmem.patch && \
    CUDA_HOME=/usr/local/cuda \
    GDRCOPY_HOME=/opt/gdrcopy \
    NVSHMEM_SHMEM_SUPPORT=0 \
    NVSHMEM_UCX_SUPPORT=0 \
    NVSHMEM_USE_NCCL=0 \
    NVSHMEM_IBGDA_SUPPORT=1 \
    NVSHMEM_PMIX_SUPPORT=0 \
    NVSHMEM_TIMEOUT_DEVICE_POLLING=0 \
    NVSHMEM_USE_GDRCOPY=1 \
    cmake -S . -B build/ -DCMAKE_INSTALL_PREFIX=/opt/nvshmem && \
    cd build && \
    make -j$(nproc) && \
    make install

# 设置 NVSHMEM 环境变量
ENV NVSHMEM_DIR=/opt/nvshmem
ENV LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:${LD_LIBRARY_PATH}"
ENV PATH="${NVSHMEM_DIR}/bin:${PATH}"

# 安装 ninja-build 并构建 DeepEP so
RUN apt-get install ninja-build -y && \
    cd /third_party/DeepEP && \
    NVSHMEM_DIR=/opt/nvshmem python setup.py build && \
    ln -s build/lib.linux-x86_64-3.10/deep_ep_cpp.cpython-310-x86_64-linux-gnu.so .

# 设置 DeepEP 的 Python 路径
ENV PYTHONPATH="${PYTHONPATH}:/third_party/DeepEP"