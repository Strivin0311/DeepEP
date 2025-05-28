
export NVSHMEM_IB_ENABLE_IBGDA=1
export NVSHMEM_IBGDA_NIC_HANDLER=gpu
export NVSHMEM_DISABLE_P2P=1 # set to 0 to enable NVLink in low-latency mode
# export NVSHMEM_SYMMETRIC_SIZE=2**30 # default: 1GB

export DEEPEP_TEST_INTRANODE_LOW_LATENCY=0

python tests/test_intranode.py > test_intranode.log 2>&1

# python tests/test_internode.py > test_internode.log 2>&1

# python tests/test_low_latency.py > test_low_latency.log 2>&1