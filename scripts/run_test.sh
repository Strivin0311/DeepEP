#!/bin/bash

LOG_ROOT=logs
mkdir -p $LOG_ROOT

# deepep test will set the env vars in the script
# export NVSHMEM_IB_ENABLE_IBGDA=1
# export NVSHMEM_IBGDA_NIC_HANDLER=gpu
# export NVSHMEM_DISABLE_P2P=0 # set to 0 to enable NVLink in low-latency mode
# export NVSHMEM_SYMMETRIC_SIZE=2**30 # default: 1GB


# ----- test-intranode ----- #

# self-added env variable to control low-latency mode for test_intranode.py
# FIXME: enable this wll raise the error:
#   assert calc_diff(recv_x[:, -1], recv_src_info.view(-1)) < 0.007
export DEEPEP_TEST_INTRANODE_LOW_LATENCY=0

# python tests/test_intranode.py > ${LOG_ROOT}/test_intranode.log 2>&1
# python tests/test_intranode_kato.py > ${LOG_ROOT}/test_intranode_kato.log 2>&1; exit 0

# ----- test-low-latency ----- #

# self-added env variable to control allow-nvlink mode for test_low_latency.py
export DEEPEP_TEST_LOW_LATENCY_ALLOW_NVLINK=1

# python tests/test_low_latency.py > ${LOG_ROOT}/test_low_latency.log 2>&1
# python tests/test_low_latency_kato.py > ${LOG_ROOT}/test_low_latency_kato.log 2>&1; exit 0


# ----- test-internode ----- #

if [ -z "$1" ]; then
    echo "Error: Please specify the rank of this node."
    echo "Usage: ./run_distributed.sh <rank>"
    echo "Example: ./run_distributed.sh 0  (for master node 0)"
    exit 1
else
    echo "Launch with node rank: $1"
fi

# init dist env vars
export OMP_NUM_THREADS=1
export MASTER_ADDR=10.119.210.140 # replace with your own master node IP
export MASTER_PORT=23457
export NNODES=2
export NPROC_PER_NODE=8
export RANK=$1

# self-added env variable to control low-latency mode for test_internode.py
export DEEPEP_TEST_INTERNODE_LL_COMPATIBILITY=0

# python tests/test_internode.py > ${LOG_ROOT}/test_internode.log 2>&1

CMD="torchrun \
--nproc_per_node=$NPROC_PER_NODE \
--nnodes=$NNODES \
--node_rank=$RANK \
--master_addr=$MASTER_ADDR \
--master_port=$MASTER_PORT \
tests/test_internode_kato.py
"

$CMD > ${LOG_ROOT}/test_internode_kato_n${RANK}.log 2>&1