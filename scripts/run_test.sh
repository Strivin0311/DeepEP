#!/bin/bash

LOG_ROOT=logs
mkdir -p $LOG_ROOT

# deepep test will set the env vars in the script
# export NVSHMEM_DEBUG=INFO
# export NVSHMEM_IB_ENABLE_IBGDA=1
# export NVSHMEM_IBGDA_NIC_HANDLER=gpu
# export NVSHMEM_DISABLE_P2P=0 # set to 0 to enable NVLink in low-latency mode
# export NVSHMEM_SYMMETRIC_SIZE=2**30 # default: 1GB
# export NVSHMEM_ENABLE_NIC_PE_MAPPING=1
# export NVSHMEM_HCA_LIST="mlx5_0,mlx5_1,^mlx5_2,^mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9"
# export NVSHMEM_HCA_PE_MAPPING="mlx5_0:1:1,mlx5_1:1:1,mlx5_4:1:1,mlx5_5:1:1,mlx5_6:1:1,mlx5_7:1:1,mlx5_8:1:1,mlx5_9:1:1"


TEST_GROUP_COLLECTIVE=true


TEST_SCRIPT_TAG=""
if [[ $TEST_GROUP_COLLECTIVE == true ]]; then
    TEST_SCRIPT_TAG="_grpcoll"
fi


# ----- test-intranode ----- #

# self-added env variable to control low-latency mode for test_intranode.py
# FIXME: enable this wll raise the error:
#   assert calc_diff(recv_x[:, -1], recv_src_info.view(-1)) < 0.007
export DEEPEP_TEST_INTRANODE_LOW_LATENCY=0

# LOG_PATH=${LOG_ROOT}/test_intranode.log
# echo "Logging to ${LOG_PATH} ..."
# python tests/test_intranode.py > ${LOG_PATH} 2>&1

# LOG_PATH=${LOG_ROOT}/test_intranode${TEST_SCRIPT_TAG}_kato.log
# echo "Logging to ${LOG_PATH} ..."
# python tests/test_intranode${TEST_SCRIPT_TAG}_kato.py > ${LOG_PATH} 2>&1; exit 0

# ----- test-low-latency ----- #

# self-added env variable to control allow-nvlink mode for test_low_latency.py
export DEEPEP_TEST_LOW_LATENCY_ALLOW_NVLINK=1

# LOG_PATH=${LOG_ROOT}/test_low_latency.log
# echo "Logging to ${LOG_PATH} ..."
# python tests/test_low_latency.py > ${LOG_PATH} 2>&1

# LOG_PATH=${LOG_ROOT}/test_low_latency${TEST_SCRIPT_TAG}_kato.log
# echo "Logging to ${LOG_PATH} ..."
# python tests/test_low_latency${TEST_SCRIPT_TAG}_kato.py > ${LOG_PATH} 2>&1; exit 0


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
export MASTER_ADDR=10.119.210.141 # replace with your own master node IP found in ifconfig
export MASTER_PORT=23456
export NNODES=2
export NPROC_PER_NODE=8
export RANK=$1

# self-added env variable to control low-latency mode for test_internode.py
export DEEPEP_TEST_INTERNODE_LL_COMPATIBILITY=0

# export WORLD_SIZE=$NNODES
# LOG_PATH=${LOG_ROOT}/test_internode.log
# echo "Logging to ${LOG_PATH} ..."
# python tests/test_internode.py > ${LOG_PATH} 2>&1; exit 0

CMD="torchrun \
--nproc_per_node=$NPROC_PER_NODE \
--nnodes=$NNODES \
--node_rank=$RANK \
--master_addr=$MASTER_ADDR \
--master_port=$MASTER_PORT \
tests/test_internode${TEST_SCRIPT_TAG}_kato.py
"

LOG_PATH=${LOG_ROOT}/test_internode${TEST_SCRIPT_TAG}_kato_n${RANK}.log
echo "Logging to ${LOG_PATH} ..."
$CMD > ${LOG_PATH} 2>&1