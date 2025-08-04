#!/bin/bash

LOG_ROOT=logs
mkdir -p $LOG_ROOT

export NVSHMEM_IB_ENABLE_IBGDA=1
export NVSHMEM_IBGDA_NIC_HANDLER=gpu
export NVSHMEM_DISABLE_P2P=0 # set to 0 to enable NVLink in low-latency mode
# export NVSHMEM_SYMMETRIC_SIZE=2**30 # default: 1GB


# ----- test-intranode ----- #

# self-added env variable to control low-latency mode for test_intranode.py
# FIXME: enable this wll raise the error:
#   assert calc_diff(recv_x[:, -1], recv_src_info.view(-1)) < 0.007
export DEEPEP_TEST_INTRANODE_LOW_LATENCY=0

# python tests/test_intranode.py > ${LOG_ROOT}/test_intranode.log 2>&1
python tests/test_intranode_kato.py > ${LOG_ROOT}/test_intranode_kato.log 2>&1

# ----- test-internode ----- #

# FIXME: single machine can not run this test due the failed check:
# Assertion error /home/littsk/kato/workspace/cuda-library/deepep/csrc/deep_ep.cpp:32 'num_ranks > NUM_MAX_NVL_PEERS or low_latency_mode'
# python tests/test_internode.py > ${LOG_ROOT}/test_internode.log 2>&1

# ----- test-low-latency ----- #

# FIXME: run this test will raise the error:
#   assert len(durations) % num_kernels_per_period == 0
# python tests/test_low_latency.py > ${LOG_ROOT}/test_low_latency.log 2>&1