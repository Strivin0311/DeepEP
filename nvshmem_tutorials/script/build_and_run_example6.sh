
BUILD_ROOT=./build
SRC_ROOT=./src

SRC_NAME=example6_ring_shift_block

NUM_PES=8
NUM_RANKS=$NUM_PES

NVCC_GENCODE="arch=compute_90,code=sm_90"

nvcc -rdc=true -ccbin mpicxx -gencode=$NVCC_GENCODE \
-I $NVSHMEM_HOME/include -I $MPI_HOME/include \
-L $NVSHMEM_HOME/lib -L $MPI_HOME/lib \
-lnvshmem -lnvidia-ml -lcuda -lcudart \
-o $BUILD_ROOT/$SRC_NAME $SRC_ROOT/$SRC_NAME.cu

export NVSHMEMTEST_USE_MPI_LAUNCHER=0

if [[ $NVSHMEMTEST_USE_MPI_LAUNCHER -eq 1 ]]; then
    mpirun --allow-run-as-root -np $NUM_RANKS $BUILD_ROOT/$SRC_NAME
else
    nvshmrun -np $NUM_PES $BUILD_ROOT/$SRC_NAME
fi


