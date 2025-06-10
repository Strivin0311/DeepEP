
BUILD_ROOT=./build
SRC_ROOT=./src

SRC_NAME=example3_ring_shift_with_mpi

NUM_RANKS=8

NVCC_GENCODE="arch=compute_90,code=sm_90"

nvcc -rdc=true -ccbin mpicxx -gencode=$NVCC_GENCODE \
-I $NVSHMEM_HOME/include -I $MPI_HOME/include \
-L $NVSHMEM_HOME/lib -L $MPI_HOME/lib \
-lnvshmem -lnvidia-ml -lcuda -lcudart \
-o $BUILD_ROOT/$SRC_NAME $SRC_ROOT/$SRC_NAME.cu

mpirun --allow-run-as-root -np $NUM_RANKS $BUILD_ROOT/$SRC_NAME