
BUILD_ROOT=./build
SRC_ROOT=./src

SRC_NAME=example9_coalesced_put

NUM_PES=8

NVCC_GENCODE="arch=compute_90,code=sm_90"

nvcc -rdc=true -ccbin mpicxx -gencode=$NVCC_GENCODE \
-I $NVSHMEM_DIR/include -I $MPI_HOME/include \
-L $NVSHMEM_DIR/lib -L $MPI_HOME/lib \
-lnvshmem -lnvidia-ml -lcuda -lcudart \
-o $BUILD_ROOT/$SRC_NAME $SRC_ROOT/$SRC_NAME.cu

nvshmrun -np $NUM_PES $BUILD_ROOT/$SRC_NAME


