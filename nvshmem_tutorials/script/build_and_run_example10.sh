
BUILD_ROOT=./build
SRC_ROOT=./src

SRC_NAME=example10_moe_dispatch

SOURCES=$(find $SRC_ROOT/$SRC_NAME -type f -name '*.cu' -or -name '*.cpp')

NUM_PES=4

NVCC_GENCODE="arch=compute_90,code=sm_90"

nvcc -rdc=true -ccbin mpicxx -gencode=$NVCC_GENCODE \
-I $NVSHMEM_HOME/include -I $MPI_HOME/include \
-L $NVSHMEM_HOME/lib -L $MPI_HOME/lib \
-lnvshmem -lnvidia-ml -lcuda -lcudart \
-o $BUILD_ROOT/$SRC_NAME $SOURCES

nvshmrun -np $NUM_PES $BUILD_ROOT/$SRC_NAME


