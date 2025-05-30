
BUILD_ROOT=./build
SRC_ROOT=./src

SRC_NAME=example2_ring_shift

NUM_PES=8

NVCC_GENCODE="arch=compute_90,code=sm_90"

nvcc -rdc=true -ccbin g++ -gencode=$NVCC_GENCODE \
-I $NVSHMEM_DIR/include \
-L $NVSHMEM_DIR/lib \
-lnvshmem -lnvidia-ml -lcuda -lcudart \
-o $BUILD_ROOT/$SRC_NAME $SRC_ROOT/$SRC_NAME.cu

nvshmrun -np $NUM_PES $BUILD_ROOT/$SRC_NAME