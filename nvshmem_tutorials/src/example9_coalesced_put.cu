#include <iostream>

#include <cuda.h>

#include <nvshmem.h>
#include <nvshmemx.h>


#define WARPSIZE 32
#define BLOCKSIZE blockDim.x

#define TID (blockIdx.x * blockDim.x + threadIdx.x)

#define WARPOFFSET (blockIdx.x * blockDim.x + ((threadIdx.x)>>5) * WARPSIZE)
#define BLOCKOFFSET (blockIdx.x * blockDim.x)
#define GRIDOFFSET (blockDim.x * gridDim.x)


struct GpuTimer {
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start() { cudaEventRecord(start, 0); }

    void Stop() { cudaEventRecord(stop, 0); }

    float ElapsedMillis() {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};


__global__ void int_band_thread(size_t bytes, int *remote_buffer, int *local_buffer, int remote_pe)
{
  uint32_t size = bytes/sizeof(int);
  for(uint32_t i = TID; i<size; i += GRIDOFFSET)
  {
    // each thread moves single element
    nvshmem_int_put(remote_buffer + i, local_buffer + i, 1, remote_pe);
  }
}

__global__ void int_band_warp(size_t bytes, int *remote_buffer, int *local_buffer, int remote_pe)
{
  uint32_t size = bytes/sizeof(int);
  for(uint32_t i = TID; i<size; i += GRIDOFFSET)
  {
    // i = __shfl_sync(0xffffffff, i, 0);
    // __syncwarp();
    // nvshmemx_int_put_warp(remote_buffer + i, local_buffer + i, WARPSIZE, remote_pe);

    // each warp together moves a warp_size of elements
    // where each thread in the same warp should call the nvshmem API with the same arguments
    nvshmemx_int_put_warp(remote_buffer + WARPOFFSET, local_buffer + WARPOFFSET, WARPSIZE, remote_pe);
  }
}

__global__ void int_band_block(size_t bytes, int *remote_buffer, int *local_buffer, int remote_pe)
{
  uint32_t size = bytes/sizeof(int);
  for(uint32_t i = TID; i<size; i += GRIDOFFSET)
  {
    // each block together moves a block_size of elements
    // where each thread in the same block should call the nvshmem API with the same arguments
    nvshmemx_int_put_block(remote_buffer + BLOCKOFFSET, local_buffer + BLOCKOFFSET, BLOCKSIZE, remote_pe);
  }
}

int main()
{
  size_t size = 1<<25; // 32MB
  nvshmem_init();
  int my_pe = nvshmem_my_pe();
  int n_pes = nvshmem_n_pes();

  int dev_count;
  cudaGetDeviceCount(&dev_count);
  cudaSetDevice(my_pe);

  int * remote_buffer = (int *)nvshmem_malloc(sizeof(int)*size*2);
  int * local_buffer;
  cudaMallocManaged(&local_buffer, sizeof(int)*size*2);

  GpuTimer timer;
  float totaltime = 0.0;
  int num_round = 50;
  cudaStream_t *streams;
  streams = (cudaStream_t *)malloc(sizeof(cudaStream_t)*(n_pes-1));
  for(int i = 0; i<n_pes-1; i++ )
      cudaStreamCreateWithFlags(streams+i, cudaStreamNonBlocking);
  size_t bytes = size*sizeof(int);

  //------------------------------------------------------------------------------------------//

  nvshmem_barrier_all();
  if(my_pe == 0)
      std::cout << "\n\nsending "<< bytes << " bytes to all " << n_pes-1 << " GPUs using nvshmem_int_put\n";
  totaltime= 0.0;
  nvshmem_barrier_all();

  for(int i=0; i<num_round; i++)
  {
    int remote_pe = (my_pe+1)%n_pes;
    timer.Start();
    for(int j=0; j<n_pes-1; j++)
    {
      int_band_thread<<<80, 512, 0, streams[j]>>>(bytes, remote_buffer, local_buffer, remote_pe);
      remote_pe = (remote_pe+1) % n_pes;
    }
    cudaDeviceSynchronize();
    timer.Stop();
    totaltime = totaltime + timer.ElapsedMillis();
  }

  nvshmem_barrier_all();
  totaltime = totaltime/num_round;
  std::cout <<"PE "<< my_pe <<  " average time: " <<  totaltime << " bandwidth: "<<(sizeof(int)*size*(n_pes-1)/(totaltime/1000))/(1024*1024*1024)<<" GB/s" << std::endl;

  //------------------------------------------------------------------------------------------//

  nvshmem_barrier_all();
  if(my_pe == 0)
      std::cout << "\n\nsending "<< bytes << " bytes to all " << n_pes-1 << " GPUs using nvshmemx_int_put_warp\n";
  totaltime= 0.0;
  nvshmem_barrier_all();

  for(int i=0; i<num_round; i++)
  {
    int remote_pe = (my_pe+1)%n_pes;
    timer.Start();
    for(int j=0; j<n_pes-1; j++)
    {
      int_band_warp<<<80, 512, 0, streams[j]>>>(bytes, remote_buffer, local_buffer, remote_pe);
      remote_pe = (remote_pe+1) % n_pes;
    }
    cudaDeviceSynchronize();
    timer.Stop();
    totaltime = totaltime + timer.ElapsedMillis();
  }

  nvshmem_barrier_all();
  totaltime = totaltime/num_round;
  std::cout <<"PE "<< my_pe <<  " average time: " <<  totaltime << " bandwidth: "<<(sizeof(int)*size*(n_pes-1)/(totaltime/1000))/(1024*1024*1024)<<" GB/s" << std::endl;

  //------------------------------------------------------------------------------------------//

  nvshmem_barrier_all();
  if(my_pe == 0)
      std::cout << "\n\nsending "<< bytes << " bytes to all " << n_pes-1 << " GPUs using nvshmemx_int_put_block\n";
  totaltime= 0.0;
  nvshmem_barrier_all();

  for(int i=0; i<num_round; i++)
  {
    int remote_pe = (my_pe+1)%n_pes;
    timer.Start();
    for(int j=0; j<n_pes-1; j++)
    {
      int_band_block<<<80, 512, 0, streams[j]>>>(bytes, remote_buffer, local_buffer, remote_pe);
      remote_pe = (remote_pe+1) % n_pes;
    }
    cudaDeviceSynchronize();
    timer.Stop();
    totaltime = totaltime + timer.ElapsedMillis();
  }

  nvshmem_barrier_all();
  totaltime = totaltime/num_round;
  std::cout <<"PE "<< my_pe <<  " average time: " <<  totaltime << " bandwidth: "<<(sizeof(int)*size*(n_pes-1)/(totaltime/1000))/(1024*1024*1024)<<" GB/s" << std::endl;

  //------------------------------------------------------------------------------------------//

  cudaDeviceSynchronize();

  for(int i = 0; i<n_pes-1; i++)
      cudaStreamDestroy(streams[i]);

  nvshmem_free(remote_buffer);
  cudaFree(local_buffer);

  nvshmem_finalize();

  std::cout << "end of the program: "<< my_pe << std::endl;

  return 0;
}