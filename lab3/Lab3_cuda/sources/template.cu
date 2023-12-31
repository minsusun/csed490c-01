#include <gputk.h>

#define N_STREAM 32
#define THREADS_PER_BLOCK 128

__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < len) {
    out[index] = in1[index] + in2[index];
  }
}

int main(int argc, char **argv) {
  gpuTKArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  float *deviceInput1;
  float *deviceInput2;
  float *deviceOutput;
  unsigned int numStreams;

  args = gpuTKArg_read(argc, argv);

  gpuTKTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 =
      (float *)gpuTKImport(gpuTKArg_getInputFile(args, 0), &inputLength);
  hostInput2 =
      (float *)gpuTKImport(gpuTKArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));
  gpuTKTime_stop(Generic, "Importing data and creating memory on host");

  gpuTKLog(TRACE, "The input length is ", inputLength);

  gpuTKTime_start(GPU, "Allocating Pinned memory.");

  //@@ Allocate GPU memory here using pinned memory here
  cudaHostRegister(hostInput1, inputLength * sizeof(float), cudaHostRegisterDefault);
  cudaHostRegister(hostInput2, inputLength * sizeof(float), cudaHostRegisterDefault);
  cudaHostRegister(hostOutput, inputLength * sizeof(float), cudaHostRegisterDefault);
  cudaMalloc((void **)&deviceInput1, inputLength * sizeof(float));
  cudaMalloc((void **)&deviceInput2, inputLength * sizeof(float));
  cudaMalloc((void **)&deviceOutput, inputLength * sizeof(float));

  //@@ Create and setup streams 
  //@@ Calculate data segment size of input data processed by each stream 
  cudaStream_t *streams;
  numStreams = N_STREAM;
  streams = (cudaStream_t *)malloc(numStreams * sizeof(cudaStream_t));

  for (int i = 0; i < numStreams; i++) cudaStreamCreate(streams + i);
 
  gpuTKTime_start(Compute, "Performing CUDA computation");
  //@@ Perform parallel vector addition with different streams. 
  size_t segmentSize = (inputLength - 1) / numStreams + 1;
  for (unsigned int s = 0; s<numStreams; s++){
          //@@ Asynchronous copy data to the device memory in segments 
          //@@ Calculate starting and ending indices for per-stream data
          size_t p = s * segmentSize;
          size_t len = (s == numStreams - 1) ? inputLength - p : segmentSize;
          cudaMemcpyAsync(deviceInput1 + p, hostInput1 + p, len * sizeof(float), cudaMemcpyHostToDevice, streams[s]);
          cudaMemcpyAsync(deviceInput2 + p, hostInput2 + p, len * sizeof(float), cudaMemcpyHostToDevice, streams[s]);

          //@@ Invoke CUDA Kernel
          //@@ Determine grid and thread block sizes (consider ococupancy)     
          dim3 gridDim_((len - 1) / THREADS_PER_BLOCK + 1, 1, 1);
          dim3 blockDim_(THREADS_PER_BLOCK, 1, 1);
          vecAdd<<<gridDim_, blockDim_, 0, streams[s]>>>(deviceInput1 + p, deviceInput2 + p, deviceOutput + p, len);

          //@@ Asynchronous copy data from the device memory in segments 
          cudaMemcpyAsync(hostOutput + p, deviceOutput + p, len * sizeof(float), cudaMemcpyDeviceToHost, streams[s]);
  }

  //@@ Synchronize
  cudaDeviceSynchronize();

  gpuTKTime_stop(Compute, "Performing CUDA computation");


  gpuTKTime_start(GPU, "Freeing Pinned Memory");
  //@@ Destory cudaStream
  for (int i = 0; i < numStreams; i++) cudaStreamDestroy(streams[i]);

  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);

  gpuTKTime_stop(GPU, "Freeing Pinned Memory");

  gpuTKSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
