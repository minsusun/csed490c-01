#include <gputk.h>

#define gpuTKCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      gpuTKLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define Mask_width 5
#define Mask_radius Mask_width / 2
#define TILE_WIDTH 16
#define w (TILE_WIDTH + Mask_width - 1)
#define clamp(x) (min(max((x), 0.0), 1.0))

//@@ INSERT CODE HERE
__global__ void convolution(float *I, float * __restrict__ M, float *P, int channel, int width, int height) {
  __shared__ float Is[w][w];
  float Pvalue = 0.0f;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  int inputx = blockIdx.x * blockDim.x + tx;
  int inputy = blockIdx.y * blockDim.y + ty;
  int inputz = blockIdx.z * blockDim.z + tz;
  int inputxpb = inputx + TILE_WIDTH;
  int inputypb = inputy + TILE_WIDTH;
  int inputxnb = inputx - TILE_WIDTH;
  int inputynb = inputy - TILE_WIDTH;
  
  Is[Mask_radius + tx][Mask_radius + ty] = (inputx < height && inputy < width) ? I[(inputx * width + inputy) * channel + inputz] : 0.0f;
  
  if(tx < Mask_radius) {
    Is[Mask_radius + TILE_WIDTH + tx][Mask_radius + ty] = (inputxpb >= height)
      ? 0.0f
      : I[(inputxpb * width + inputy) * channel + inputz];
  }
  if(ty < Mask_radius) {
    Is[Mask_radius + tx][Mask_radius + TILE_WIDTH + ty] = (inputypb >= width)
      ? 0.0f
      : I[(inputx * width + inputypb) * channel + inputz];
  }
  if(tx < Mask_radius && ty < Mask_radius) {
    Is[Mask_radius + TILE_WIDTH + tx][Mask_radius + TILE_WIDTH + ty] = (inputxpb >= height || inputypb >= width)
      ? 0.0f
      : I[(inputxpb * width + inputypb) * channel + inputz];
  }
  
  if(tx >= TILE_WIDTH - Mask_radius) {
    Is[Mask_radius + tx - TILE_WIDTH][Mask_radius + ty] = (inputxnb < 0)
      ? 0.0f
      : I[(inputxnb * width + inputy) * channel + inputz];
  }
  if(ty >= TILE_WIDTH - Mask_radius) {
    Is[Mask_radius + tx][Mask_radius + ty - TILE_WIDTH] = (inputynb < 0)
      ? 0.0f
      : I[(inputx * width + inputynb) * channel + inputz];
  }
  if(tx >= TILE_WIDTH - Mask_radius && ty >= TILE_WIDTH - Mask_radius) {
    Is[Mask_radius + tx - TILE_WIDTH][Mask_radius + ty - TILE_WIDTH] = (inputxnb < 0 || inputynb < 0)
      ? 0.0f
      : I[(inputxnb * width + inputynb) * channel + inputz];
  }

  __syncthreads();
  if(inputx < height && inputy < width) {
    for(int i = 0; i < Mask_width; i++) {
      for(int j = 0; j < Mask_width; j++) {
        Pvalue += Is[tx + i][ty + j] * M[i * Mask_width + j];
      }
    }
    P[(inputx * width + inputy) * channel + inputz] = Pvalue;
  }
}

int main(int argc, char *argv[]) {
  gpuTKArg_t arg;
  int maskRows;
  int maskColumns;
  int imageChannels;
  int imageWidth;
  int imageHeight;
  char *inputImageFile;
  char *inputMaskFile;
  gpuTKImage_t inputImage;
  gpuTKImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  float *hostMaskData;
  float *deviceInputImageData;
  float *deviceOutputImageData;
  float *deviceMaskData;

  arg = gpuTKArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = gpuTKArg_getInputFile(arg, 0);
  inputMaskFile  = gpuTKArg_getInputFile(arg, 1);

  inputImage   = gpuTKImport(inputImageFile);
  hostMaskData = (float *)gpuTKImport(inputMaskFile, &maskRows, &maskColumns);

  assert(maskRows == 5);    /* mask height is fixed to 5 in this mp */
  assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

  imageWidth    = gpuTKImage_getWidth(inputImage);
  imageHeight   = gpuTKImage_getHeight(inputImage);
  imageChannels = gpuTKImage_getChannels(inputImage);

  outputImage = gpuTKImage_new(imageWidth, imageHeight, imageChannels);

  hostInputImageData  = gpuTKImage_getData(inputImage);
  hostOutputImageData = gpuTKImage_getData(outputImage);

  gpuTKTime_start(GPU, "Doing GPU Computation (memory + compute)");

  gpuTKTime_start(GPU, "Doing GPU memory allocation");
  //@@ INSERT CODE HERE
  cudaMalloc((void **)&deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **)&deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **)&deviceMaskData, maskRows * maskColumns * sizeof(float));
  gpuTKTime_stop(GPU, "Doing GPU memory allocation");

  gpuTKTime_start(Copy, "Copying data to the GPU");
  //@@ INSERT CODE HERE
  cudaMemcpy(deviceInputImageData, hostInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceMaskData, hostMaskData, maskRows * maskColumns * sizeof(float), cudaMemcpyHostToDevice);
  gpuTKTime_stop(Copy, "Copying data to the GPU");

  gpuTKTime_start(Compute, "Doing the computation on the GPU");
  //@@ INSERT CODE HERE
  dim3 dimGrid((imageHeight - 1)/TILE_WIDTH + 1, (imageWidth - 1)/TILE_WIDTH + 1, imageChannels);
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  convolution<<<dimGrid, dimBlock, w * w * sizeof(float)>>>(deviceInputImageData, deviceMaskData,
                                     deviceOutputImageData, imageChannels,
                                     imageWidth, imageHeight);
  cudaDeviceSynchronize();
  gpuTKTime_stop(Compute, "Doing the computation on the GPU");

  gpuTKTime_start(Copy, "Copying data from the GPU");
  //@@ INSERT CODE HERE
  cudaMemcpy(hostOutputImageData, deviceOutputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float),
             cudaMemcpyDeviceToHost);
  gpuTKTime_stop(Copy, "Copying data from the GPU");

  gpuTKTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  gpuTKSolution(arg, outputImage);

  //@@ Insert code here
  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);
  cudaFree(deviceMaskData);

  free(hostMaskData);
  gpuTKImage_delete(outputImage);
  gpuTKImage_delete(inputImage);

  return 0;
}
