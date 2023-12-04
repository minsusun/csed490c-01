#include <gputk.h>
#include <algorithm>

#define THREADS_PER_BLOCK 512

using namespace std;

#define gpuTKCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      gpuTKLog(ERROR, "Failed to run stmt ", #stmt);                         \
      gpuTKLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)


//@@ Insert code to implement SPMV using JDS with transposed input here 
__global__ void SpMV_JDS_T(int numRows, float *data, int *col_idx, int *col_ptr, int *row_idx, float *B, float *C) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if(row < numRows) {
    float value = 0;
    unsigned int idx = 0;
    while(col_ptr[idx + 1] - col_ptr[idx] > row) {
      value += data[col_ptr[idx] + row] * B[col_idx[idx] + row];
      idx++;
    }
    C[row_idx[row]] = value;
  }
}


int main(int argc, char **argv) {
  gpuTKArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  // float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  int numTotalElems = 0;
  int *cnt_o;
  int *cnt_s;

  float *jds_data;
  int *jds_col_idx;
  int *jds_row_ptr;
  int *jds_row_idx;

  float *jds_t_data;
  int *jds_t_col_idx;
  int *jds_t_col_ptr;
  int *jds_t_row_idx;

  float *device_data;
  int *device_col_idx;
  int *device_col_ptr;
  int *device_row_idx;

  args = gpuTKArg_read(argc, argv);

  gpuTKTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows    = numARows;
  numCColumns = numBColumns;
  //@@ Allocate the hostC matrix
  hostC = (float *)malloc(numCRows * numCColumns * sizeof(float));
  gpuTKTime_stop(Generic, "Importing data and creating memory on host");

  gpuTKLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  gpuTKLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  
  gpuTKTime_start(GPU, "Converting matrix A to JDS format (transposed).");
  //@@ Create JDS format data
  cnt_o = (int *)malloc(numARows * sizeof(int));
  cnt_s = (int *)malloc(numARows * sizeof(int));
  memset(cnt_o, 0, numARows * sizeof(int));

  for(int i = 0; i < numARows; i++) {
    for(int j = 0; j < numAColumns; j++)
      cnt_o[i] += (hostA[i * numAColumns + j] != 0.0);
    cnt_s[i] = cnt_o[i], numTotalElems += cnt_o[i];
  }
  
  sort(cnt_s, cnt_s + numARows, greater<float>());
  
  jds_data = (float *)malloc(numTotalElems * sizeof(int));
  jds_col_idx = (int *)malloc(numTotalElems * sizeof(int));
  jds_row_idx = (int *)malloc(numARows * sizeof(int));
  jds_row_ptr = (int *)malloc((numARows + 1) * sizeof(int));
  
  jds_row_ptr[0] = 0;

  for(int i = 0; i < numARows; i++) jds_t_row_idx[i] = jds_row_idx[i];
  
  for(int i = 0, idx = 0; i < numARows; i++) {
    for(int j = 0; j < numARows; j++) {
      if(cnt_s[i] == cnt_o[j]) {
        jds_row_idx[i] = j;
        for(int k = 0; k < numAColumns; k++) {
          if(hostA[j * numAColumns + k] != 0.0) {
            jds_data[idx] = hostA[j * numAColumns + k];
            jds_col_idx[idx] = k;
            idx++;
          }
        }
        cnt_o[j] = -1;
        jds_row_ptr[i + 1] = idx;
        break;
      }
    }
  }
  
  jds_t_data = (float *)malloc(numTotalElems * sizeof(float));
  jds_t_col_idx = (int *)malloc(numTotalElems * sizeof(int));
  jds_t_row_idx = (int *)malloc(numARows * sizeof(int));
  jds_t_col_ptr = (int *)malloc((numARows + 1) * sizeof(int));
  
  jds_t_col_ptr[0] = 0;
  
  for(int i = 0, idx = 0; i < numAColumns; i++) {
    for(int j = 0; j < numARows && jds_row_ptr[j] + i < jds_row_ptr[j + 1]; j++) {
      jds_t_data[idx] = jds_data[jds_row_ptr[j] + i];
      jds_t_col_idx[idx] = jds_col_idx[jds_row_ptr[j] + i];
      idx++;
    }
    jds_t_col_ptr[i + 1] = idx;
  }
  
  free(cnt_o);
  free(cnt_s);

  free(jds_data);
  free(jds_col_idx);
  free(jds_row_idx);
  free(jds_row_ptr);

  gpuTKTime_stop(GPU, "Converting matirx A to JDS format (transposed).");


  gpuTKTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  gpuTKCheck(cudaMalloc((void **)&device_data, numTotalElems * sizeof(float)));
  gpuTKCheck(cudaMalloc((void **)&device_col_idx, numTotalElems * sizeof(int)));
  gpuTKCheck(cudaMalloc((void **)&device_col_ptr, (numARows + 1) * sizeof(int)));
  gpuTKCheck(cudaMalloc((void **)&device_row_idx, numARows * sizeof(int)));

  gpuTKCheck(cudaMalloc((void **)&deviceB, numBRows * numBColumns * sizeof(float)));
  gpuTKCheck(cudaMalloc((void **)&deviceC, numCRows * numCColumns * sizeof(float)));

  gpuTKTime_stop(GPU, "Allocating GPU memory.");

  gpuTKTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  gpuTKCheck(cudaMemcpy(device_data, jds_t_data, numTotalElems * sizeof(float), cudaMemcpyHostToDevice));
  gpuTKCheck(cudaMemcpy(device_col_idx, jds_t_col_idx, numTotalElems * sizeof(int), cudaMemcpyHostToDevice));
  gpuTKCheck(cudaMemcpy(device_col_ptr, jds_t_col_ptr, (numARows + 1) * sizeof(int), cudaMemcpyHostToDevice));
  gpuTKCheck(cudaMemcpy(device_row_idx, jds_t_row_idx, numARows * sizeof(int), cudaMemcpyHostToDevice));

  gpuTKCheck(cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice));

  gpuTKTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid((numARows - 1) / THREADS_PER_BLOCK + 1, 1, 1);
  dim3 dimBlock(THREADS_PER_BLOCK, 1, 1);

  gpuTKTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  SpMV_JDS_T<<<dimGrid, dimBlock>>>(numARows, device_data, device_col_idx, device_col_ptr, device_row_idx, deviceB, deviceC);

  cudaDeviceSynchronize();
  gpuTKTime_stop(Compute, "Performing CUDA computation");

  gpuTKTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  gpuTKCheck(cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(float), cudaMemcpyHostToDevice));

  gpuTKTime_stop(Copy, "Copying output memory to the CPU");


  gpuTKTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here

  gpuTKTime_stop(GPU, "Freeing GPU Memory");

  gpuTKSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  free(jds_t_data);
  free(jds_t_col_idx);
  free(jds_t_col_ptr);
  free(jds_t_row_idx);

  return 0;
}
