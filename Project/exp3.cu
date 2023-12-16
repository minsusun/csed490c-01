#include <iostream>
#include <string>
#include <cassert>
#include <fstream>
#include <algorithm>
#include <ctime>

#define TILE_WIDTH 32
#define THREADS_PER_BLOCK 512
using namespace std;

#define CUDACheck(stmt)                                                     \
  do {                                                                      \
    cudaError_t err = stmt;                                                 \
    if (err != cudaSuccess) {                                               \
      cout << "Failed to run " << #stmt << endl;                            \
      cout << "Got CUDA error ...  " << cudaGetErrorString(err) << endl;    \
      return -1;                                                            \
    }                                                                       \
  } while (0)

string name = "CUDA(pinned memory/matrix distance calculation/bitonic sort)";

// For float32

__device__ void _bitonicStep1_fp64(double * smem, int tid, int tpp, int d)
{
	int m = tid / (d >> 1);
	int tib = tid - m*(d >> 1);
	int addr1 = d*m + tib;
	int addr2 = (m + 1)*d - tib - 1;

	double A = smem[addr1];
	double B = smem[addr2];
	smem[addr1] = max(A, B);
	smem[addr2] = min(A, B);
}

__device__ void _bitonicStep2_fp64(double * smem, int tid, int tpp, int d)
{
	int m = tid / (d >> 1);
	int tib = tid - m*(d >> 1);
	int addr1 = d*m + tib;
	int addr2 = addr1 + (d >> 1);

	double A = smem[addr1];
	double B = smem[addr2];
	smem[addr1] = max(A, B);
	smem[addr2] = min(A, B);
}

__global__ void bitonicSortKernel128_fp64(double* mem)
{
	// Operating on 64 samples
	int bid = blockIdx.x; // Block UID
	int tpp = threadIdx.x; // Thread position in block
	int tid = blockIdx.x*blockDim.x + threadIdx.x; // Thread global UID
	__shared__ double smem[256]; // Two blocks worth of shared memory
	smem[tpp] = mem[blockDim.x*(2 * bid) + tpp]; // Coalesced memory load
	smem[tpp + blockDim.x] = mem[blockDim.x*((2 * bid) + 1) + tpp]; // Coalesced memory load
	int blocks = 8;
	for (int blockNum = 1; blockNum <= blocks; blockNum++)
	{
		int d = 1 << blockNum;
		_bitonicStep1_fp64(smem, tpp, tpp, d);
		__syncthreads();
		d = d >> 1;
		while (d >= 2)
		{
			_bitonicStep2_fp64(smem, tpp, tpp, d);
			__syncthreads();
			d = d >> 1;
		}
	}

	mem[blockDim.x*(2 * bid) + tpp] = smem[tpp];
	mem[blockDim.x*((2 * bid) + 1) + tpp] = smem[tpp + blockDim.x];
}

__global__ void bitonicSortKernelXBlock1_fp64(double* mem, int blockNum)
{
	int bid = blockIdx.x; // Block UID
	int tpp = threadIdx.x; // Thread position in block
	int tid = blockIdx.x*blockDim.x + threadIdx.x; // Thread global UID
	int d = 1 << blockNum;
	_bitonicStep1_fp64(mem, tid, tpp, d);
}
__global__ void bitonicSortKernelXBlock2_fp64(double* mem, int blockNum, int d)
{
	int bid = blockIdx.x; // Block UID
	int tpp = threadIdx.x; // Thread position in block
	int tid = blockIdx.x*blockDim.x + threadIdx.x; // Thread global UID
	_bitonicStep2_fp64(mem, tid, tpp, d);

}

cudaError_t BitonicSortCUDA(double* mem, int N)
{
	cudaError_t cudaStatus;
	double* dev_mem = mem;
	int numBlocks;

	// Launch a kernel on the GPU with one thread for each element.
	numBlocks = log2((float)N);

	bitonicSortKernel128_fp64 << <N / 256, 128 >> >(dev_mem);
	for (int b = 9; b <= numBlocks; b++)
	{
		int d = 1 << b;
		bitonicSortKernelXBlock1_fp64 << <N / 512, 256 >> >(dev_mem, b);
		d = d >> 1;
		while (d >= 2)
		{
			bitonicSortKernelXBlock2_fp64 << <N / 512, 256 >> >(dev_mem, b, d);
			d = d >> 1;
		}
	}

	//bitonicSortKernelTestDbg <<< N / 256, 128 >>> (dev_mem);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "bitonicSortKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	}
	return cudaStatus;
}

double interval(clock_t *p) {
  clock_t t = clock();
  double result = double(t - *p) / CLOCKS_PER_SEC * 1000;
  *p = t;
  return result;
}

__global__ void matrixDistance(double *X, double *X_t, double *D, int *idxMat, int N, int dim) {
  __shared__ double ds_A[TILE_WIDTH][TILE_WIDTH];
  __shared__ double ds_B[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int Row = by * blockDim.y + ty;
  int Col = bx * blockDim.x + tx;
  
  double Cvalue = 0.0;

  for (int phase = 0; phase < (dim - 1) / TILE_WIDTH + 1; phase++) {
    if (Row < N && phase * TILE_WIDTH + tx < dim) ds_A[ty][tx] = X[Row * dim + phase * TILE_WIDTH + tx];
    else ds_A[ty][tx] = 0.0;

    if (Col < N && phase * TILE_WIDTH + ty < dim) ds_B[ty][tx] = X_t[(phase * TILE_WIDTH + ty) * N + Col];
    else ds_B[ty][tx] = 0.0;

    __syncthreads();

    if (Row < N && Col < N) {
        for (int ii = 0; ii < TILE_WIDTH; ii++) {
            double t = ds_A[ty][ii] - ds_B[ii][tx];
            Cvalue += t * t;
        }
    }

    __syncthreads();
  }

  if (Row < N && Col < N) D[Row * N + Col] = sqrt(Cvalue), idxMat[Row * N + Col] = Col;
}

int main(int argc, char *argv[]) {
  assert(argc == 6);

  int N = atoi(argv[1]);
  int dim = atoi(argv[2]);
  int K = atoi(argv[3]);
  int *IdxMat;
  int *dIdxMat;
  double *X;
  double *X_t;
  double *dX;
  double *dX_t;
  double *D;
  double *dD;
  string title;
  clock_t p;

  CUDACheck(cudaMallocHost((void **)&X, N * dim * sizeof(double)));
  CUDACheck(cudaMallocHost((void **)&X_t, dim * N * sizeof(double)));
  CUDACheck(cudaMallocHost((void **)&D, N * N * sizeof(double)));
  CUDACheck(cudaMallocHost((void **)&IdxMat, N * N * sizeof(int)));

  CUDACheck(cudaMalloc((void **)&dX, N * dim * sizeof(double)));
  CUDACheck(cudaMalloc((void **)&dX_t, dim * N *sizeof(double)));
  CUDACheck(cudaMalloc((void **)&dD, N * N * sizeof(double)));
  CUDACheck(cudaMalloc((void **)&dIdxMat, N * N * sizeof(int)));
  
  cout << name << endl;
  cout << "N=" << N << " dim=" << dim << " K=" << K << " " << argv[4] << endl;

  p = clock();

  ifstream fin;
  fin.open(argv[4]);

  fin >> title;

  for(int i = 0; i < N; i++) {
    for(int j = 0; j < dim; j++) {
      fin >> X[i * dim + j];
      X_t[j * N + i] = X[i * dim + j];
    }
  }

  fin.close();
  
  cout << "step 0:Data Import::" << interval(&p) << "ms" << endl;

  CUDACheck(cudaMemcpy(dX, X, N * dim * sizeof(double), cudaMemcpyHostToDevice));
  CUDACheck(cudaMemcpy(dX_t, X_t, dim * N * sizeof(double), cudaMemcpyHostToDevice));

  cout << "step 1:Memcpy H2D::" << interval(&p) << "ms" << endl;

  dim3 grid_1((N - 1) / TILE_WIDTH + 1, (N - 1) / TILE_WIDTH + 1, 1);
  dim3 block_1(TILE_WIDTH, TILE_WIDTH, 1);
  size_t shm_size = 2 * TILE_WIDTH * TILE_WIDTH * sizeof(double);
  matrixDistance<<<grid_1, block_1, shm_size>>>(dX, dX_t, dD, dIdxMat, N, dim);

  CUDACheck(cudaDeviceSynchronize());

  cout << "step 2:Distance::" << interval(&p) << "ms" << endl;

  dim3 grid_2((N - 1) / THREADS_PER_BLOCK + 1, 1, 1);
  dim3 block_2(THREADS_PER_BLOCK, 1, 1);

  for(int i = 0; i < N; i++) {
    BitonicSortCUDA(dD + i * N, N);
  }

  CUDACheck(cudaDeviceSynchronize());

  cout << "step 3:Sort::" << interval(&p) << "ms" << endl;

  CUDACheck(cudaMemcpy(IdxMat, dIdxMat, N * N * sizeof(int), cudaMemcpyDeviceToHost));

  cout << "step 4:Memcpy D2H::" << interval(&p) << "ms" << endl;

  ofstream fout;
  fout.open(string(argv[5]));

  for(int i = 0; i < N; i++) {
    // omit first one -> i-i pair
    for(int j = 1; j < K + 1; j++) fout << IdxMat[i * N + j] << " ";

    fout << endl;
  }

  fout.close();

  cout << "step 5:Export Result::" << interval(&p) << "ms" << endl;

  cudaFreeHost(X);
  cudaFreeHost(X_t);
  cudaFreeHost(D);
  cudaFreeHost(IdxMat);

  cudaFree(dX);
  cudaFree(dX_t);
  cudaFree(dD);
  cudaFree(dIdxMat);
}