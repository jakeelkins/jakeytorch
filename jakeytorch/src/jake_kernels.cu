#include <helper_cuda.h>
#include <jake_kernels.h>
#include <cuda/atomic>
#include <cmath>

// relu
__global__ void relu_kernel (float* A_out, float* A_in, unsigned int N) {
  // A is vector of dim N
  for (unsigned int i = blockIdx.x*blockDim.x + threadIdx.x; i < N; i += blockDim.x*gridDim.x){
    float in_ = A_in[i];
    if (in_ > 0.0f){
        A_out[i] = in_;
    } else {
        A_out[i] = 0.0f;
    }
  }
}

__global__ void relu_prime_kernel (float* A_out, float* A_in, unsigned int N) {
  // A is vector of dim N
  for (unsigned int i = blockIdx.x*blockDim.x + threadIdx.x; i < N; i += blockDim.x*gridDim.x){
    float in_ = A_in[i];
    if (in_ > 0.0f){
        A_out[i] = 1.0f;
    } else {
        A_out[i] = 0.0f;
    }
  }
}

__global__ void log_kernel (float* A_out, float* A_in, unsigned int N) {
  // A is vector of dim N
  // used in cross entropy
  for (unsigned int i = blockIdx.x*blockDim.x + threadIdx.x; i < N; i += blockDim.x*gridDim.x){
    A_out[i] = log(A_in[i]);
  }
}

__global__ void exp_kernel (float* A_out, float* A_in, unsigned int N) {
  // A is vector of dim N
  // used in our softmax
  for (unsigned int i = blockIdx.x*blockDim.x + threadIdx.x; i < N; i += blockDim.x*gridDim.x){
    A_out[i] = exp(A_in[i]);
  }
}

__global__ void batched_elementwise_divide_kernel (float* A, float* b, unsigned int N, unsigned int batch_size) {
  // A is vector of dim N*batch_size
  // b is vector of dim batch_size
  // do A = A/b[batch]
  // used in our softmax
  // ASSUMES block_dim = N, num_blocks = batch_size
  for (unsigned int i = blockIdx.x*blockDim.x + threadIdx.x; i < batch_size*N; i += blockDim.x*gridDim.x){
    A[i] = A[i]/b[blockIdx.x];
  }
}

__global__ void inplace_elementwise_multiplication_kernel (float* A, float* B, unsigned int N, unsigned int M) {
  // A = A .* B, A and B NxM
  // used in our cross entropy
  for (unsigned int i = blockIdx.x*blockDim.x + threadIdx.x; i < N*M; i += blockDim.x*gridDim.x){
    A[i] = A[i]*B[i];
  }
}

__global__ void elementwise_multiplication_kernel (float* A, float* B, float* C, unsigned int N) {
  // A = B .* C, all vectors of length N
  // used in backprop
  for (unsigned int i = blockIdx.x*blockDim.x + threadIdx.x; i < N; i += blockDim.x*gridDim.x){
    A[i] = B[i]*C[i];
  }
}

__global__ void elementwise_subtraction_kernel (float* x, float* y, float* z, unsigned int N) {
  // x = y - z, all dim N
  // used in our cross entropy derivative
  for (unsigned int i = blockIdx.x*blockDim.x + threadIdx.x; i < N; i += blockDim.x*gridDim.x){
    x[i] = y[i] - z[i];
  }
}

__global__ void custom_batch_mean_reduce_kernel(float* A_mean, float* A_batched, unsigned int N, unsigned int M, unsigned int batch_size) {
  // used for finding mean of gradients by batch
  // reduces along batch dimension
  // A is NxM in batches.
  // assumes num threads = N*M
  for (unsigned int i = blockIdx.x*blockDim.x + threadIdx.x; i < N*M; i += blockDim.x*gridDim.x){

    // first: put sum into first batch of A_mean.
    // should be threadsafe the way I have it
    if (i < N*M){
      for (unsigned int batch = 0; batch<batch_size; ++batch){
        A_mean[i] += A_batched[batch*N*M + i];
      }
    }
    __syncthreads();

    // now divide first batch of A_mean to get actual mean
    if (i < N*M){
      A_mean[i] = A_mean[i]/batch_size;
    }

    __syncthreads();

    // copy out first batch into rest of the batches
    if (i < N*M){
      for (unsigned int batch = 1; batch<batch_size; ++batch){
        A_mean[batch*N*M + i] = A_mean[i];
      }
    }
  }
}

// __global__ void custom_batch_mean_reduce_kernel(float* A_mean, float* A_batched, unsigned int N, unsigned int M, unsigned int batch_size) {
//   // used for finding mean of gradients by batch
//   // reduces along batch dimension
//   // A is NxM in batches.
//   // assumes num threads = N*M
//   for (unsigned int i = blockIdx.x*blockDim.x + threadIdx.x; i < N*M; i += blockDim.x*gridDim.x){

//     // first: put sum into A_mean.
//     // should be threadsafe the way I have it
//     if (i < N*M){
//       for (unsigned int batch = 0; batch<batch_size; ++batch){
//         A_mean[i] += A_batched[batch*N*M + i];
//       }
//     }
//     __syncthreads();

//     if (i < N*M){
//       A_mean[i] = A_mean[i]/batch_size;
//     }
//   }

// }

// __global__ void softmax_kernel(float* y, float* x, int N, int batch_size){
//   // x is input of length batch_size*N, y is output. 

//   // ASSUMED: block size is N (which is output dim here).
//   unsigned int batch = blockIdx.x*blockDim.x
//   unsigned int i = batch + threadIdx.x

//   // put sum_exp_x in shared
//   __shared__ float sum_exp_x[1];

//   // first: raise all of x to exp, put it in y.
//   if (i < batch_size*N){
//     y[i] = exp(x[i]);
//   }
  
//   __syncthreads();

//   // atomic add the sum of the exponentials in each block
//   if (i < batch_size*N){
//     atomicAdd(&sum_exp_x[0], y[i]);
//   }

//   __syncthreads();

//   // now, elementwise divide by the respective sum_exp_x
//   if (i < batch_size*N){
//     y[i] = y[i]/sum_exp_x[0];
//   }
// }

void relu(float* A_out, float* A_in, unsigned int N){

  int blockSize1D;
  int gridSize;

  blockSize1D = 32;
  gridSize = (N + blockSize1D - 1)/blockSize1D;

  relu_kernel<<<gridSize, blockSize1D>>> (A_out, A_in, N); 
  
  checkCudaErrors(cudaGetLastError());

}

void relu_prime(float* A_out, float* A_in, unsigned int N){

  int blockSize1D;
  int gridSize;

  blockSize1D = 32;
  gridSize = (N + blockSize1D - 1)/blockSize1D;

  relu_prime_kernel<<<gridSize, blockSize1D>>> (A_out, A_in, N); 
  
  checkCudaErrors(cudaGetLastError());

}

void custom_log(float* A_out, float* A_in, unsigned int N){
  int blockSize1D;
  int gridSize;

  blockSize1D = 32;
  gridSize = (N + blockSize1D - 1)/blockSize1D;

  log_kernel<<<gridSize, blockSize1D>>> (A_out, A_in, N); 
  
  checkCudaErrors(cudaGetLastError());
}

void custom_exp(float* A_out, float* A_in, unsigned int N){
  int blockSize1D;
  int gridSize;

  blockSize1D = 32;
  gridSize = (N + blockSize1D - 1)/blockSize1D;

  exp_kernel<<<gridSize, blockSize1D>>> (A_out, A_in, N); 
  
  checkCudaErrors(cudaGetLastError());
}

void batched_elementwise_divide(float* A, float* b, unsigned int N, unsigned int batch_size){

  int blockSize1D = N;
  int gridSize = batch_size;

  batched_elementwise_divide_kernel<<<gridSize, blockSize1D>>> (A, b, N, batch_size); 
  
  checkCudaErrors(cudaGetLastError());
}

void inplace_elementwise_multiplication (float* A, float* B, unsigned int N, unsigned int M){
  int blockSize1D;
  int gridSize;

  blockSize1D = 32;
  gridSize = (N*M + blockSize1D - 1)/blockSize1D;

  inplace_elementwise_multiplication_kernel<<<gridSize, blockSize1D>>> (A, B, N, M); 
  
  checkCudaErrors(cudaGetLastError());
}

void elementwise_multiplication(float* A, float* B, float* C, unsigned int N){
  int blockSize1D;
  int gridSize;

  blockSize1D = 32;
  gridSize = (N + blockSize1D - 1)/blockSize1D;

  elementwise_multiplication_kernel<<<gridSize, blockSize1D>>> (A, B, C, N); 
  
  checkCudaErrors(cudaGetLastError());
}

void elementwise_subtraction (float* x, float* y, float* z, unsigned int N){
  int blockSize1D;
  int gridSize;

  blockSize1D = 32;
  gridSize = (N + blockSize1D - 1)/blockSize1D;

  elementwise_subtraction_kernel<<<gridSize, blockSize1D>>> (x, y, z, N); 
  
  checkCudaErrors(cudaGetLastError());
}

void custom_batch_mean_reduce(float* A_mean, float* A_batched, unsigned int N, unsigned int M, unsigned int batch_size){
  int blockSize1D;
  int gridSize;

  blockSize1D = 32; // for dev
  gridSize = (N*M + blockSize1D - 1)/blockSize1D;

  custom_batch_mean_reduce_kernel<<<gridSize, blockSize1D>>> (A_mean, A_batched, N, M, batch_size); 
  
  checkCudaErrors(cudaGetLastError());
}

// void softmax(float* y, float* x, int N, int batch_size){
//     int blockSize1D;
//     int gridSize;

//     blockSize1D = OUTPUT_DIM;
//     gridSize = (N*batch_size + blockSize1D - 1)/blockSize1D;

//     softmax_kernel<<<gridSize, blockSize1D>>> (y, x, N, batch_size); 

//     checkCudaErrors(cudaGetLastError());
// }

