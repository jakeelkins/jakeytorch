#ifndef __jake_kernels_h__
#define __jake_kernels_h__

#ifdef __cplusplus
extern "C" {
#endif

// ^we could delete this above if we want since we are running c++ code

void relu(float* A_out, float* A_in, unsigned int N);
void relu_prime(float* A_out, float* A_in, unsigned int N);
void custom_log(float* A_out, float* A_in, unsigned int N);
void custom_exp(float* A_out, float* A_in, unsigned int N);
void batched_elementwise_divide(float* A, float* b, unsigned int N, unsigned int batch_size);
void inplace_elementwise_multiplication (float* A, float* B, unsigned int N, unsigned int M);
void elementwise_multiplication(float* A, float* B, float* C, unsigned int N);
void elementwise_subtraction (float* x, float* y, float* z, unsigned int N);
void custom_batch_mean_reduce(float* A_mean, float* A_batched, unsigned int N, unsigned int M, unsigned int batch_size);

#ifdef __cplusplus
}
#endif

#endif
