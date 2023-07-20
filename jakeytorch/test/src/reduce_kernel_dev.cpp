#include <jake_kernels.h>
#include <Timer.hpp>

#include <cuda_runtime.h>
//#include <cublas.h>
#include <cublas_v2.h>
#include <helper_cuda.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>

// file readers
#include <iostream>
#include <string>
#include <sstream> 
#include <fstream>

using namespace std;

// forward declare some helpers
void _batched_gemv(float* y, float* A, float* x, int N, int M, int batch_size);
void batched_print_matrix(vector<float>& Z, unsigned int rows, unsigned int cols, unsigned int batch_size);
void print_matrix(vector<float>& Z, unsigned int rows, unsigned int cols);
int read_csv(string file_path, float* data, int N, int M);
float vector_mean_reduction(float* a, int N);
void build_batch(float* batch_X, float* batch_Y, float* X, float* Y, int batch_size, int input_size, int output_size, int N);
void _batched_mean_reduction(float* A_mean, float* A, int N, int M, int batch_size);

int main(){

    // data dimensions:
    int train_N = 160;
    int test_N = 18;
    int output_dim = 3;
    int input_dim = 13;

    // we will store the data in row-major order.
    // features
    vector<float> X_train (
        train_N*input_dim,
        0.0f
    );

    vector<float> X_test (
        test_N*input_dim,
        0.0f
    );

    // labels
    vector<float> Y_train (
        train_N*output_dim,
        0.0f
    );

    vector<float> Y_test (
        test_N*output_dim,
        0.0f
    );
    
    // read our dataset
    read_csv("/home/uahclsc0007/jakee/project/data/wine_x_train.csv", X_train.data(), train_N, input_dim);
    read_csv("/home/uahclsc0007/jakee/project/data/wine_x_test.csv", X_test.data(), test_N, input_dim);
    read_csv("/home/uahclsc0007/jakee/project/data/wine_y_train.csv", Y_train.data(), train_N, output_dim);
    read_csv("/home/uahclsc0007/jakee/project/data/wine_y_test.csv", Y_test.data(), test_N, output_dim);


    // layer dimensions:
    int l1_dim = 64;
    int l2_dim = 32;
    float learning_rate = 0.1;
    int batch_size = 8;
    int max_epochs = 50;

    // build NN:
    //NeuralNetwork NN(input_dim, l1_dim, l2_dim, output_dim, learning_rate);

    // get a test X and Y from the dataset:
    vector<float> batch_X (
        batch_size*input_dim,
        0.0f
    );

    vector<float> batch_Y (
        batch_size*output_dim,
        0.0f
    );

    // put batch of loss here:
    vector<float> batch_loss (
        batch_size*1,
        0.0f
    );

    // make the batch
    build_batch(batch_X.data(), batch_Y.data(), X_train.data(), Y_train.data(), batch_size, input_dim, output_dim, train_N);

    // forward pass test:
    //NN.forward(batch_X.data(), batch_size);

    // we get the prediction out with NN.y_hat.
    //printf("y_hat: \n");
    //print_matrix(NN.y_hat, batch_size, output_dim);

    // backward pass test:
    //NN.backward(batch_loss.data(), batch_X.data(), batch_Y.data(), batch_size);

    //float mean_loss = vector_mean_reduction(batch_loss.data(), batch_size);
    //printf("mean loss: %.4f \n", mean_loss);
    
    // need to get cuBLAS in here and running, testing all the major calculations.
    
    // ----------- TEST AREA -------------
    cublasStatus_t stat;
    cublasHandle_t handle;
    
    
    // for cublas regular matmuls.
    float alpha_default = 1.0f;
    float beta_default = 0.0f;
    
    stat = cublasCreate(&handle);
    
    if (stat != CUBLAS_STATUS_SUCCESS){
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }
    
    
    batch_size = 3;
    output_dim = 3;
    l2_dim = 16;

    float batch_reciprocal = (float)(1/batch_size);
    
    vector<float> dL_dW3_mean (
        output_dim*l2_dim,
        0.0f
    );
    
    vector<float> dL_dW3_mean_ref (
        output_dim*l2_dim,
        0.0f
    );
    
    vector<float> dL_dW3 (
        output_dim*l2_dim*batch_size,
        0.0f
    );
    
    
    for (unsigned int i = 0; i < output_dim*l2_dim*batch_size; ++i){
        dL_dW3[i] = ((float) rand()) / (float) RAND_MAX;
    }
    
    vector<float> ones_vector_batch (
        batch_size*l2_dim,
        1.0f
    );
    
    _batched_mean_reduction(dL_dW3_mean_ref.data(), dL_dW3.data(), output_dim, l2_dim, batch_size);
    
    printf("ones vector: \n");
    batched_print_matrix(ones_vector_batch, batch_size, 1, l2_dim);

    printf("dL_dW3: \n");
    batched_print_matrix(dL_dW3, output_dim, l2_dim, batch_size);

    printf("dL_dW3_mean_ref: \n");
    batched_print_matrix(dL_dW3_mean_ref, output_dim, l2_dim, 1);
    
    // allocate device memory
    float * dL_dW3_device = nullptr;
    float * ones_vector_batch_device = nullptr;
    float * dL_dW3_mean_device = nullptr;
    
    size_t byteSize_dL_dW3 = dL_dW3.size() * sizeof(float);
    size_t byteSize_ones_vector_batch = ones_vector_batch.size() * sizeof(float);
    size_t byteSize_dL_dW3_mean = dL_dW3_mean.size() * sizeof(float);
    
    checkCudaErrors(cudaMalloc(&dL_dW3_device, byteSize_dL_dW3));
    checkCudaErrors(cudaMalloc(&ones_vector_batch_device, byteSize_ones_vector_batch));
    checkCudaErrors(cudaMalloc(&dL_dW3_mean_device, byteSize_dL_dW3_mean));
    
    // copy input
    checkCudaErrors(cudaMemcpy(dL_dW3_device, dL_dW3.data(), byteSize_dL_dW3, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(ones_vector_batch_device, ones_vector_batch.data(), byteSize_ones_vector_batch, cudaMemcpyHostToDevice));
    
    // do y = Ax
    //stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, P, &alpha_default, B_device, M, A_device, P, &beta_default, C_device, M);

    // stat = cublasSgemvStridedBatched(handle,
    //                                 CUBLAS_OP_T,
    //                                 batch_size, output_dim,
    //                                 &alpha_default,
    //                                 dL_dW3_device, batch_size, output_dim*batch_size,
    //                                 ones_vector_batch_device, 1, batch_size,
    //                                 &beta_default,
    //                                 dL_dW3_mean_device, 1, output_dim,
    //                                 l2_dim);

    // stat = cublasSgemvStridedBatched(handle,
    //                                 CUBLAS_OP_T,
    //                                 batch_size, output_dim,
    //                                 &alpha_default,
    //                                 dL_dW3_device, batch_size, output_dim*batch_size,
    //                                 ones_vector_batch_device, 1, batch_size,
    //                                 &beta_default,
    //                                 dL_dW3_mean_device, 1, output_dim,
    //                                 l2_dim);


    // stat = cublasSgemvStridedBatched(handle,
    //                                 CUBLAS_OP_T,
    //                                 l2_dim, output_dim,
    //                                 &alpha_default,
    //                                 dL_dW3_device, output_dim, output_dim*l2_dim,
    //                                 ones_vector_batch_device, 1, l2_dim,
    //                                 &beta_default,
    //                                 dL_dW3_mean_device, 1, output_dim,
    //                                 batch_size);
  
    custom_batch_mean_reduce(dL_dW3_mean_device, dL_dW3_device, output_dim, l2_dim, batch_size);

    // stat = cublasSscal(handle,
    //                     output_dim*l2_dim,
    //                     &batch_reciprocal,
    //                     dL_dW3_mean_device, 1);
  
    // if (stat != CUBLAS_STATUS_SUCCESS){
    //     printf ("error on scal \n");
    // }
    
    // get C down from device
    //stat = cublasGetMatrix(N, M, N*M*sizeof(float), C_device, N, C.data(), N);
    
    // copy result down from device
    checkCudaErrors(cudaMemcpy(dL_dW3_mean.data(), dL_dW3_mean_device, byteSize_dL_dW3_mean, cudaMemcpyDeviceToHost));
    
    printf("dL_dW3_mean: \n");
    batched_print_matrix(dL_dW3_mean, output_dim, l2_dim, 1);
    
    cudaFree(dL_dW3_device);
    cudaFree(ones_vector_batch_device);
    cudaFree(dL_dW3_mean_device);
    
    stat = cublasDestroy(handle);
    
    if (stat != CUBLAS_STATUS_SUCCESS){
        printf ("CUBLAS failure at destroy \n");
        return EXIT_FAILURE;
    }

    return 0;
}


void print_matrix(vector<float>& Z, unsigned int rows, unsigned int cols){
    for (unsigned int row = 0; row < rows; ++row){
        for (unsigned int col = 0; col < cols; ++col){
            printf("%.4f ",Z[row*cols + col]);
        }
        printf("\n");
    }
}

void batched_print_matrix(vector<float>& Z, unsigned int rows, unsigned int cols, unsigned int batch_size){
    for (unsigned int batch = 0; batch < batch_size; ++batch){
        printf("batch %i: \n", batch);
        for (unsigned int row = 0; row < rows; ++row){
            for (unsigned int col = 0; col < cols; ++col){
                printf("%.4f ",Z[batch*rows*cols + row*cols + col]);
            }
            printf("\n");
        }
    }
}

int read_csv(string file_path, float* data, int N, int M){

    ifstream full_data;

    full_data.open(file_path);

    // be sure to handle error
    if (full_data.fail()){
        cerr << "Unable to open file "<< file_path << endl;
        return 1;
    }

    string line;
    string cell;
    unsigned int i = 0;
    while(getline(full_data, line)){   // get entire line (split by newline)
        stringstream ss(line);
        while(getline(ss, cell, ',')){  // split each line by commas
            if (i < (N*M)){
                data[i] = stof(cell);
                ++i;
            }
        }
    }

    full_data.close();

    return 0;
}



void build_batch(float* batch_X, float* batch_Y, float* X, float* Y, int batch_size, int input_size, int output_size, int N){
    // N here is the size of the dataset.
    // samples random indices for a batch
    mt19937 rng(5); // I seeded the rng with 5
    uniform_int_distribution<int> uni(0,N-1);

    for (unsigned int batch = 0; batch<batch_size; ++batch){

        auto batch_row_idx = uni(rng);  // random idx for each data point in batch

        // build batch_X
        for (unsigned int i = 0; i < input_size; ++i){
            batch_X[batch*input_size + i] = X[batch_row_idx*input_size + i];
        }

        // build batch_Y
        for (unsigned int j = 0; j < output_size; ++j){
            batch_Y[batch*output_size + j] = Y[batch_row_idx*output_size + j];
        }
    }
}

// for loss prints
float vector_mean_reduction(float* a, int N){
    // takes vector a (Nx1) and returns its mean value.
    float sum_ = 0.0f;
    for (unsigned int i = 0; i<N; ++i){
        sum_ += a[i];
    }
    return sum_/N;
}

void _batched_gemv(float* y, float* A, float* x, int N, int M, int batch_size){
    // y = Ax, A NxM, all in batches.
    // loop over batch
    for (unsigned int batch = 0; batch<batch_size; ++batch){
        for (unsigned int i = 0; i < N; ++i){
            float sum_ = 0.0f;
            for (unsigned int j = 0; j < M; ++j){
                sum_ += A[batch*N*M + i*M + j]*x[batch*M + j];
            }
            y[batch*N + i] = sum_;
        }
    }
}

void _batched_mean_reduction(float* A_mean, float* A, int N, int M, int batch_size){
    // takes A (batch_size,N,M) and takes mean by batch s.t. A_mean (N, M)
    for (unsigned int i = 0; i<N; ++i){
        for (unsigned int j = 0; j<M; ++j){
            float sum_ = 0.0f;
            for (unsigned int batch = 0; batch<batch_size; ++batch){
                sum_ += A[batch*N*M + i*M + j];
            }
            A_mean[i*M + j] = sum_/batch_size;
        }
    }
}
