#include <jake_kernels.h>
#include <Timer.hpp>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <helper_cuda.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <chrono>

// file readers
#include <iostream>
#include <string>
#include <sstream> 
#include <fstream>

using namespace std;

class NeuralNetwork {

    public:
        cublasStatus_t stat;
        cublasHandle_t handle;

        int input_dim;
        int l1_dim;
        int l2_dim;
        int output_dim;
        int batch_size;
        float learning_rate;
        float negative_lr;

        // network params
        vector<float> W1;
        vector<float> b1;
        vector<float> W2;
        vector<float> b2;
        vector<float> W3;
        vector<float> b3;

        // device network params
        float * W1_device;
        float * b1_device;

        float * W2_device;
        float * b2_device;

        float * W3_device;
        float * b3_device;

        // used for network calculations (backwards and forwards)
        float* X_device;
        size_t byteSize_X;

        // for interim cross_entropy calcs
        float* loss_calcs;
        float* loss_device;
        size_t byteSize_loss_device;

        // calcs used for backprop
        float * z1;
        size_t byteSize_z1;
        float * a1;
        size_t byteSize_a1;
        float * z2;
        size_t byteSize_z2;
        float * a2;
        size_t byteSize_a2;
        float * z3;
        size_t byteSize_z3;
        float * y_hat;
        size_t byteSize_y_hat;
        float * y_device;
        size_t byteSize_y_device;
        //vector<float> y_hat;

        float* dL_dz3;
        float* dL_dz2;
        float* dL_dz1;

        float* relu_prime_z2;
        float* relu_prime_z1;

        float* dL_da2;
        float* dL_da1;

        float* dL_dW3;
        float* dL_dW2;
        float* dL_dW1;

        float* dL_dW3_mean;
        float* dL_dW2_mean;
        float* dL_dW1_mean;

        float* dL_db3_mean;
        float* dL_db2_mean;
        float* dL_db1_mean;

        // vectors used for fast softmax
        float * ones_for_softmax;
        float * sum_exp_y;

        // for cublas regular matmuls.
        float alpha_default;
        float alpha_neg_one;
        float beta_default;

        // constructor
        NeuralNetwork(unsigned int _input_dim, unsigned int _l1_dim, unsigned int _l2_dim, unsigned int _output_dim, unsigned int _batch_size, float _learning_rate){

            input_dim = _input_dim;
            l1_dim = _l1_dim;
            l2_dim = _l2_dim;
            output_dim = _output_dim;
            batch_size = _batch_size;
            learning_rate = _learning_rate;

            stat = cublasCreate(&handle);

            // if (stat != CUBLAS_STATUS_SUCCESS){
            //     printf ("CUBLAS initialization failed\n");
            // }

            alpha_default = 1.0f;
            alpha_neg_one = -1.0f;
            beta_default = 0.0f;
            negative_lr = -_learning_rate; // for saxpy

            // first vector of weights
            vector<float> _W1 (
                batch_size*l1_dim*input_dim,
                0.0f
            );

            // first layer bias
            vector<float> _b1 (
                batch_size*l1_dim,
                0.0f
            );

            // second vector of weights
            vector<float> _W2 (
                batch_size*l2_dim*l1_dim,
                0.0f
            );

            // second layer bias
            vector<float> _b2 (
                batch_size*l2_dim,
                0.0f
            );

            // third vector of weights
            vector<float> _W3 (
                batch_size*output_dim*l2_dim,
                0.0f
            );

            // third layer bias
            vector<float> _b3 (
                batch_size*output_dim,
                0.0f
            );

            vector<float> ones_for_softmax_host (
                output_dim,
                1.0f
            );

            // -- call internal random_init method --
            mt19937 rng(5); // seed anoter rng just so we dont have to pass it in
            _random_init(rng, _W1.data(), l1_dim, input_dim, batch_size);
            _random_init(rng, _b1.data(), l1_dim, 1, batch_size);
            _random_init(rng, _W2.data(), l2_dim, l1_dim, batch_size);
            _random_init(rng, _b2.data(), l2_dim, 1, batch_size);
            _random_init(rng, _W3.data(), output_dim, l2_dim, batch_size);
            _random_init(rng, _b3.data(), output_dim, 1, batch_size);

            // now copy these parameters
            W1 = _W1;
            b1 = _b1;
            W2 = _W2;
            b2 = _b2;
            W3 = _W3;
            b3 = _b3;

            // ----allocate device memory for all the parameters, batch input----
            // input
            X_device = nullptr;

            byteSize_X = input_dim*batch_size*sizeof(float);

            checkCudaErrors(cudaMalloc(&X_device, byteSize_X));

            // loss calcs
            loss_calcs = nullptr;
            loss_device = nullptr;

            byteSize_loss_device = batch_size*sizeof(float);

            checkCudaErrors(cudaMalloc(&loss_calcs, batch_size*output_dim*sizeof(float)));
            checkCudaErrors(cudaMalloc(&loss_device, byteSize_loss_device));

            // --allocate and copy for softmax--
            ones_for_softmax = nullptr;
            sum_exp_y = nullptr;

            checkCudaErrors(cudaMalloc(&ones_for_softmax, output_dim*sizeof(float)));
            checkCudaErrors(cudaMalloc(&sum_exp_y, batch_size*sizeof(float)));

            checkCudaErrors(cudaMemcpy(ones_for_softmax, ones_for_softmax_host.data(), output_dim*sizeof(float), cudaMemcpyHostToDevice));

            // -- interim calcs --
            z1 = nullptr;
            a1 = nullptr;
            z2 = nullptr;
            a2 = nullptr;
            z3 = nullptr;
            y_hat = nullptr;
            y_device = nullptr;

            byteSize_z1 = batch_size*l1_dim*sizeof(float);
            byteSize_a1 = batch_size*l1_dim*sizeof(float);
            byteSize_z2 = batch_size*l2_dim*sizeof(float);
            byteSize_a2 = batch_size*l2_dim*sizeof(float);
            byteSize_z3 = batch_size*output_dim*sizeof(float);
            byteSize_y_hat = batch_size*output_dim*sizeof(float);
            byteSize_y_device = batch_size*output_dim*sizeof(float);

            checkCudaErrors(cudaMalloc(&z1, byteSize_z1));
            checkCudaErrors(cudaMalloc(&a1, byteSize_a1));
            checkCudaErrors(cudaMalloc(&z2, byteSize_z2));
            checkCudaErrors(cudaMalloc(&a2, byteSize_a2));
            checkCudaErrors(cudaMalloc(&z3, byteSize_z3));
            checkCudaErrors(cudaMalloc(&y_hat, byteSize_y_hat));
            checkCudaErrors(cudaMalloc(&y_device, byteSize_y_device));

            // --- backprop calcs ---
            dL_dz3 = nullptr;
            dL_dz2 = nullptr;
            dL_dz1 = nullptr;

            dL_dW3 = nullptr;
            dL_dW2 = nullptr;
            dL_dW1 = nullptr;

            dL_dW3_mean = nullptr;
            dL_dW2_mean = nullptr;
            dL_dW1_mean = nullptr;

            dL_db3_mean = nullptr;
            dL_db2_mean = nullptr;
            dL_db1_mean = nullptr;

            relu_prime_z2 = nullptr;
            relu_prime_z1 = nullptr;

            dL_da2 = nullptr;
            dL_da1 = nullptr;

            checkCudaErrors(cudaMalloc(&dL_dz3, batch_size*output_dim*sizeof(float)));
            checkCudaErrors(cudaMalloc(&dL_dz2, batch_size*l2_dim*sizeof(float)));
            checkCudaErrors(cudaMalloc(&dL_dz1, batch_size*l1_dim*sizeof(float)));

            checkCudaErrors(cudaMalloc(&dL_dW3, batch_size*output_dim*l2_dim*sizeof(float)));
            checkCudaErrors(cudaMalloc(&dL_dW2, batch_size*l2_dim*l1_dim*sizeof(float)));
            checkCudaErrors(cudaMalloc(&dL_dW1, batch_size*l1_dim*input_dim*sizeof(float)));

            checkCudaErrors(cudaMalloc(&dL_dW3_mean, batch_size*output_dim*l2_dim*sizeof(float)));
            checkCudaErrors(cudaMalloc(&dL_dW2_mean, batch_size*l2_dim*l1_dim*sizeof(float)));
            checkCudaErrors(cudaMalloc(&dL_dW1_mean, batch_size*l1_dim*input_dim*sizeof(float)));

            checkCudaErrors(cudaMalloc(&dL_db3_mean, batch_size*output_dim*sizeof(float)));
            checkCudaErrors(cudaMalloc(&dL_db2_mean, batch_size*l2_dim*sizeof(float)));
            checkCudaErrors(cudaMalloc(&dL_db1_mean, batch_size*l1_dim*sizeof(float)));

            checkCudaErrors(cudaMalloc(&relu_prime_z2, batch_size*l2_dim*sizeof(float)));
            checkCudaErrors(cudaMalloc(&relu_prime_z1, batch_size*l1_dim*sizeof(float)));

            checkCudaErrors(cudaMalloc(&dL_da2, batch_size*l2_dim*sizeof(float)));
            checkCudaErrors(cudaMalloc(&dL_da1, batch_size*l1_dim*sizeof(float)));

            // --first layer--
            W1_device = nullptr;
            b1_device = nullptr;
            
            size_t byteSize_W1 = W1.size() * sizeof(float);
            size_t byteSize_b1 = b1.size() * sizeof(float);
            
            checkCudaErrors(cudaMalloc(&W1_device, byteSize_W1));
            checkCudaErrors(cudaMalloc(&b1_device, byteSize_b1));

            // --second layer--
            W2_device = nullptr;
            b2_device = nullptr;
            
            size_t byteSize_W2 = W2.size() * sizeof(float);
            size_t byteSize_b2 = b2.size() * sizeof(float);
            
            checkCudaErrors(cudaMalloc(&W2_device, byteSize_W2));
            checkCudaErrors(cudaMalloc(&b2_device, byteSize_b2));

            // --third layer--
            W3_device = nullptr;
            b3_device = nullptr;
            
            size_t byteSize_W3 = W3.size() * sizeof(float);
            size_t byteSize_b3 = b3.size() * sizeof(float);
            
            checkCudaErrors(cudaMalloc(&W3_device, byteSize_W3));
            checkCudaErrors(cudaMalloc(&b3_device, byteSize_b3));

            // ---- copy parameters over ----
            checkCudaErrors(cudaMemcpy(W1_device, W1.data(), byteSize_W1, cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(b1_device, b1.data(), byteSize_b1, cudaMemcpyHostToDevice));

            checkCudaErrors(cudaMemcpy(W2_device, W2.data(), byteSize_W2, cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(b2_device, b2.data(), byteSize_b2, cudaMemcpyHostToDevice));

            checkCudaErrors(cudaMemcpy(W3_device, W3.data(), byteSize_W3, cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(b3_device, b3.data(), byteSize_b3, cudaMemcpyHostToDevice));
        
        }

        void forward(float* X){
            // forward pass.
            // X is the batch input from host.

            // copy over X to X_device (note: X is already a pointer)
            checkCudaErrors(cudaMemcpy(X_device, X, byteSize_X, cudaMemcpyHostToDevice));
            
            // run the forward pass
            //z1 = W1 x + b1
            stat = cublasSgemvStridedBatched(handle,
                                            CUBLAS_OP_T,
                                            input_dim, l1_dim,
                                            &alpha_default,
                                            W1_device, input_dim, l1_dim*input_dim,
                                            X_device, 1, input_dim,
                                            &beta_default,
                                            z1, 1, l1_dim,
                                            batch_size);
            
            stat = cublasSaxpy(handle,
                              batch_size*l1_dim,
                              &alpha_default,
                              b1_device, 1,
                              z1, 1);

            // a1 = relu(z1)
            // custom kernel
            relu(a1, z1, batch_size*l1_dim);

            // z2 = W2 a1 + b2
            stat = cublasSgemvStridedBatched(handle,
                                            CUBLAS_OP_T,
                                            l1_dim, l2_dim,
                                            &alpha_default,
                                            W2_device, l1_dim, l2_dim*l1_dim,
                                            a1, 1, l1_dim,
                                            &beta_default,
                                            z2, 1, l2_dim,
                                            batch_size);

            stat = cublasSaxpy(handle,
                              batch_size*l2_dim,
                              &alpha_default,
                              b2_device, 1,
                              z2, 1);

            // a2 = relu(z2)
            // custom kernel
            relu(a2, z2, batch_size*l2_dim);

            // z3 = W3 a2 + b3
            stat = cublasSgemvStridedBatched(handle,
                                            CUBLAS_OP_T,
                                            l2_dim, output_dim,
                                            &alpha_default,
                                            W3_device, l2_dim, output_dim*l2_dim,
                                            a2, 1, l2_dim,
                                            &beta_default,
                                            z3, 1, output_dim,
                                            batch_size);

            stat = cublasSaxpy(handle,
                              batch_size*output_dim,
                              &alpha_default,
                              b3_device, 1,
                              z3, 1);

            // y = softmax(z3)
            //_softmax(y_hat.data(), z3.data(), output_dim, batch_size);
            // first, use custom kernel to raise to exp:
            custom_exp(y_hat, z3, output_dim*batch_size);
            // then use simple gemv with ones vector to get sum_exp_y
            // y_hat is batch of vectors raised to exp now.
            stat = cublasSgemv(handle,
                                CUBLAS_OP_T,
                                output_dim, batch_size,
                                &alpha_default,
                                y_hat, output_dim,
                                ones_for_softmax, 1,
                                &beta_default,
                                sum_exp_y, 1);

            // now one more custom kernel to divide by this new sum_exp_y and we are done
            // this is done inplace on y_hat.
            batched_elementwise_divide(y_hat, sum_exp_y, output_dim, batch_size);

        }

        void backward(float* loss, float* X, float* Y){
            // backward pass
            // loss is pointer to where to output the batch of loss
            // Y is the batch of the true labels
            // we have everything else we need saved in memory already.

            // copy Y over:
            checkCudaErrors(cudaMemcpy(y_device, Y, byteSize_y_device, cudaMemcpyHostToDevice));

            // edit the loss in-place for monitoring training
            // custom kernels here for cross entropy
            custom_log(loss_calcs, y_hat, batch_size*output_dim);
            // loss_calcs is now log(y_hat)
            // y*log(y_hat):
            inplace_elementwise_multiplication (loss_calcs, y_device, batch_size, output_dim);
            // loss = -sum(y*log(y_hat))
            stat = cublasSgemv(handle,
                                CUBLAS_OP_T,
                                output_dim, batch_size,
                                &alpha_neg_one,
                                loss_calcs, output_dim,
                                ones_for_softmax, 1,
                                &beta_default,
                                loss_device, 1);

            // copy loss down from device to host loss pointer
            checkCudaErrors(cudaMemcpy(loss, loss_device, byteSize_loss_device, cudaMemcpyDeviceToHost));

            // should be good on loss.

            // ------ backprop ------

            // do y_hat - y (cross entropy with softmax's derivative)
            // custom kernel
            elementwise_subtraction(dL_dz3, y_hat, y_device, batch_size*output_dim);

            // get first derivative for W3. (unreduced)
            // dL_dW3 = dL_dz3 a2^T
            stat = cublasSgemmStridedBatched(handle,
                                            CUBLAS_OP_N, CUBLAS_OP_N,
                                            l2_dim, output_dim, 1,
                                            &alpha_default,
                                            a2, l2_dim, l2_dim,
                                            dL_dz3, 1, output_dim,
                                            &beta_default,
                                            dL_dW3, l2_dim, output_dim*l2_dim,
                                            batch_size);

            // --- next layer ---
            // do W3^T dL_dz3 = dL_da2
            stat = cublasSgemmStridedBatched(handle,
                                            CUBLAS_OP_N, CUBLAS_OP_T,
                                            1, l2_dim, output_dim,
                                            &alpha_default,
                                            dL_dz3, 1, output_dim,
                                            W3_device, l2_dim, output_dim*l2_dim,
                                            &beta_default,
                                            dL_da2, 1, l2_dim,
                                            batch_size);

            // need relu_prime
            // custom relu_prime kernel
            relu_prime(relu_prime_z2, z2, l2_dim*batch_size);

            // do dL_a2 da2_dz2
            // this is also dL_db2
            // custom elementwise multiply kernel
            elementwise_multiplication(dL_dz2, dL_da2, relu_prime_z2, l2_dim*batch_size);

            //dL_dW2 = dL_dz2 * a1^T
            stat = cublasSgemmStridedBatched(handle,
                                            CUBLAS_OP_N, CUBLAS_OP_N,
                                            l1_dim, l2_dim, 1,
                                            &alpha_default,
                                            a1, l1_dim, l1_dim,
                                            dL_dz2, 1, l2_dim,
                                            &beta_default,
                                            dL_dW2, l1_dim, l2_dim*l1_dim,
                                            batch_size);
            // --- next layer ---

            // do W2^T dL_dz2 = dL_da1
            stat = cublasSgemmStridedBatched(handle,
                                            CUBLAS_OP_N, CUBLAS_OP_T,
                                            1, l1_dim, l2_dim,
                                            &alpha_default,
                                            dL_dz2, 1, l2_dim,
                                            W2_device, l1_dim, l2_dim*l1_dim,
                                            &beta_default,
                                            dL_da1, 1, l1_dim,
                                            batch_size);

            // need relu_prime(z1)
            // custom relu prime
            relu_prime(relu_prime_z1, z1, l1_dim*batch_size);

            // do dL_a1 da1_dz1 = dL_dz1
            // this is also dL_db1
            // custom elementwise multiply kernel
            elementwise_multiplication(dL_dz1, dL_da1, relu_prime_z1, l1_dim*batch_size);

            //dL_dW1 = dL_dz1 * x^T
            stat = cublasSgemmStridedBatched(handle,
                                            CUBLAS_OP_N, CUBLAS_OP_N,
                                            input_dim, l1_dim, 1,
                                            &alpha_default,
                                            X_device, input_dim, input_dim,
                                            dL_dz1, 1, l1_dim,
                                            &beta_default,
                                            dL_dW1, input_dim, l1_dim*input_dim,
                                            batch_size);

            // --- now, reduce by batch and update ---
            // reductions (mean)
            custom_batch_mean_reduce(dL_dW3_mean, dL_dW3, output_dim, l2_dim, batch_size);
            custom_batch_mean_reduce(dL_dW2_mean, dL_dW2, l2_dim, l1_dim, batch_size);
            custom_batch_mean_reduce(dL_dW1_mean, dL_dW1, l1_dim, input_dim, batch_size);

            custom_batch_mean_reduce(dL_db3_mean, dL_dz3, output_dim, 1, batch_size);
            custom_batch_mean_reduce(dL_db2_mean, dL_dz2, l2_dim, 1, batch_size);
            custom_batch_mean_reduce(dL_db1_mean, dL_dz1, l1_dim, 1, batch_size);

            //printf("made it to end of backprop reductions \n");

            // do updates:
            // use saxpy with -LR as alpha to get inplace SGD.
            stat = cublasSaxpy(handle,
                                batch_size*output_dim*l2_dim,
                                &negative_lr,
                                dL_dW3_mean, 1,
                                W3_device, 1);

            stat = cublasSaxpy(handle,
                                batch_size*l2_dim*l1_dim,
                                &negative_lr,
                                dL_dW2_mean, 1,
                                W2_device, 1);

            stat = cublasSaxpy(handle,
                                batch_size*l1_dim*input_dim,
                                &negative_lr,
                                dL_dW1_mean, 1,
                                W1_device, 1);

            stat = cublasSaxpy(handle,
                                batch_size*output_dim,
                                &negative_lr,
                                dL_db3_mean, 1,
                                b3_device, 1);

            stat = cublasSaxpy(handle,
                                batch_size*l2_dim,
                                &negative_lr,
                                dL_dW2_mean, 1,
                                b2_device, 1);

            stat = cublasSaxpy(handle,
                                batch_size*l1_dim,
                                &negative_lr,
                                dL_db1_mean, 1,
                                b1_device, 1);

            //printf("made it to end of param updates \n");

        }

        // --internal methods--
        void _random_init(mt19937 &rng, float* W, int N, int M, int batch_size);
        void _inplace_gradient_descent(float* A, float* dA, int N, int M, float learning_rate);
        void _batched_elementwise_subtract(float* y, float* a, float* b, int N, int batch_size);
        void _batched_elementwise_multiply(float* y, float* a, float* b, int N, int batch_size);
        void _batched_gemv(float* y, float* A, float* x, float* b, int N, int M, int batch_size);
        void _batched_gemm(float* C, float* A, float* B, int N, int P, int M, int batch_size);
        void _batched_mean_reduction(float* A_mean, float* A, int N, int M, int batch_size);
        void _build_batch_of_transpose(float* A_T_batch, float* A, int N, int M, int batch_size);
        void _batched_transpose(float* A_T, float* A, int N, int M, int batch_size);
        void _transpose(float* A_T, float* A, int N, int M);
        void _relu(float* y, float* x, int N, int batch_size);
        void _relu_prime(float* y, float* x, int N, int batch_size);
        void _softmax(float* y, float* x, int N, int batch_size);
        void _cross_entropy(float* loss, float* y, float* y_hat, int N, int batch_size);
        void print_matrix(vector<float>& Z, unsigned int rows, unsigned int cols);
};

void print_matrix(vector<float>& Z, unsigned int rows, unsigned int cols);
int read_csv(string file_path, float* data, int N, int M);
float vector_mean_reduction(float* a, int N);
void build_batch(mt19937 &rng, float* batch_X, float* batch_Y, float* X, float* Y, int batch_size, int input_size, int output_size, int N);

int main(){

    // data dimensions: wine
    // int train_N = 160;
    // int test_N = 18;
    // int output_dim = 3;
    // int input_dim = 13;

    // data dimensions: cifar10
    int train_N = 45000;
    int test_N = 5000;
    int output_dim = 10;
    int input_dim = 3072;

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
    printf("reading data ...\n");

    // read_csv("/home/uahclsc0007/jakee/project/data/wine_x_train.csv", X_train.data(), train_N, input_dim);
    // read_csv("/home/uahclsc0007/jakee/project/data/wine_x_test.csv", X_test.data(), test_N, input_dim);
    // read_csv("/home/uahclsc0007/jakee/project/data/wine_y_train.csv", Y_train.data(), train_N, output_dim);
    // read_csv("/home/uahclsc0007/jakee/project/data/wine_y_test.csv", Y_test.data(), test_N, output_dim);

    read_csv("/home/uahclsc0007/jakee/project/data/cifar_x_train.csv", X_train.data(), train_N, input_dim);
    read_csv("/home/uahclsc0007/jakee/project/data/cifar_x_test.csv", X_test.data(), test_N, input_dim);
    read_csv("/home/uahclsc0007/jakee/project/data/cifar_y_train.csv", Y_train.data(), train_N, output_dim);
    read_csv("/home/uahclsc0007/jakee/project/data/cifar_y_test.csv", Y_test.data(), test_N, output_dim);

    // ----------- ------ ---------
    // layer dimensions:
    int l1_dim = 400;
    int l2_dim = 300;
    float learning_rate = 0.0001;
    const unsigned int batch_size = 32;
    int max_epochs = 100000;
    mt19937 rng(5); // seed the rng with 5 here

    // for timing
    chrono::steady_clock::time_point ti;
    chrono::steady_clock::time_point tf;
    //chrono::steady_clock::time_point tb_i;
    //chrono::steady_clock::time_point tb_f;

    // build NN:
    NeuralNetwork NN(input_dim, l1_dim, l2_dim, output_dim, batch_size, learning_rate);

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

    // // for dev ---------
    vector<float> y_hat_host (
        batch_size*output_dim,
        0.0f
    );
    // --------------------
    float running_loss = 0.0f;

    ti = chrono::steady_clock::now();

    //train loop test:
    for (unsigned int epoch = 0; epoch<max_epochs; ++epoch){
        //tb_i = chrono::steady_clock::now();

        // make the batch
        build_batch(rng, batch_X.data(), batch_Y.data(), X_train.data(), Y_train.data(), batch_size, input_dim, output_dim, train_N);

        NN.forward(batch_X.data());

        NN.backward(batch_loss.data(), batch_X.data(), batch_Y.data());

        // get result (dev)
        checkCudaErrors(cudaMemcpy(y_hat_host.data(), NN.y_hat, NN.byteSize_y_hat, cudaMemcpyDeviceToHost));

        float mean_loss = vector_mean_reduction(batch_loss.data(), batch_size);

        running_loss += mean_loss;
        if (epoch%1000 == 999){
            printf("--epoch: %i \t mean loss: %.4f-- \n", epoch+1, running_loss/1000);
            running_loss = 0.0f;
        }
        
    }

    tf = chrono::steady_clock::now();
    float time_elapsed_ms = chrono::duration_cast<chrono::milliseconds>(tf - ti).count();
    printf("elapsed time: %.4f s \n", time_elapsed_ms/1000);
    printf("avg throughput: %.4f updates/s \n", max_epochs/(time_elapsed_ms/1000));

    printf("y_hat: \n");
    print_matrix(y_hat_host, batch_size, output_dim);

    printf("y: \n");
    print_matrix(batch_Y, batch_size, output_dim);

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



void build_batch(mt19937 &rng, float* batch_X, float* batch_Y, float* X, float* Y, int batch_size, int input_size, int output_size, int N){
    // N here is the size of the dataset.
    // samples random indices for a batch
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

void NeuralNetwork::_random_init(mt19937 &rng, float* W, int N, int M, int batch_size){
    // this is for batches. copies over the batches.
    // samples random values for weights and biases.
    // generalized for a matrix NxM.
    // init like pytorch does, uniform ()
    float stddev = 1/sqrt((float)N);
    uniform_real_distribution<float> uni(-stddev, stddev);

    for (unsigned int i = 0; i<N; ++i){
        for (unsigned int j = 0; j<M; ++j){
            auto rand_val = uni(rng); // use same val in each batch spot.
            for (unsigned int batch = 0;batch < batch_size; ++batch){
                W[batch*N*M + i*M + j] = rand_val;
            }
        }
    }
}

// void NeuralNetwork::_gemv(float* y, float* A, float* x, float* b, int N, int M){
//     // y = Ax + b, where A is NxM.
//     for (unsigned int row = 0; row < N; ++row){
//         float sum_ = 0.0f;
//         for (unsigned int col = 0; col < M; ++col){
//             sum_ += A[row*M + col]*x[col];
//         }
//         y[row] = sum_ + b[row];
//     }
// }

void NeuralNetwork::_inplace_gradient_descent(float* A, float* dA, int N, int M, float learning_rate){
    // A and dA are (N x M)
    // A = A - learning_rate*dA
    for (unsigned int i = 0; i<N; ++i){
        for (unsigned int j = 0; j<M; ++j){
            A[i*M + j] = A[i*M + j] - learning_rate*dA[i*M + j];
        }
    }
}

void NeuralNetwork::_batched_gemv(float* y, float* A, float* x, float* b, int N, int M, int batch_size){
    // y = Ax + b, where A is NxM.
    // this is in batches, so the batch dimension is unchanged (first dim)

    // loop over batch
    for (unsigned int batch = 0; batch<batch_size; ++batch){
        for (unsigned int row = 0; row < N; ++row){
            float sum_ = 0.0f;
            for (unsigned int col = 0; col < M; ++col){
                sum_ += A[row*M + col]*x[batch*M + col];
            }
            y[batch*N + row] = sum_ + b[row];
        }
    }
}

void NeuralNetwork::_batched_gemm(float* C, float* A, float* B, int N, int P, int M, int batch_size){
    // C = AB, where C is NxM, A NxP, B PxM.
    // this is in batches, so the batch dimension is unchanged (first dim)

    // loop over batch
    for (unsigned int batch = 0; batch<batch_size; ++batch){
        for (unsigned int i = 0; i < N; ++i){
            for (unsigned int j = 0; j < M; ++j){
                float sum_ = 0.0f;
                for (unsigned int k = 0; k < P; ++k){
                    sum_ += A[batch*N*P + i*P + k]*B[batch*P*M + k*M + j];
                }
                C[batch*N*M + i*M + j] = sum_;
            }
        }
    }
}

void NeuralNetwork::_batched_elementwise_subtract(float* y, float* a, float* b, int N, int batch_size){
    // y = a - b, where a and b Nx1, matrices stored as (batch_size, N)
    // this is in batches, so the batch dimension is unchanged (first dim)

    // loop over batch
    for (unsigned int batch = 0; batch<batch_size; ++batch){
        for (unsigned int row = 0; row < N; ++row){
            y[batch*N + row] = a[batch*N + row] - b[batch*N + row];
        }
    }
}

void NeuralNetwork::_batched_elementwise_multiply(float* y, float* a, float* b, int N, int batch_size){
    // y = a - b, where a and b Nx1, matrices stored as (batch_size, N)
    // this is in batches, so the batch dimension is unchanged (first dim)

    // loop over batch
    for (unsigned int batch = 0; batch<batch_size; ++batch){
        for (unsigned int row = 0; row < N; ++row){
            y[batch*N + row] = a[batch*N + row]*b[batch*N + row];
        }
    }
}

void NeuralNetwork::_batched_mean_reduction(float* A_mean, float* A, int N, int M, int batch_size){
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

void NeuralNetwork::_build_batch_of_transpose(float* A_T_batch, float* A, int N, int M, int batch_size){
    // need this fxn for backprop
    // takes a matrix A (NxM) and makes a batch of A_T (batch_size, M, N)
    for (unsigned int batch = 0; batch<batch_size; ++batch){
        for (unsigned int i = 0; i<N; ++i){
            for (unsigned int j = 0; j<M; ++j){
                A_T_batch[batch*M*N + j*N + i] = A[i*M + j];
            }
        }
    }
}

void NeuralNetwork::_batched_transpose(float* A_T, float* A, int N, int M, int batch_size){
    // batched matrix transpose: A is NxM, A_T is MxN. first dim (batch) unchanged.
    for (unsigned int batch = 0; batch<batch_size; ++batch){
        for (unsigned int i = 0; i<N; ++i){
            for (unsigned int j = 0; j<M; ++j){
                A_T[batch*M*N + j*N + i] = A[batch*N*M + i*M + j];
            }
        }
    }
}

void NeuralNetwork::_transpose(float* A_T, float* A, int N, int M){
    // single matrix transpose: A is NxM, A_T is MxN.
    for (unsigned int i = 0; i<N; ++i){
        for (unsigned int j = 0; j<M; ++j){
            A_T[j*N + i] = A[i*M + j];
        }
    }
}

void NeuralNetwork::_relu(float* y, float* x, int N, int batch_size){
    // elementwise relu activation
    // input x, output y, N is length of x.
    for (unsigned int batch = 0; batch<batch_size; ++batch){
        for (unsigned int i = 0; i<N; ++i){
            float in_;
            float out_;
            in_ = x[batch*N + i];

            if (in_ > 0.0f){
                out_ = in_;
            } else {
                out_ = 0.0f;
            }

            y[batch*N + i] = out_;
        }
    }
}

void NeuralNetwork::_relu_prime(float* y, float* x, int N, int batch_size){
    // elementwise relu activation derivative
    // input x, output y, N is length of x.
    for (unsigned int batch = 0; batch<batch_size; ++batch){
        for (unsigned int i = 0; i<N; ++i){
            float in_;
            float out_;
            in_ = x[batch*N + i];

            if (in_ > 0.0f){
                out_ = 1.0f;
            } else {
                out_ = 0.0f;
            }

            y[batch*N + i] = out_;
        }
    }
}

void NeuralNetwork::_softmax(float* y, float* x, int N, int batch_size){
    // TODO: numerically stable softmax.
    // this will need parallelized? (the sum reduction)
    // softmax for output
    // input x, output y, N is length of x

    for (unsigned int batch = 0; batch<batch_size; ++batch){
        // first: convert y to exp(x)
        for (unsigned int i = 0; i<N; ++i){
            y[batch*N + i] = exp(x[batch*N + i]);
        }

        // sum each row of exp(x) (now y)
        float sum_exp = 0.0f;
        for (unsigned int i = 0; i<N; ++i){
            sum_exp += y[batch*N + i];
        }

        // now go back and divide each element of y with this value.
        for (unsigned int i = 0; i<N; ++i){
            y[batch*N + i] = y[batch*N + i]/sum_exp;
        }
    }
}

void NeuralNetwork::_cross_entropy(float* loss, float* y, float* y_hat, int N, int batch_size){
    // cross entropy loss for batch
    // put output in loss vector.

    for (unsigned int batch = 0; batch<batch_size; ++batch){
        float sum_of_logs = 0.0f;
        for (unsigned int i = 0; i<N; ++i){
            sum_of_logs -= y[batch*N + i]*log(y_hat[batch*N + i]);
        }
        loss[batch] = sum_of_logs;
    }

}

void NeuralNetwork::print_matrix(vector<float>& Z, unsigned int rows, unsigned int cols){
    for (unsigned int row = 0; row < rows; ++row){
        for (unsigned int col = 0; col < cols; ++col){
            printf("%.4f ",Z[row*cols + col]);
        }
        printf("\n");
    }
}
