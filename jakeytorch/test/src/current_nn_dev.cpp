//#include <reduce.h>
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

        // calcs used for backprop
        vector<float> z1;
        vector<float> a1;
        vector<float> z2;
        vector<float> a2;
        vector<float> z3;
        vector<float> y_hat;

        // for cublas regular matmuls.
        float alpha_default;
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

            if (stat != CUBLAS_STATUS_SUCCESS){
                printf ("CUBLAS initialization failed\n");
                return EXIT_FAILURE;
            }

            alpha_default = 1.0f;
            beta_default = 0.0f;

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

            // -- call internal random_init method --
            _random_init(_W1.data(), l1_dim, input_dim, batch_size);
            _random_init(_b1.data(), l1_dim, 1, batch_size);
            _random_init(_W2.data(), l2_dim, l1_dim, batch_size);
            _random_init(_b2.data(), l2_dim, 1, batch_size);
            _random_init(_W3.data(), output_dim, l2_dim, batch_size);
            _random_init(_b3.data(), output_dim, 1, batch_size);

            // now copy these parameters
            W1 = _W1;
            b1 = _b1;
            W2 = _W2;
            b2 = _b2;
            W3 = _W3;
            b3 = _b3;

            // ----allocate device memory for all the parameters----
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
            // X is the batch input.
            // access the output outside the class by NN.y_hat.

            // -- init the interim calculations --
            vector<float> _z1 (
                batch_size*l1_dim,
                0.0f
            );

            vector<float> _a1 (
                batch_size*l1_dim,
                0.0f
            );

            vector<float> _z2 (
                batch_size*l2_dim,
                0.0f
            );

            vector<float> _a2 (
                batch_size*l2_dim,
                0.0f
            );

            vector<float> _z3 (
                batch_size*output_dim,
                0.0f
            );

            vector<float> _y_hat (
                batch_size*output_dim,
                0.0f
            );

            z1 = _z1;
            a1 = _a1;
            z2 = _z2;
            a2 = _a2;
            z3 = _z3;
            y_hat = _y_hat;

            // --- where I'm at ---
            // copy over X to X_device TODO

            // run the forward pass (note: X is already a pointer)
            //z1 = W1 x + b1
            //_batched_gemv(z1.data(), W1.data(), X, b1.data(), l1_dim, input_dim, batch_size);
            stat = cublasSgemvStridedBatched(handle,
                                            CUBLAS_OP_T,
                                            l1_dim, input_dim,
                                            &alpha_default,
                                            W1_device, M, strideB,
                                            X_device, input_dim, strideA,
                                            &beta_default,
                                            C_device, M, strideC,
                                            batch_size);

            // a1 = relu(z1)
            _relu(a1.data(), z1.data(), l1_dim, batch_size);

            // z2 = W2 a1 + b2
            _batched_gemv(z2.data(), W2.data(), a1.data(), b2.data(), l2_dim, l1_dim, batch_size);

            // a2 = relu(z2)
            _relu(a2.data(), z2.data(), l2_dim, batch_size);

            // z3 = W3 a2 + b3
            _batched_gemv(z3.data(), W3.data(), a2.data(), b3.data(), output_dim, l2_dim, batch_size);

            // y = softmax(z3)
            _softmax(y_hat.data(), z3.data(), output_dim, batch_size);

        }

        void backward(float* loss, float* X, float* Y){
            // backward pass
            // loss is pointer to where to output the batch of loss
            // Y is the batch of the true labels
            // we have everything else we need saved in memory already.

            // edit the loss in-place for monitoring training
            _cross_entropy(loss, Y, y_hat.data(), output_dim, batch_size);

            // ------ backprop ------
            vector<float> dL_dW3 (
                batch_size*output_dim*l2_dim,
                0.0f
            );

            vector<float> dL_dW2 (
                batch_size*l2_dim*l1_dim,
                0.0f
            );

            vector<float> dL_dW1 (
                batch_size*l1_dim*input_dim,
                0.0f
            );

            vector<float> dL_dW3_mean (
                output_dim*l2_dim,
                0.0f
            );

            vector<float> dL_dW2_mean (
                l2_dim*l1_dim,
                0.0f
            );

            vector<float> dL_dW1_mean (
                l1_dim*input_dim,
                0.0f
            );

            //printf("1 \n");
            vector<float> dL_dz3 (
                batch_size*output_dim,
                0.0f
            ); // note: this is also dL/db3.

            vector<float> dL_dz2 (
                batch_size*l2_dim,
                0.0f
            ); // note: this is also dL/db2.

            vector<float> dL_dz1 (
                batch_size*l1_dim,
                0.0f
            ); 

            vector<float> dL_db3_mean (
                output_dim,
                0.0f
            );

            vector<float> dL_db2_mean (
                l2_dim,
                0.0f
            ); 

            vector<float> dL_db1_mean (
                l1_dim,
                0.0f
            );

            //printf("2 \n");

            vector<float> batch_W3_T (
                batch_size*output_dim*l1_dim,
                0.0f
            );

            vector<float> batch_W2_T (
                batch_size*l2_dim*l1_dim,
                0.0f
            );

           //printf("3 \n");

            vector<float> dL_da2 (
                batch_size*l2_dim,
                0.0f
            );

            vector<float> dL_da1 (
                batch_size*l1_dim,
                0.0f
            );

            //printf("4 \n");

            vector<float> relu_prime_z2 (
                batch_size*l2_dim,
                0.0f
            );

            vector<float> relu_prime_z1 (
                batch_size*l1_dim,
                0.0f
            );

            // do y_hat - y (cross entropy with softmax's derivative)
            _batched_elementwise_subtract(dL_dz3.data(), y_hat.data(), Y, output_dim, batch_size);
            //printf("elementwise subtract \n");

            //print_matrix(dL_dz3, batch_size, output_dim);

            // get first derivative for W3. (unreduced)
            // NOTE: we will have to be very careful on our reduce eventually
            // instead of transposing, in batched gemm, I just send it 1 as the internal dimension in the matmul,
            // since the transpose of the vector is the same vector, just reordered.
            _batched_gemm(dL_dW3.data(), dL_dz3.data(), a2.data(), output_dim, 1, l2_dim, batch_size);
            //printf("first gemm \n");

            // --- next layer ---

            // we need W3_T batched to pass to batched gemm
            _build_batch_of_transpose(batch_W3_T.data(), W3.data(), output_dim, l2_dim, batch_size);
            //printf("built batch of transpose \n");

            // do W3^T dL_dz3 = dL_da2
            _batched_gemm(dL_da2.data(), batch_W3_T.data(), dL_dz3.data(), l2_dim, output_dim, 1, batch_size);
            //printf("made it thru batched_gemm \n");

            // need relu_prime
            _relu_prime(relu_prime_z2.data(), z2.data(), l2_dim, batch_size);

            //print_matrix(relu_prime_z2, batch_size, l2_dim);

            // do dL_a2 da2_dz2
            // this is also dL_db2
            _batched_elementwise_multiply(dL_dz2.data(), dL_da2.data(), relu_prime_z2.data(), l2_dim, batch_size);
            //print_matrix(dL_dz2, batch_size, l2_dim);

            //dL_dW2 = dL_dz2 * a1^T
            _batched_gemm(dL_dW2.data(), dL_dz2.data(), a1.data(), l2_dim, 1, l1_dim, batch_size);

            // --- next layer ---

            // we need W2_T batched to pass to batched gemm
            _build_batch_of_transpose(batch_W2_T.data(), W2.data(), l2_dim, l1_dim, batch_size);

            // do W2^T dL_dz2 = dL_da1
            _batched_gemm(dL_da1.data(), batch_W2_T.data(), dL_dz2.data(), l1_dim, l2_dim, 1, batch_size);

            // need relu_prime(z1)
            _relu_prime(relu_prime_z1.data(), z1.data(), l1_dim, batch_size);

            // do dL_a1 da1_dz1 = dL_dz1
            // this is also dL_db1
            _batched_elementwise_multiply(dL_dz1.data(), dL_da1.data(), relu_prime_z1.data(), l1_dim, batch_size);

            //dL_dW1 = dL_dz1 * x^T
            _batched_gemm(dL_dW1.data(), dL_dz1.data(), X, l1_dim, 1, input_dim, batch_size);
            //printf("made it to end of backprop pre-reduction \n");
            //print_matrix(dL_dz1, batch_size, l1_dim);

            // --- now, reduce by batch and update ---
            // reductions (mean)
            _batched_mean_reduction(dL_dW3_mean.data(), dL_dW3.data(), output_dim, l2_dim, batch_size);
            _batched_mean_reduction(dL_dW2_mean.data(), dL_dW2.data(), l2_dim, l1_dim, batch_size);
            _batched_mean_reduction(dL_dW1_mean.data(), dL_dW1.data(), l1_dim, input_dim, batch_size);

            _batched_mean_reduction(dL_db3_mean.data(), dL_dz3.data(), output_dim, 1, batch_size);
            _batched_mean_reduction(dL_db2_mean.data(), dL_dz2.data(), l2_dim, 1, batch_size);
            _batched_mean_reduction(dL_db1_mean.data(), dL_dz1.data(), l1_dim, 1, batch_size);

            //printf("made it to end of backprop reductions \n");

            // do updates:
            _inplace_gradient_descent(W3.data(), dL_dW3_mean.data(), output_dim, l2_dim, learning_rate);
            _inplace_gradient_descent(W2.data(), dL_dW2_mean.data(), l2_dim, l1_dim, learning_rate);
            _inplace_gradient_descent(W1.data(), dL_dW1_mean.data(), l1_dim, input_dim, learning_rate);

            _inplace_gradient_descent(b3.data(), dL_db3_mean.data(), output_dim, 1, learning_rate);
            _inplace_gradient_descent(b2.data(), dL_db2_mean.data(), l2_dim, 1, learning_rate);
            _inplace_gradient_descent(b1.data(), dL_db1_mean.data(), l1_dim, 1, learning_rate);

            //printf("made it to end of param updates \n");

        }

        // --internal methods--
        void _random_init(float* W, int N, int M, int batch_size);
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
void build_batch(float* batch_X, float* batch_Y, float* X, float* Y, int batch_size, int input_size, int output_size, int N);

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

    // ----------- ------ ---------
    // layer dimensions:
    int l1_dim = 64;
    int l2_dim = 32;
    float learning_rate = 0.1;
    const unsigned int batch_size = 8;
    int max_epochs = 50;

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

    // make the batch
    build_batch(batch_X.data(), batch_Y.data(), X_train.data(), Y_train.data(), batch_size, input_dim, output_dim, train_N);

    // forward pass test:
    NN.forward(batch_X.data());

    // we get the prediction out with NN.y_hat.
    //printf("y_hat: \n");
    //print_matrix(NN.y_hat, batch_size, output_dim);
    //printf("y: \n");
    //print_matrix(batch_Y, batch_size, output_dim);

    // backward pass test:
    NN.backward(batch_loss.data(), batch_X.data(), batch_Y.data());

    //printf("loss: \n");
    //print_matrix(batch_loss, batch_size, 1);
    // TODO: reduce this for training monitoring.
    float mean_loss = vector_mean_reduction(batch_loss.data(), batch_size);
    printf("mean loss: %.4f \n", mean_loss);


    //print_matrix(NN.a2, batch_size, l2_dim);

    // train loop test:
    for (unsigned int epoch = 0; epoch<max_epochs; ++epoch){
        printf("epoch: %i \n", epoch);

        // make the batch
        build_batch(batch_X.data(), batch_Y.data(), X_train.data(), Y_train.data(), batch_size, input_dim, output_dim, train_N);

        NN.forward(batch_X.data());

        NN.backward(batch_loss.data(), batch_X.data(), batch_Y.data());

        float mean_loss = vector_mean_reduction(batch_loss.data(), batch_size);
        printf("mean loss: %.4f \n", mean_loss);
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

void NeuralNetwork::_random_init(float* W, int N, int M, int batch_size){
    // this is for batches. copies over the batches.
    // samples random values for weights and biases.
    // generalized for a matrix NxM.
    // init like pytorch does, uniform ()
    mt19937 rng(5); // I seeded the rng with 5
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
