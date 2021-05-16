#include <iostream>
#include <stdio.h>
#include <chrono>
#include <cuda_runtime.h>
#include <vector>
#include <math.h>
#include <device_launch_parameters.h>

using namespace std;
using namespace chrono;

template <class T>
class MatrixMul {
public:
    explicit MatrixMul() : start_(0), end_(0) {}

    explicit MatrixMul(size_t size) {
        allocate(size);
    }

    ~MatrixMul() {
        free();
    }

    void resize(size_t size) {
        free();
        allocate(size);
    }

    size_t getSize() const {
        return end_ - start_;
    }

    const T* getData() const {
        return start_;
    }

    T* getData() {
        return start_;
    }

    void set(const T* src, size_t size) {
        size_t min = std::min(size, getSize());
        cudaError_t result = cudaMemcpy(start_, src, min * sizeof(T), cudaMemcpyHostToDevice);
    }

    void get(T* dest, size_t size) {
        size_t min = std::min(size, getSize());
        cudaError_t result = cudaMemcpy(dest, start_, min * sizeof(T), cudaMemcpyDeviceToHost);
    }

private:
    void allocate(size_t size) {
        cudaError_t result = cudaMalloc((void**)&start_, size * sizeof(T));
        if (result != cudaSuccess)
        {
            start_ = end_ = 0;
            throw std::runtime_error("failed to allocate device memory");
        }
        end_ = start_ + size;
    }

    void free() {
        if (start_ != 0)
        {
            cudaFree(start_);
            start_ = end_ = 0;
        }
    }

    T* start_;
    T* end_;

};

__global__ void matrixMultiplicationKernel(float* A, float* B, float* C, int N) {

    int ROW = blockIdx.y * blockDim.y + threadIdx.y;
    int COL = blockIdx.x * blockDim.x + threadIdx.x;

    float tmpSum = 0;

    if (ROW < N && COL < N) {
        for (int i = 0; i < N; i++) {
            tmpSum += A[ROW * N + i] * B[i * N + COL];
        }
    }
    C[ROW * N + COL] = tmpSum;
}


void matrixMultiplication(float* A, float* B, float* C, int N) {
    dim3 threadsPerBlock(N, N);
    dim3 blocksPerGrid(1, 1);
    if (N * N > 512) {
        threadsPerBlock.x = 512;
        threadsPerBlock.y = 512;
        blocksPerGrid.x = ceil(double(N) / double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(N) / double(threadsPerBlock.y));
    }
    
    matrixMultiplicationKernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
}


int main()
{
    int N = 16;
    int SIZE = N * N;

    vector<float> h_A(SIZE);
    vector<float> h_B(SIZE);
    vector<float> h_C(SIZE);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            h_A[i * N + j] = sin(i);
            h_B[i * N + j] = cos(j);
        }
    }

    MatrixMul<float> d_A(SIZE);
    MatrixMul<float> d_B(SIZE);
    MatrixMul<float> d_C(SIZE);

    d_A.set(&h_A[0], SIZE);
    d_B.set(&h_B[0], SIZE);
    system_clock::time_point start = system_clock::now();
    matrixMultiplication(d_A.getData(), d_B.getData(), d_C.getData(), N);
    system_clock::time_point end = system_clock::now();
    duration<double> sec = end - start;
    cudaDeviceSynchronize();

    d_C.get(&h_C[0], SIZE);
    cudaDeviceSynchronize();
    printf("%f\n", d_C);
    printf("%f .sec\n", sec.count());

    return 0;
}