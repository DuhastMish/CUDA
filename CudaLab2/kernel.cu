
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <chrono>
#define BASE_TYPE float
#define N 32*10
#define M 32

using namespace std;
using namespace chrono;

__global__
void fScMult(BASE_TYPE* a, BASE_TYPE* b, BASE_TYPE* scMult) {
    BASE_TYPE sum = 0;
    __shared__ BASE_TYPE ash[M];
    __shared__ BASE_TYPE bsh[M];

    ash[threadIdx.x] = a[blockIdx.x * blockDim.x + threadIdx.x];
    bsh[threadIdx.x] = b[blockIdx.x * blockDim.x + threadIdx.x];
    __syncthreads();
    if (threadIdx.x == 0) {
        sum = 0;
        for (int j = 0; j < blockDim.x; j++) {
            sum += ash[j] * bsh[j];
        }
        atomicAdd(scMult, sum);
        //c[blockIdx.x] = sum;
    }

}

int main()
{
    BASE_TYPE a[N];
    BASE_TYPE b[N];
    BASE_TYPE scMult = 0;;

    size_t size = N * sizeof(BASE_TYPE);

    for (int k = 0; k < N; k++) {
        a[k] = k;
        b[k] = k;
    }

    int t = 0;
    for (int k = 0; k < N; k++) {
        t += a[k] * b[k];
    }
    printf("%d\n", t);

    BASE_TYPE* dev_a;
    BASE_TYPE* dev_b;
    BASE_TYPE* dev_scMult;

    cudaMalloc((void**)&dev_a, size);
    cudaMalloc((void**)&dev_b, size);
    cudaMalloc((void**)&dev_scMult, sizeof(BASE_TYPE));

    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_scMult, &scMult, sizeof(BASE_TYPE), cudaMemcpyHostToDevice);

    dim3 blocksPerGrid = dim3(N / M);
    system_clock::time_point start = system_clock::now();
    fScMult << < blocksPerGrid, M >> > (dev_a, dev_b, dev_scMult);
    system_clock::time_point end = system_clock::now();
    duration<double> sec = end - start;
    cudaMemcpy(&scMult, dev_scMult, sizeof(BASE_TYPE), cudaMemcpyDeviceToHost);

    printf("%f\n", scMult);
    printf("%f .sec\n", sec.count());
    return 0;
}