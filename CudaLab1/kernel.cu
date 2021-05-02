#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#define N 10

__global__ void umn_transp(float* a, float* c) {
	float sum = (float)0;
	for (int i = 0; i < N; i++) {
		sum += a[blockIdx.x * N + i] * a[threadIdx.x * N + i];
	}
	c[blockIdx.x * N + threadIdx.x] = sum;
}

void ort() {
	float* a = (float*)malloc(N * N * sizeof(float));
	float* c = (float*)malloc(N * N * sizeof(float));
	float* a_gpu;
	float* c_gpu;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j <= N; j++)
		{
			a[i * N + j] = (float)0;
		}
	}
	a[2] = 1.0f;
	a[N * 1] = (float)1;
	a[N * 2 + 1] = (float)1;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++)
			printf("%f ", a[i * N + j]);
		printf("\n");
	}
	printf("\n");
	cudaMalloc((void**)&a_gpu, N * N * sizeof(float));
	cudaMalloc((void**)&c_gpu, N * N * sizeof(float));
	cudaMemcpy(a_gpu, a, N * N * sizeof(float), cudaMemcpyHostToDevice);
	umn_transp << <N, N >> > (a_gpu, c_gpu);
	cudaMemcpy(c, c_gpu, N * N * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(a_gpu);
	cudaFree(c_gpu);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++)
			printf("%f ", c[i * N + j]);
		printf("\n");
	}
}

int main() {
	ort();
}