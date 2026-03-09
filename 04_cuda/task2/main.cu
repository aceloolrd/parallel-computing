#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#define BLOCK_SIZE 32 

void getInfoCUDADevice(cudaDeviceProp& prop, int id) {
	printf("CUDA device %i Название   - %s\n", id, prop.name);
	printf("CUDA device %i Размер warp'а в потоках выполнения  - %i\n", id, prop.warpSize);
	printf("CUDA device %i Максимальное количество потоков выполнения в блоке  - %i\n", id, prop.maxThreadsPerBlock);
	printf("CUDA device %i Количество мультипроцессоров на устройстве  - %i\n", id, prop.multiProcessorCount);
	printf("CUDA device %i Максимальный размер каждого измерения блока потоков выполнения  - %i %i %i\n", id, prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
	printf("CUDA device %i Максимальный размер каждого измерения сетки блоков потоков выполнения  - %i %i %i\n", id, prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
}

__global__ void matrixMult(const __int16* Am, const __int16* Bm, __int16* result, int size) {
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	// вычисление индексов, используемых для доступа к элементам матриц и мас-сиву результата в ядре умножения матриц
	int ia = size * (gridDim.y * by + ty);
	int ib = gridDim.x * bx + tx;
	int ic = ia + ib;

	__int16 sum = 0;

	for (int k = 0; k < size; k++) {
		sum += Am[ia + k] * Bm[k * size + ib];
	}
	result[ic] = sum;
}


void compareMatrix(const __int16* f, const __int16* s, int size) {
	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < size; ++j) {
			if (f[i * size + j] != s[i * size + j]) {
				printf("Matrixes not equal!\n");
				return;
			}
		}
	}
	printf("Matrices are equal!\n");
}

int main()
{
	setlocale(LC_ALL, "Russian");

	int count;
	cudaDeviceProp prop;
	cudaGetDeviceCount(&count);
	cudaGetDeviceProperties(&prop, count - 1);
	getInfoCUDADevice(prop, count - 1);

	int size = 1024;

	size_t byte_size = size * size * sizeof(__int16);
	__int16* Am = (__int16*)malloc(byte_size);
	__int16* Bm = (__int16*)malloc(byte_size);
	__int16* GPU_C = (__int16*)malloc(byte_size);
	__int16* CPU_C = (__int16*)malloc(byte_size);

	for (int i = 0; i < size * size; ++i) {
		Am[i] = rand() % 6;
		Bm[i] = rand() % 6;
		CPU_C[i] = 0;
	}

	printf("\nScalar: \n");
	auto start = std::chrono::system_clock::now();
	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < size; ++j) {
			for (int k = 0; k < size; ++k) {
				CPU_C[i * size + j] += Am[i * size + k] * Bm[k * size + j];
			}
		}
	}
	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;

	printf("Time: %f seconds\n", diff);


	printf("GPU: \n");

	__int16* d_A = NULL;
	cudaMalloc((void**)&d_A, byte_size);
	cudaMemcpy(d_A, Am, byte_size, cudaMemcpyHostToDevice);

	__int16* d_B = NULL;
	cudaMalloc((void**)&d_B, byte_size);
	cudaMemcpy(d_B, Bm, byte_size, cudaMemcpyHostToDevice);

	__int16* d_C = NULL;
	cudaMalloc((void**)&d_C, byte_size);


	cudaEvent_t startEvent, stopEvent;
	cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);

	cudaEventRecord(startEvent, 0);
	const dim3 block(32, 32);
	const dim3 grid((size) / block.x, (size) / block.y);
	matrixMult << < grid, block >> > (d_A, d_B, d_C, size);

	// ждем завершения
	cudaDeviceSynchronize();
	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);

	cudaMemcpy(GPU_C, d_C, byte_size, cudaMemcpyDeviceToHost);

	printf("Time: %f seconds\n", milliseconds / 1000);
	compareMatrix(GPU_C, CPU_C, size);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	free(Am);
	free(Bm);
	free(GPU_C);
	free(CPU_C);
	cudaEventDestroy(startEvent);
	cudaEventDestroy(stopEvent);

	return 0;
}
