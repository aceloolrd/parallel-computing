#include <iostream>
#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <algorithm>  

#define BLOCK_SIZE 32  // Размер блока для запуска ядра CUDA

__global__ void convolutionKernel(const uchar* input, uchar* output, int width, int height, const float* kernel, int kernelSize) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int halfKernelSize = kernelSize / 2;
        uchar result = 0;

        for (int i = -halfKernelSize; i <= halfKernelSize; ++i) {
            for (int j = -halfKernelSize; j <= halfKernelSize; ++j) {
                int currentCol = col + j;
                int currentRow = row + i;

                currentCol = fmaxf(0, fminf(width - 1, currentCol));
                currentRow = fmaxf(0, fminf(height - 1, currentRow));

                uchar pixelValue = input[currentRow * width + currentCol];
                result += pixelValue * kernel[(i + halfKernelSize) * kernelSize + (j + halfKernelSize)];
            }
        }

        output[row * width + col] = result;
    }
}


cv::Mat performConvolution(const cv::Mat& input, const float* kernel, int kernelSize) {
    int width = input.cols;
    int height = input.rows;

    cv::Mat output(height, width, CV_8UC1);

    // Выделение памяти на устройстве для входного, выходного изображения и ядра свёртки
    uchar* d_input, * d_output;
    float* d_kernel;
    cudaMalloc(&d_input, width * height * sizeof(uchar));
    cudaMalloc(&d_output, width * height * sizeof(uchar));
    cudaMalloc(&d_kernel, kernelSize * kernelSize * sizeof(float));

    // Копирование входного изображения и ядра свёртки на устройство
    cudaMemcpy(d_input, input.data, width * height * sizeof(uchar), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);

    // Вычисление размеров сетки и блока для запуска ядра CUDA
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Запуск ядра CUDA
    convolutionKernel << <gridSize, blockSize >> > (d_input, d_output, width, height, d_kernel, kernelSize);

    // Копирование результата обратно на хост
    cudaMemcpy(output.data, d_output, width * height * sizeof(uchar), cudaMemcpyDeviceToHost);

    // Освобождение памяти на устройстве
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);

    return output;
}

/*
__global__ void convolutionKernel(const float* input, float* output, int width, int height, const float* kernel, int kernelSize) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int halfKernelSize = kernelSize / 2;
        float result = 0.0f;

        for (int i = -halfKernelSize; i <= halfKernelSize; ++i) {
            for (int j = -halfKernelSize; j <= halfKernelSize; ++j) {
                int currentCol = col + j;
                int currentRow = row + i;

                // Проверка и коррекция на выход за границы
                //currentCol = (currentCol < 0) ? 0 : ((currentCol >= width) ? width - 1 : currentCol);
                //currentRow = (currentRow < 0) ? 0 : ((currentRow >= height) ? height - 1 : currentRow);

                currentCol = fmaxf(0, fminf(width - 1, currentCol));
                currentRow = fmaxf(0, fminf(height - 1, currentRow));

                float pixelValue = input[currentRow * width + currentCol];
                result += pixelValue * kernel[(i + halfKernelSize) * kernelSize + (j + halfKernelSize)];
            }
        }

        output[row * width + col] = result;
    }
}

cv::Mat performConvolution(const cv::Mat& input, const float* kernel, int kernelSize) {
    int width = input.cols;
    int height = input.rows;

    cv::Mat output(height, width, CV_32FC1);  // Используйте CV_32FC1 для хранения значений float

    // Выделение памяти на устройстве для входного и выходного изображения
    float* d_input, * d_output;
    cudaMalloc(&d_input, width * height * sizeof(float));
    cudaMalloc(&d_output, width * height * sizeof(float));

    // Копирование входного изображения на устройство и преобразование к типу float
    input.convertTo(output, CV_32FC1);
    cudaMemcpy(d_input, output.data, width * height * sizeof(float), cudaMemcpyHostToDevice);

    // Вычисление размеров сетки и блока для запуска ядра CUDA
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Запуск ядра CUDA
    convolutionKernel << <gridSize, blockSize >> > (d_input, d_output, width, height, kernel, kernelSize);

    // Копирование результата обратно на устройство
    cudaMemcpy(output.data, d_output, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    // Освобождение памяти на устройстве
    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}
*/

int main() {
    // Загрузка изображения
    cv::Mat inputImage = cv::imread("test.png", cv::IMREAD_GRAYSCALE);

    // Определение ядра свертки 3x3 
    float kernel[3][3] = { {1, 1, 1},
                           {1, -8, 1},
                           {1, 1, 1} };
    /*
    // Normalize the kernel
    float sum = 0.0f;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            sum += kernel[i][j];
        }
    }

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            kernel[i][j] /= sum;
        }
    }
    */

    // Преобразование ядра в одномерный массив
    int kernelSize = 3;
    float flatKernel[9];
    for (int i = 0; i < kernelSize; ++i) {
        for (int j = 0; j < kernelSize; ++j) {
            flatKernel[i * kernelSize + j] = kernel[i][j];
        }
    }

    // Выполнение свертки с использованием CUDA
    cv::Mat outputImage = performConvolution(inputImage, flatKernel, kernelSize);

    // Вывод результатов
    cv::imshow("Input Image", inputImage);
    cv::imshow("Convolved Image", outputImage);
    cv::waitKey(0);

    return 0;
}