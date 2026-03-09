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
        float result = 0.0f;

        for (int i = -halfKernelSize; i <= halfKernelSize; ++i) {
            for (int j = -halfKernelSize; j <= halfKernelSize; ++j) {
                int currentCol = max(0, min(width - 1, col + j));
                int currentRow = max(0, min(height - 1, row + i));

                uchar pixelValue = input[currentRow * width + currentCol];
                result += pixelValue * kernel[(i + halfKernelSize) * kernelSize + (j + halfKernelSize)];
            }
        }

        output[row * width + col] = (uchar)max(0.0f, min(255.0f, result));
    }
}


cv::Mat performConvolution(const cv::Mat& input, const float* kernel, int kernelSize) {
    int width = input.cols;
    int height = input.rows;

    cv::Mat output(height, width, CV_8UC1);

    uchar* d_input, * d_output;
    float* d_kernel;
    cudaMalloc(&d_input, width * height * sizeof(uchar));
    cudaMalloc(&d_output, width * height * sizeof(uchar));
    cudaMalloc(&d_kernel, kernelSize * kernelSize * sizeof(float));

    cudaMemcpy(d_input, input.data, width * height * sizeof(uchar), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    convolutionKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, d_kernel, kernelSize);

    cudaMemcpy(output.data, d_output, width * height * sizeof(uchar), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);

    return output;
}

int main() {
    cv::Mat inputImage = cv::imread("test.png", cv::IMREAD_GRAYSCALE);

    float kernel[3][3] = { {1, 1, 1},
                           {1, -8, 1},
                           {1, 1, 1} };

    int kernelSize = 3;
    float flatKernel[9];
    for (int i = 0; i < kernelSize; ++i)
        for (int j = 0; j < kernelSize; ++j)
            flatKernel[i * kernelSize + j] = kernel[i][j];

    cv::Mat outputImage = performConvolution(inputImage, flatKernel, kernelSize);

    cv::imshow("Input Image", inputImage);
    cv::imshow("Convolved Image", outputImage);
    cv::waitKey(0);

    return 0;
}
