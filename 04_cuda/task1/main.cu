#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "opencv2/opencv.hpp"

// Объявляем массив квантованных цветов в константной памяти
__constant__ uchar3 d_quantizedColors[10];

// CUDA-ядро для квантования цветового изображения
__global__ void quantizeColorGPU(uchar* input, uchar* output, int rows, int cols, int K) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // Проверяем, что текущий поток находится в пределах размеров изображения
    if (i < rows && j < cols) {
        int idx = i * cols + j;

        // Рассчитываем интенсивность пикселя
        int intensity = (input[3 * idx + 2] + input[3 * idx + 1] + input[3 * idx]) / 3;

        // Рассчитываем индекс квантованного цвета на основе интенсивности
        int quantIndex = (intensity * (K - 1)) / 255;

        // Получаем квантованный цвет из константной памяти
        uchar3 quantColor = d_quantizedColors[quantIndex];

        // Записываем квантованный цвет в выходной массив
        output[3 * idx] = quantColor.x;
        output[3 * idx + 1] = quantColor.y;
        output[3 * idx + 2] = quantColor.z;
    }
}

// Функция для квантования цвета и сохранения результата
void quantizeAndSave(const cv::Mat& input, const char* outputFilename, int K) {
    int rows = input.rows;
    int cols = input.cols;
    int image_size = rows * cols * 3;

    // Выделение памяти на устройстве для входного и выходного изображения
    uchar* d_input, * d_output;
    cudaMalloc((void**)&d_input, image_size);
    cudaMalloc((void**)&d_output, image_size);

    // Выделение памяти на хосте для результата
    uchar* result = new uchar[image_size];

    // Проверка успешности выделения памяти на устройстве и хосте
    if (!d_input || !d_output || !result) {
        fprintf(stderr, "Memory allocation error\n");
        return;
    }

    // Копирование входного изображения на устройство
    cudaMemcpy(d_input, input.data, image_size, cudaMemcpyHostToDevice);

    // Задаем значения массива d_quantizedColors на хосте
    uchar3 h_quantizedColors[10] = {
        make_uchar3(0, 0, 0),
        make_uchar3(127, 0, 0),
        make_uchar3(255, 0, 0),
        make_uchar3(0, 127, 0),
        make_uchar3(0, 255, 0),
        make_uchar3(0, 0, 127),
        make_uchar3(0, 0, 255),
        make_uchar3(127, 0, 127),
        make_uchar3(127, 127, 0),
        make_uchar3(0, 127, 127)
    };

    // Копирование массива квантованных цветов в константную память
    cudaMemcpyToSymbol(d_quantizedColors, h_quantizedColors, sizeof(h_quantizedColors));

    // Задание конфигурации блока и сетки для запуска ядра
    dim3 threadsPerBlock(16, 16);
    // когда cols или rows не являются точным кратным threadsPerBlock.x
    dim3 numBlocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Замер времени выполнения ядра
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Запуск ядра
    quantizeColorGPU << <numBlocks, threadsPerBlock >> > (d_input, d_output, rows, cols, K);


    cudaEventRecord(stop);

    // Синхронизация событий и вычисление времени выполнения
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Время выполнения: %f мс\n", milliseconds);
    //std::cout << milliseconds << std::endl;

    // Копирование результата на хост
    cudaMemcpy(result, d_output, image_size, cudaMemcpyDeviceToHost);

    // Создание объекта cv::Mat для результата
    cv::Mat output(rows, cols, CV_8UC3, result);

    // Сохранение результата в файл
    cv::imwrite(outputFilename, output);

    // Освобождение выделенной памяти на устройстве и хосте
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] result;
}


int main() {
    setlocale(LC_ALL, "Russian");

    int K = 10; // Количество уровней квантования

    if (K < 4 || K > 10) {
        std::cout << "Недопустимый уровень квантования. Допустимый диапазон: 4-10." << std::endl;
        return 1;
    }

    // Загрузка изображений
    cv::Mat image1 = cv::imread("data/f1024x768.jpg");
    cv::Mat image2 = cv::imread("data/f1280x960.jpg");
    cv::Mat image3 = cv::imread("data/f2560x1440.jpg");

    if (image1.empty() || image2.empty() || image3.empty()) {
        std::cout << "Ошибка загрузки изображениЙ" << std::endl;
        return 1;
    }

    int numRuns = 10; // Количество запусков

    for (int run = 0; run < numRuns; run++) {
        // Квантование и сохранение изображений
        std::cout << "[*] Тест: " << run + 1 << std::endl;
        quantizeAndSave(image1, ("result/quantized_f1024x768_k" + std::to_string(K) + ".jpg").c_str(), K);
        quantizeAndSave(image2, ("result/quantized_f1280x960_k" + std::to_string(K) + ".jpg").c_str(), K);
        quantizeAndSave(image3, ("result/quantized_f2560x1440_k" + std::to_string(K) + ".jpg").c_str(), K);
    }

    return 0;
}
