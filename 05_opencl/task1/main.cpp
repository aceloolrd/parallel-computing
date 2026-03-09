#include <CL/cl.h>
#include "opencv2/opencv.hpp"
#include <iostream>
#include <chrono>


// Объявляем массив квантованных цветов в глобальной памяти
cl_uchar3 h_quantizedColors[10] = {
    {0, 0, 0},
    {127, 0, 0},
    {255, 0, 0},
    {0, 127, 0},
    {0, 255, 0},
    {0, 0, 127},
    {0, 0, 255},
    {127, 0, 127},
    {127, 127, 0},
    {0, 127, 127}
};

// Определение исходного кода OpenCL ядра
const char* kernelSource = R"(
    __kernel void quantizeColorGPU(__global uchar* input, __global uchar* output, int rows, int cols, __global uchar3* quantizedColors, int K) {
        int i = get_global_id(1);
        int j = get_global_id(0);

        if (i < rows && j < cols) {
            int idx = i * cols + j;
            int intensity = (input[3 * idx + 2] + input[3 * idx + 1] + input[3 * idx]) / 3;
            int quantIndex = (intensity * (K - 1)) / 255;

            uchar3 quantColor = quantizedColors[quantIndex];

            output[3 * idx] = quantColor.x;
            output[3 * idx + 1] = quantColor.y;
            output[3 * idx + 2] = quantColor.z;
        }
    }
)";

void checkOpenCLError(cl_int err, const char* operation) {
    if (err != CL_SUCCESS) {
        std::cerr << "OpenCL Error during " << operation << ": " << err << std::endl;
        throw std::runtime_error("OpenCL error");
    }
}

void quantizeAndSave(const cv::Mat& input, const char* outputFilename, int K) {
    int rows = input.rows;
    int cols = input.cols;
    int image_size = rows * cols * 3;

    // Выделение памяти на хосте для результата
    uchar* result = new uchar[image_size];

    // Инициализация OpenCL
    cl_platform_id platform;
    checkOpenCLError(clGetPlatformIDs(1, &platform, nullptr), "clGetPlatfor-mIDs");

    cl_device_id device;
    checkOpenCLError(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr), "clGetDeviceIDs");

    // Создание контекста
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);

    // Создание очереди
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, nullptr);

    // Создание буферов OpenCL
    cl_mem d_input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uchar) * image_size, (void*)input.data, nullptr);
    cl_mem d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(uchar) * image_size, nullptr, nullptr);

    // Компиляция и создание ядра OpenCL
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, nullptr);
    checkOpenCLError(clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr), "clBuildProgram");

    // Проверка статуса сборки программы и вывод журнала компиляции при ошибке
    cl_build_status buildStatus;
    checkOpenCLError(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &buildStatus, nullptr), "clGetProgramBuildInfo");

    if (buildStatus != CL_SUCCESS) {
        size_t logSize;
        checkOpenCLError(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize), "clGetProgramBuildInfo");

        char* log = new char[logSize];
        checkOpenCLError(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log, nullptr), "clGetProgramBuildInfo");

        std::cerr << "Ошибка компиляции OpenCL-программы: " << log << std::endl;
        delete[] log;

        throw std::runtime_error("Ошибка компиляции OpenCL-программы");
    }

    cl_kernel kernel = clCreateKernel(program, "quantizeColorGPU", nullptr);

    // Создание буфера для quantizedColors
    cl_mem d_quantizedColors = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_uchar3) * K, h_quantizedColors, nullptr);

    // Создание переменной K
    int K_value = K;

    // Установка аргументов ядра
    checkOpenCLError(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input), "clSetKernelArg");
    checkOpenCLError(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output), "clSetKernelArg");
    checkOpenCLError(clSetKernelArg(kernel, 2, sizeof(int), &rows), "clSetKernelArg");
    checkOpenCLError(clSetKernelArg(kernel, 3, sizeof(int), &cols), "clSetKernelArg");

    // Передача quantizedColors и K как аргументов ядра
    checkOpenCLError(clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_quantizedColors), "clSetKernelArg");
    checkOpenCLError(clSetKernelArg(kernel, 5, sizeof(int), &K_value), "clSetKernelArg");


    // Замер времени выполнения chrono
    auto start_time = std::chrono::high_resolution_clock::now();

    // Запуск ядра
    size_t globalWorkSize[2] = { static_cast<size_t>(cols), static_cast<size_t>(rows) };
    // Для замера времени cl
    checkOpenCLError(clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr), "clEnqueueNDRangeKernel");
    checkOpenCLError(clFinish(queue), "clFinish");

    // Замер времени выполнения chrono
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    float microseconds = duration.count();
    std::cout << "Время выполнения: " << microseconds / 1000 << " мс" << std::endl;


    // Чтение данных с устройства
    checkOpenCLError(clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, sizeof(uchar) * image_size, result, 0, nullptr, nullptr), "clEnqueueReadBuffer");

    // Создание объекта cv::Mat для результата
    cv::Mat output(rows, cols, CV_8UC3, result);

    // Сохранение результата в файл
    cv::imwrite(outputFilename, output);

    // Освобождение ресурсов OpenCL
    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    // Освобождение выделенной памяти на хосте
    delete[] result;
}

int main() {
    setlocale(LC_ALL, "Russian");

    int K = 10;

    if (K < 4 || K > 10) {
        std::cout << "Недопустимый уровень квантования. Допустимый диапазон: 4-10." << std::endl;
        return 1;
    }

    // Загрузка изображений
    cv::Mat image1 = cv::imread("data/f1024x768.jpg");
    cv::Mat image2 = cv::imread("data/f1280x960.jpg");
    cv::Mat image3 = cv::imread("data/f2560x1440.jpg");

    if (image1.empty() || image2.empty() || image3.empty()) {
        std::cout << "Ошибка загрузки изображений" << std::endl;
        return 1;
    }

    int numRuns = 10; // Количество запусков

    // Квантование и сохранение изображения
    try {
        for (int run = 0; run < numRuns; run++) {
            // Квантование и сохранение изображений
            std::cout << "[*] Тест: " << run + 1 << std::endl;
            quantizeAndSave(image1, ("result/quantized_f1024x768_k" + std::to_string(K) + ".jpg").c_str(), K);
            quantizeAndSave(image2, ("result/quantized_f1280x960_k" + std::to_string(K) + ".jpg").c_str(), K);
            quantizeAndSave(image3, ("result/quantized_f2560x1440_k" + std::to_string(K) + ".jpg").c_str(), K);
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
