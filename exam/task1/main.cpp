#include <opencv2/opencv.hpp>
#include <omp.h>
#include <random>

using namespace cv;
using namespace std;


// Функция для применения сверточного слоя с заданным ядром
Mat applyConvolution(const Mat& inputImage, const Mat& kernel) {
    Mat outputImage = inputImage.clone();
    // чтобы не выйти за пределы изображения
    int kernelRadius = kernel.rows / 2;

#pragma omp parallel for collapse(2) shared(inputImage, outputImage, kernel)
    for (int y = kernelRadius; y < inputImage.rows - kernelRadius; ++y) {
        for (int x = kernelRadius; x < inputImage.cols - kernelRadius; ++x) {
            // Применение свертки в текущем пикселе
            float sum = 0.0f;
            // ky - вертикальное смещение относительно текущего пикселя (y, x).
            // kx - горизонтальное смещение относительно текущего пикселя (y, x).
            for (int ky = -kernelRadius; ky <= kernelRadius; ++ky) {
                for (int kx = -kernelRadius; kx <= kernelRadius; ++kx) {
                    sum += inputImage.at<uchar>(y + ky, x + kx) * kernel.at<float>(ky + kernelRadius, kx + kernelRadius);
                }
            }

            // Запись результата в выходное изображение 
            // для текущего пикселя с координатами (y, x)

            // uchar - 8-битный беззнаковый символ (unsigned char)
            // saturate_cast - приведение значений из одного 
            // числового типа данных в другой
            outputImage.at<uchar>(y, x) = saturate_cast<uchar>(sum);
        }
    }

    return outputImage;
}

// ядро свертки для усиления границ
Mat simpleKernel = (Mat_<float>(3, 3) << -1, -1, -1, -1, 8, -1, -1, -1, -1);

Mat simpleGenerateRandomKernel(int size) {
    Mat kernel(size, size, CV_32F);
    srand(static_cast<unsigned>(time(0)));  // Инициализация генератора случайных чисел

    // Генерация случайных чисел и заполнение матрицы
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            // static_cast<float> - приводит целочисленное значение к типу float
            // RAND_MAX - 0x7fff - 32767 
            kernel.at<float>(i, j) = static_cast<float>(rand()) / RAND_MAX - 0.5f;
        }
    }

    return kernel;
}

int main() {
    // Загрузка изображения с использованием OpenCV
    Mat inputImage = imread("test.png", IMREAD_GRAYSCALE);

    if (inputImage.empty()) {
        cerr << "Could not open or find the image!" << endl;
        return -1;
    }

    // Генерация случайного ядра свертки 3x3
    //Mat kernel = simpleGenerateRandomKernel(3);

    // Или применение ядра свертки для усиления границ
    Mat kernel = simpleKernel;

    // Применение сверточного слоя
    Mat outputImage = applyConvolution(inputImage, kernel);

    // Вывод результатов
    namedWindow("Input Image", WINDOW_NORMAL);
    namedWindow("Output Image", WINDOW_NORMAL);

    imshow("Input Image", inputImage);
    imshow("Output Image", outputImage);

    waitKey(0);

    return 0;
}
