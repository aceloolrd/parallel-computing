#include "opencv2/opencv.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <omp.h> 

// Определяем квантованные цвета как статический массив
static const std::vector<cv::Scalar> quantizedColors = {
    cv::Scalar(0, 0, 0),
    cv::Scalar(127, 0, 0),
    cv::Scalar(255, 0, 0),
    cv::Scalar(0, 127, 0),
    cv::Scalar(0, 255, 0),
    cv::Scalar(0, 0, 127),
    cv::Scalar(0, 0, 255),
    cv::Scalar(127, 0, 127),
    cv::Scalar(127, 127, 0),
    cv::Scalar(0, 127, 127)
};

// Квантование цвета на основе интенсивности и K
cv::Scalar quantizeColor(int intensity, int K) {
    int quantIndex = (intensity * (K - 1)) / 255;
    return quantizedColors[quantIndex];
}

// Обработка изображения
void processImage(const cv::Mat& img, int K, int itr, bool parallel = false) {
    int rows = img.rows;
    int cols = img.cols;
    cv::Mat quantizedImage(rows, cols, img.type());

    auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for if (parallel)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cv::Vec3b pixel = img.at<cv::Vec3b>(i, j);
            int intensity = (pixel[2] + pixel[1] + pixel[0]) / 3;
            cv::Scalar quantColor = quantizeColor(intensity, K);
            quantizedImage.at<cv::Vec3b>(i, j) = cv::Vec3b(static_cast<uchar>(quantColor[0]),
                static_cast<uchar>(quantColor[1]),
                static_cast<uchar>(quantColor[2]));
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Отображение времени выполнения в консоли
    std::cout << (parallel ? "Многопоточное" : "Однопоточное") << " время выпол-нения (Запуск " << itr + 1 << "): " << duration.count() << " мс\n";

    cv::imwrite((parallel ? "../result/multi_" : "../result/single_") + std::to_string(K) + "_quantized.jpg", quantizedImage);
}

int main() {
    setlocale(LC_ALL, "Russian");

    int K = 10; // Количество уровней квантования

    if (K < 4 || K > 10) {
        std::cout << "Недопустимый уровень квантования. Допустимый диапазон: 4-10." << std::endl;
        return 1;
    }

    cv::Mat img = cv::imread("../data/f2560x1440.jpg");
    if (img.empty()) {
        std::cout << "Ошибка загрузки изображения" << std::endl;
        return 1;
    }

    int numRuns = 1; // Количество запусков

    for (int run = 0; run < numRuns; run++) {
        processImage(img, K, run, false); // Однопоточное выполнение
        processImage(img, K, run, true);  // Многопоточное выполнение
    }

    return 0;
}
