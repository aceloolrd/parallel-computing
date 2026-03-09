#include <iostream>
#include <vector>
#include <chrono>
#include <immintrin.h>
#include <thread>

using namespace std;

typedef vector<vector<__int16>> matrix;

void initializeRandomMatrix(matrix& mat, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            mat[i][j] = rand() % 6;
        }
    }
}

matrix transposeMatrix(const matrix& mat) {
    int size = mat.size();
    matrix result(size, vector<__int16>(size, 0));

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            result[i][j] = mat[j][i];
        }
    }

    return result;
}

void multiplyScalarSingleThread(const matrix& Am, const matrix& Bm, matrix& Rs, int start, int end) {
    int size = Am.size();

    for (int i = start; i < end; ++i) {
        for (int j = 0; j < size; ++j) {
            for (int k = 0; k < size; ++k) {
                Rs[i][j] += Am[i][k] * Bm[k][j];
            }
        }
    }
}

void multiplyScalar(const matrix& Am, const matrix& Bm, matrix& Rs, int numThreads) {
    int size = Am.size();
    vector<thread> threads;

    int chunkSize = size / numThreads;

    for (int i = 0; i < numThreads; ++i) {
        int start = i * chunkSize;
        int end = (i == numThreads - 1) ? size : (i + 1) * chunkSize;

        threads.emplace_back(multiplyScalarSingleThread, std::ref(Am), std::ref(Bm), std::ref(Rs), start, end);
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

void multiplyVectorSingleThread(const matrix& Am, const matrix& BT, matrix& Rv, int start, int end) {
    int size = Am.size();

    for (int i = start; i < end; i++) {
        for (int j = 0; j < size; j++) {
            __m128i sum = _mm_setzero_si128();
            for (int k = 0; k < size; k += 8) {
                __m128i al = _mm_loadu_si128((__m128i*) & Am[i][k]);
                __m128i bl = _mm_loadu_si128((__m128i*) & BT[j][k]);
                __m128i mult = _mm_mullo_epi16(al, bl);
                sum = _mm_add_epi16(sum, mult);
            }

            __m128i line_sum = _mm_hadd_epi16(_mm_hadd_epi16(_mm_hadd_epi16(sum, sum), _mm_hadd_epi16(sum, sum)), _mm_hadd_epi16(_mm_hadd_epi16(sum, sum), _mm_hadd_epi16(sum, sum)));
            _mm_storeu_si16((__m128i*) & Rv[i][j], line_sum);
        }
    }
}

void multiplyVector(const matrix& Am, const matrix& BT, matrix& Rv, int numThreads) {
    int size = Am.size();
    vector<thread> threads;

    int chunkSize = size / numThreads;

    for (int i = 0; i < numThreads; ++i) {
        int start = i * chunkSize;
        int end = (i == numThreads - 1) ? size : (i + 1) * chunkSize;

        threads.emplace_back(multiplyVectorSingleThread, std::ref(Am), std::ref(BT), std::ref(Rv), start, end);
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

bool areMatricesEqual(const matrix& A, const matrix& B) {
    int size = A.size();

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (A[i][j] != B[i][j]) {
                return false;
            }
        }
    }

    return true;
}

int main() {
    int size = 256; // 4096
    int numThreads = thread::hardware_concurrency();

    matrix Am(size, vector<__int16>(size, 0));
    matrix Bm(size, vector<__int16>(size, 0));
    matrix Rs_scalar_single(size, vector<__int16>(size, 0)); // Результат ска-лярного однопоточного
    matrix Rs_scalar_multi(size, vector<__int16>(size, 0));  // Результат ска-лярного многопоточного
    matrix Rv_simd_single(size, vector<__int16>(size, 0));   // Результат SIMD однопоточного
    matrix Rv_simd_multi(size, vector<__int16>(size, 0));    // Результат SIMD многопоточного

    initializeRandomMatrix(Am, size);
    initializeRandomMatrix(Bm, size);

    // Однопоточное скалярное умножение
    cout << "Scalar multiplication (single-threaded): \n";
    auto start_1 = chrono::high_resolution_clock::now();
    multiplyScalar(Am, Bm, Rs_scalar_single, 1); // Используем 1 поток
    auto end_1 = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> dur1 = (end_1 - start_1);
    cout << "Time: " << dur1.count() << " milliseconds\n";

    // Многопоточное скалярное умножение
    cout << "Scalar multiplication (multi-threaded): \n";
    auto start_2 = chrono::high_resolution_clock::now();
    multiplyScalar(Am, Bm, Rs_scalar_multi, numThreads); // Используем указанное количество потоков
    auto end_2 = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> dur2 = (end_2 - start_2);
    cout << "Time: " << dur2.count() << " milliseconds\n";

    matrix BT = transposeMatrix(Bm);

    // Однопоточное SIMD умножение
    cout << "SIMD multiplication (single-threaded): \n";
    auto start_3 = chrono::high_resolution_clock::now();
    multiplyVector(Am, BT, Rv_simd_single, 1); // Используем 1 поток
    auto end_3 = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> dur3 = (end_3 - start_3);
    cout << "Time: " << dur3.count() << " milliseconds\n";

    // Многопоточное SIMD умножение
    cout << "SIMD multiplication (multi-threaded): \n";
    auto start_4 = chrono::high_resolution_clock::now();
    multiplyVector(Am, BT, Rv_simd_multi, numThreads); // Используем указанное количество потоков
    auto end_4 = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> dur4 = (end_4 - start_4);
    cout << "Time: " << dur4.count() << " milliseconds\n";

    // Проверка на равенство результатов
    bool th_scalar_equal = areMatricesEqual(Rs_scalar_single, Rs_scalar_multi);
    bool simd_equal = areMatricesEqual(Rv_simd_single, Rs_scalar_single);
    bool th_simd_equal = areMatricesEqual(Rs_scalar_single, Rv_simd_multi);


    cout << "The results of multithreaded scalar multiplication are " << (th_scalar_equal ? "equal" : "not equal") << endl;
    cout << "The results of single-threaded SIMD multiplication are " << (simd_equal ? "equal" : "not equal") << endl;
    cout << "The results of multithreaded SIMD multiplication are " << (th_simd_equal ? "equal" : "not equal") << endl;

    return 0;
}
