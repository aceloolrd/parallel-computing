#include <iostream>
#include <mpi.h>
#include <cstdlib>
#include <ctime>
#include <vector>

using namespace std;

typedef vector<__int16> matrix;

void initializeRandomMatrix(matrix& mat, int size) {
    for (int i = 0; i < size; i++) {
        mat[i] = rand() % 6;
    }
}
void multiplyScalar(const int size, const matrix& Am, const matrix& Bm, matrix& Rs) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            for (int k = 0; k < size; ++k) {
                Rs[i * size + j] += Am[i * size + k] * Bm[k * size + j];
            }
        }
    }
}
void MPI_process_rank_0(int size, int sizeProc, int offset, int blockSize, int rank, matrix& Am, matrix& Bm, matrix& Rt) {
    for (int dest = 1; dest < sizeProc; dest++) {
        int currentBlockSize = blockSize + (dest <= (size % sizeProc) ? 1 : 0);
        MPI_Send(&offset, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
        MPI_Send(&currentBlockSize, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
        MPI_Send(&Am[offset * size], currentBlockSize * size, MPI_SHORT, dest, 1, MPI_COMM_WORLD);
        MPI_Send(&Bm[0], size * size, MPI_SHORT, dest, 1, MPI_COMM_WORLD);
        offset += currentBlockSize;
    }
    for (int i = offset; i < offset + blockSize; i++) {
        for (int j = 0; j < size; j++) {
            Rt[i * size + j] = 0;
            for (int k = 0; k < size; k++) {
                Rt[i * size + j] += Am[i * size + k] * Bm[k * size + j];
            }
        }
    }
    for (int i = 1; i < sizeProc; i++) {
        int currentBlockSize = blockSize + (i <= (size % sizeProc) ? 1 : 0);
        MPI_Recv(&offset, 1, MPI_INT, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&currentBlockSize, 1, MPI_INT, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&Rt[offset * size], currentBlockSize * size, MPI_SHORT, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}
void MPI_process(int size, int offset, int blockSize, matrix& Am, matrix& Bm, matrix& Rt) {
    MPI_Recv(&offset, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&blockSize, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    Am.resize(blockSize * size);
    MPI_Recv(&Am[0], blockSize * size, MPI_SHORT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    Bm.resize(size * size);
    MPI_Recv(&Bm[0], size * size, MPI_SHORT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    Rt.resize(blockSize * size);
    for (int i = 0; i < blockSize; i++) {
        for (int j = 0; j < size; j++) {
            Rt[i * size + j] = 0;
            for (int k = 0; k < size; k++) {
                Rt[i * size + j] += Am[i * size + k] * Bm[k * size + j];
            }
        }
    }
    MPI_Send(&offset, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
    MPI_Send(&blockSize, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
    MPI_Send(&Rt[0], blockSize * size, MPI_SHORT, 0, 2, MPI_COMM_WORLD);
}

bool areMatricesEqual(const matrix& A, const matrix& B) {
    int size = A.size();
    for (int i = 0; i < size; i++) {
        if (A[i] != B[i]) {
            return false;
        }
    }
    return true;
}
int main(int argc, char* argv[]) {
    int size = 256;
    matrix Am(size * size);
    matrix Bm(size * size);
    matrix Rs(size * size, 0);
    matrix Rt(size * size, 0);
    // Инициализация генератора случайных чисел текущим временем
    srand(static_cast<unsigned>(time(nullptr)));
    // Заполнение матриц случайными числами
    initializeRandomMatrix(Am, size * size);
    initializeRandomMatrix(Bm, size * size);

    MPI_Init(&argc, &argv);

    int rank, sizeProc;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &sizeProc);

    int blockSize = size / sizeProc;
    int offset = 0;

    if (rank == 0) {
        auto startScalar = MPI_Wtime();
        multiplyScalar(size, Am, Bm, Rs);
        auto endScalar = MPI_Wtime();

        cout << "[*] Time scalar: " << endScalar - startScalar << " s." << endl;

        auto start = MPI_Wtime();
        MPI_process_rank_0(size, sizeProc, offset, blockSize, rank, Am, Bm, Rt);
        auto end = MPI_Wtime();

        cout << "[*] Time MPI: " << end - start << " s." << endl;

        bool me = areMatricesEqual(Rs, Rt);
        cout << endl << "The results of MPI and scalar multiplication are " << (me ? "equal." : "not equal.") << endl << endl;

        Am.clear();
        Bm.clear();
        Rs.clear();
        Rt.clear();
    }

    else {
        MPI_process(size, offset, blockSize, Am, Bm, Rt);

        Am.clear();
        Bm.clear();
        Rt.clear();
    };

    MPI_Finalize();
    return 0;
}
