#include "MatrixMultiplication.hpp"
#include <stdio.h>
#include <omp.h>

void sequentialMatrixMatrixMul(int** A, int** B, int** C) {
	for (int row = 0; row != ROWS; ++row) {
 		for (int col = 0; col != COLS; ++col) {
			int sum = 0;
			for (int k = 0; k != COLS; ++k)
				sum += A[row][k] * B[k][col];
			C[row][col] = sum;
		}
	}
}

void openmpMatrixMatrixMul(int** A, int** B, int** C) {
	int col, row, sum, k;
	#pragma omp parallel for private(row, col, sum, k)
	for (row = 0; row < ROWS; ++row) {
		// #pragma omp parallel for private(row, col, sum, k)
 		for (col = 0; col < COLS; ++col) {
			sum = 0;
			// #pragma omp parallel for firstprivate(sum) lastprivate(sum)
			for (k = 0; k < COLS; ++k)
				sum += A[row][k] * B[k][col];
			C[row][col] = sum;
		}
	}
}

