
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef SIZE
# define SIZE 4
#endif

#ifndef BLOCK_SIZE
# define BLOCK_SIZE 2
#endif

#ifndef MULT_TYPE
# define MULT_TYPE 1
#endif

static __always_inline int min(int a, int b)
{
	return a < b ? a : b;
}

static void init_matrix(float **array, int size)
{
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			array[i][j] = (float)rand() / RAND_MAX * 10;
		}
	}
}

static void zero_matrix(float **array, int size)
{
	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < size; ++j) {
			array[i][j] = 0;
		}
	}
}

static void print_matrix(float **array, int size)
{
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			printf("\t%f", array[i][j]);
		}
		printf("\n");
	}
}

void multiply_simple(float **A, float **B, float **C, int size)
{
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
//#pragma novector
			for (int k = 0; k < size; k++) {
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}
}

static void multiply_cache_friendly(float **A, float **B, float **C, int size)
{
	for (int i = 0; i < size; i++) {
		for (int k = 0; k < size; k++) {
//#pragma novector
			for (int j = 0; j < size; j++) {
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}
}

__always_inline
static void __multiply_block(float **A, float **B, float **C, int off_i, int off_j, int off_k, int size, int block_size)
{
	const int max_i = min(off_i + block_size, size);
	const int max_j = min(off_j + block_size, size);
	const int max_k = min(off_k + block_size, size);

	for (int i = off_i; i < max_i; ++i) {
		for (int j = off_j; j < max_j; ++j) {
			for (int k = off_k; k < max_k; ++k) {
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}
}

static void multiply_block(float **A, float **B, float **C, int size, int block_size)
{
	for (int i = 0; i < size; i += block_size) {
		for (int j = 0; j < size; j += block_size) {
			for (int k = 0; k < size; k += block_size) {
				__multiply_block(A, B, C, i, j, k, size, block_size);
			}
		}
	}
}

static float timer_diff(struct timespec *begin, struct timespec *end)
{
	float res = 0;

	res += end->tv_sec - begin->tv_sec;
	res += (end->tv_nsec - begin->tv_nsec) * 1e-9f;
	return res;
}

static void alloc_matrix(float ***a, float ***b, float ***c, int size)
{
	*a = malloc(sizeof(float *) * size);
	*b = malloc(sizeof(float *) * size);
	*c = malloc(sizeof(float *) * size);

	for (int i = 0; i < size; ++i) {
		(*a)[i] = malloc(sizeof(float) * size);
		(*b)[i] = malloc(sizeof(float) * size);
		(*c)[i] = malloc(sizeof(float) * size);
	}
}

int main()
{
	float **A, **B, **C;
	struct timespec begin, end;

	srand(time(NULL));

	alloc_matrix(&A, &B, &C, SIZE);
	init_matrix((float **)A, SIZE);
	printf("init A\n");
	init_matrix((float **)B, SIZE);
	printf("init B\n");
	zero_matrix((float **)C, SIZE);
	printf("init C\n");

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &begin);
	switch (MULT_TYPE) {
	case 1:
		multiply_simple(A, B, C, SIZE);
		break;
	case 2:
		multiply_cache_friendly(A, B, C, SIZE);
		break;
	case 3:
		multiply_block(A, B, C, SIZE, BLOCK_SIZE);
		break;
	default:
		printf("mult type\n");
		abort();
	}
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);

	if (SIZE < 20) {
		printf("A:\n");
		print_matrix(A, SIZE);
		printf("B:\n");
		print_matrix(B, SIZE);

		printf("multiply_simple:\n");
		zero_matrix(C, SIZE);
		multiply_simple(A, B, C, SIZE);
		print_matrix(C, SIZE);

		printf("multiply_cache_friendly:\n");
		zero_matrix(C, SIZE);
		multiply_cache_friendly(A, B, C, SIZE);
		print_matrix(C, SIZE);

		printf("multiply_block:\n");
		zero_matrix(C, SIZE);
		multiply_block(A, B, C, SIZE, BLOCK_SIZE);
		print_matrix(C, SIZE);
	}

	printf("elapsed time: %fs\n", timer_diff(&begin, &end));

	return 0;
}
