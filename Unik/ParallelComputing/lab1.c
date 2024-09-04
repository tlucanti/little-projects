
#include <time.h>
#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>

#ifndef SIZE
# define SIZE 10
#endif

#ifndef NR_THREADS
# define NR_THREADS 8
#endif

static void init_matrix(float **array, int size)
{
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			array[i][j] = (float)rand() / (float)RAND_MAX * 10;
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

static void multiply_single_thread(float **A, float **B, float **C, int size)
{
	for (int i = 0; i < size; i++) {
		for (int k = 0; k < size; k++) {
			for (int j = 0; j < size; j++) {
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}
}

static float **pA;
static float **pB;
static float **pC;

static void *multiply_worker(unsigned long tid)
{
	for (int i = tid; i < SIZE; i += NR_THREADS) {
		for (int k = 0; k < SIZE; k++) {
			for (int j = 0; j < SIZE; j++) {
				pC[i][j] += pA[i][k] * pB[k][j];
			}
		}
	}

	return NULL;
}

static void multiply_pthread(float **A, float **B, float **C, int size)
{
	(void)size;
	pthread_t threads[NR_THREADS];
	pA = A;
	pB = B;
	pC = C;

	for (unsigned long i = 0; i < NR_THREADS; i++) {
		pthread_create(&threads[i], NULL, (void *)multiply_worker, (void *)i);
	}

	for (int i = 0; i < NR_THREADS; i++) {
		pthread_join(threads[i], NULL);
	}
}

static void multiply_omp(float **A, float **B, float **C, int size)
{
#pragma omp parallel for num_threads(NR_THREADS)
	for (int i = 0; i < size; i++) {
		for (int k = 0; k < size; k++) {
			for (int j = 0; j < size; j++) {
				C[i][j] += A[i][k] * B[k][j];
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

int main(int argc, char **argv)
{
	float **A, **B, **C;
	struct timespec begin, end;

	if (argc != 2)
		return 1;
	srand(123);

	alloc_matrix(&A, &B, &C, SIZE);
	init_matrix((float **)A, SIZE);
	init_matrix((float **)B, SIZE);
	zero_matrix((float **)C, SIZE);

	clock_gettime(CLOCK_MONOTONIC, &begin);
	switch (argv[1][0]) {
	case 's':
		multiply_single_thread(A, B, C, SIZE);
		break;
	case 'o':
		multiply_omp(A, B, C, SIZE);
		break;
	case 'p':
		multiply_pthread(A, B, C, SIZE);
		break;
	}
	clock_gettime(CLOCK_MONOTONIC, &end);

	if (SIZE < 20)
		print_matrix(C, SIZE);

	printf("%fs\n", timer_diff(&begin, &end));

	return 0;
}
