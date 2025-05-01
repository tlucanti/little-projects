
#include "common.h"

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
		pthread_create(&threads[i], NULL, (void *)multiply_worker,
			       (void *)i);
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

int main(int argc, char **argv)
{
	float **A, **B, **C;
	struct timespec begin, end;

	if (argc != 2)
		goto args;
	srand(123);

	alloc_matrix3(&A, &B, &C, SIZE);
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
	default:
		goto args;
	}
	clock_gettime(CLOCK_MONOTONIC, &end);

	if (SIZE < 20)
		print_matrix(C, SIZE);

	printf("%fs\n", time_diff(&begin, &end));

	return 0;

args:
	printf("invalid args, expected 's', 'o', or 'p'\n"
	       "\ts: single thread execution\n"
	       "\to: OMP multithreaded execution\n"
	       "\tp: pthread multithreaded execution\n");
	return 1;
}
