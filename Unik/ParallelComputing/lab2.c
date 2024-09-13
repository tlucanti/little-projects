
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

#define SIZE 4096
#define NR_THREADS 20
#ifndef FINALIZE_PASSES
//# define FINALIZE_PASSES 0
#error
#endif

typedef float flt;

static flt random_float(void)
{
	flt r = (flt)random() / (flt)RAND_MAX;
	return r * (flt)2 - (flt)1;
}

static double time_diff(struct timespec end, struct timespec start)
{
	double sec = end.tv_sec - start.tv_sec;
	sec += (end.tv_nsec -  start.tv_nsec) * 1e-9;
	return sec;
}

static flt absf(flt val)
{
	if (val < 0)
		return -val;

	return val;
}

static void allocate_matrix(flt ***mat, int size)
{
	*mat = malloc(sizeof(flt *) * size);

	if (*mat == NULL)
		goto oom;

	for (int y = 0; y < size; y++) {
		(*mat)[y] = malloc(sizeof(flt) * (size + 1));
		if ((*mat)[y] == NULL)
			goto oom;
	}
	return;

oom:
	printf("out of memory\n");
	abort();
}

static void generate_matrix(flt **mat, int size)
{
	for (int y = 0; y < size; y++) {
		for (int x = 0; x <= size; x++) {
			int v = rand() % 18; // 0 .. 18
			v -= 9; // -9 .. 8
			if (v >= 0)
				v++; // -9 .. -1, 1 .. 9

			mat[y][x] = v;
		}
	}
}

static void copy_matrix(flt **dst, flt **src, int size)
{
	for (int y = 0; y < size; y++) {
		for (int x = 0; x <= size; x++) {
			dst[y][x] = src[y][x];
		}
	}
}

static void print_matrix(flt **mat, int size)
{
	for (int y = 0; y < size; y++) {
		for (int x = 0; x < size; x++) {
			printf("% 8.3f ", mat[y][x]);
		}

		printf(" | ");
		printf("% 8.3f\n", mat[y][size]);
	}
}

static void gauss_single_thread(flt **mat, int size)
{
	// forward pass
	for (int pass = 0; pass < size - 1; pass++) {
		for (int row = pass + 1; row < size; row++) {
			flt frac = mat[row][pass] / mat[pass][pass];

			for (int col = pass + 1; col <= size; col++) {
				mat[row][col] -= frac * mat[pass][col];
			}
		}
	}

	// backward pass
	for (int pass = 0; pass < size; pass++) {
		for (int row = size - 2 - pass; row >= 0; row--) {
			flt frac = mat[row][size - 1 - pass] / mat[size - 1 - pass][size - 1 - pass];

			mat[row][size] -= frac * mat[size - 1 - pass][size];
		}

		mat[size - pass - 1][size] /= mat[size - 1 - pass][size - 1 - pass];
	}

}

static flt **Mat;
static volatile int Pass;
static pthread_barrier_t barrier;

static void *gauss_worker_barrier(long tid)
{
	const int size = SIZE;
	int pass;

	// forward pass
	for (pass = 0; pass < size - 1; pass++) {
		if ((size - 1) - pass < FINALIZE_PASSES) {
			break;
		} else {
			/* multithread main body */
			for (int row = pass + 1 + tid; row < size; row += NR_THREADS) {
				flt frac = Mat[row][pass] / Mat[pass][pass];

				for (int col = pass + 1; col <= size; col++) {
					Mat[row][col] -= frac * Mat[pass][col];
				}
			}
		}
		pthread_barrier_wait(&barrier);
	}

	/* single thread ending */
	if (tid == 0) {
		for (; pass < size - 1; pass++) {
			for (int row = pass + 1; row < size; row++) {
				flt frac = Mat[row][pass] / Mat[pass][pass];

				for (int col = pass + 1; col <= size; col++) {
					Mat[row][col] -= frac * Mat[pass][col];
				}
			}
		}
	}
	pthread_barrier_wait(&barrier);

	// backward pass
	for (pass = 0; pass < size; pass++) {
		if (size - pass < FINALIZE_PASSES) {
			break;
		} else {
			/* multithread main body */
			for (int row = size - 2 - pass - tid; row >= 0; row -= NR_THREADS) {
				flt frac = Mat[row][size - 1 - pass] / Mat[size - 1 - pass][size - 1 - pass];

				Mat[row][size] -= frac * Mat[size - 1 - pass][size];
			}

			pthread_barrier_wait(&barrier);
			if (tid == 0)
				Mat[size - pass - 1][size] /= Mat[size - 1 - pass][size - 1 - pass];
		}
	}

	/* single thread ending */
	if (tid == 0) {
		for (; pass < size; pass++) {
			for (int row = size - 2 - pass - tid; row >= 0; row--) {
				flt frac = Mat[row][size - 1 - pass] / Mat[size - 1 - pass][size - 1 - pass];

				Mat[row][size] -= frac * Mat[size - 1 - pass][size];
			}
			Mat[size - pass - 1][size] /= Mat[size - 1 - pass][size - 1 - pass];
		}
	}

	return NULL;
}

static void gauss_pthread_barrier(flt **mat, int size)
{
	pthread_t threads[NR_THREADS];

	Mat = mat;
	pthread_barrier_init(&barrier, NULL, NR_THREADS);

	for (long i = 0; i < NR_THREADS; i++) {
		pthread_create(&threads[i], NULL, (void *)gauss_worker_barrier, (void *)i);
	}

	for (int i = 0; i < NR_THREADS; i++) {
		pthread_join(threads[i], NULL);
	}

	pthread_barrier_destroy(&barrier);
}

static void *gauss_worker_forward_join(long tid)
{
	const int size = SIZE;
	const int pass = Pass;

	for (int row = pass + 1 + tid; row < size - 1; row += NR_THREADS) {
		flt frac = Mat[row][pass] / Mat[pass][pass];

		for (int col = pass + 1; col <= size; col++) {
			Mat[row][col] -= frac * Mat[pass][col];
		}
	}

	return NULL;
}

static void *gauss_worker_backward_join(long tid)
{
	const int size = SIZE;
	const int pass = Pass;

	for (int row = size - 2 - pass - tid; row >= 0; row -= NR_THREADS) {
		flt frac = Mat[row][size - 1 - pass] / Mat[size - 1 - pass][size - 1 - pass];

		Mat[row][size] -= frac * Mat[size - 1 - pass][size];
	}

	return NULL;
}

static void gauss_pthread_join(flt **mat, int size)
{
	pthread_t threads[NR_THREADS];
	Mat = mat;

	// forward pass
	for (int pass = 0; pass < size - 1; pass++) {
		Pass = pass;
		for (long t = 0; t < NR_THREADS; t++) {
			pthread_create(&threads[t], NULL, (void *)gauss_worker_forward_join, (void *)t);
		}
		for (long t = 0; t < NR_THREADS; t++) {
			pthread_join(threads[t], NULL);
		}
	}

	// backward pass
	for (int pass = 0; pass < size; pass++) {
		Pass = pass;
		for (long t = 0; t < NR_THREADS; t++) {
			pthread_create(&threads[t], NULL, (void *)gauss_worker_backward_join, (void *)t);
		}
		for (long t = 0; t < NR_THREADS; t++) {
			pthread_join(threads[t], NULL);
		}
		mat[size - pass - 1][size] /= mat[size - 1 - pass][size - 1 - pass];
	}
}

static void gauss_omp(flt **mat, int size)
{
	// forward pass
	for (int pass = 0; pass < size - 1; pass++) {
		#pragma omp parallel for
		for (int row = pass + 1; row < size; row++) {
			flt frac = mat[row][pass] / mat[pass][pass];

			for (int col = pass + 1; col <= size; col++) {
				mat[row][col] -= frac * mat[pass][col];
			}
		}
	}

	// backward pass
	for (int pass = 0; pass < size; pass++) {
		#pragma omp parallel for
		for (int row = size - 2 - pass; row >= 0; row--) {
			flt frac = mat[row][size - 1 - pass] / mat[size - 1 - pass][size - 1 - pass];

			mat[row][size] -= frac * mat[size - 1 - pass][size];
		}

		mat[size - pass - 1][size] /= mat[size - 1 - pass][size - 1 - pass];
	}

}

static void check_solution(flt **orig, flt **res, int size)
{
	for (int row = 0; row < size; row++) {
		flt r = 0;

		for (int col = 0; col < size; col++) {
			r += res[col][size] * orig[row][col];
		}
		r -= orig[row][size];
		printf("root %d (%f) error: %f\n", row, res[row][size], r);
	}
}

int main(int argc, char **argv)
{
	flt **mat = NULL;
	flt **orig = NULL;
	struct timespec start, end;

	if (argc != 2)
		return 1;

	srand(time(NULL));
	allocate_matrix(&mat, SIZE);
	generate_matrix(mat, SIZE);

	if (SIZE < 20) {
		allocate_matrix(&orig, SIZE);
		copy_matrix(orig, mat, SIZE);

		printf("initial matrix\n");
		print_matrix(mat, SIZE);
	}

	clock_gettime(CLOCK_MONOTONIC, &start);
	switch (argv[1][0]) {
	case 's':
		gauss_single_thread(mat, SIZE);
		break;
	case 'b':
		gauss_pthread_barrier(mat, SIZE);
		break;
	case 'j':
		gauss_pthread_join(mat, SIZE);
		break;
	case 'o':
		gauss_omp(mat, SIZE);
		break;
	default:
		return 1;
	}
	clock_gettime(CLOCK_MONOTONIC, &end);

	if (SIZE < 20)
		check_solution(orig, mat, SIZE);

	printf("time: %f\n", time_diff(end, start));
}

