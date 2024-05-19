
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TASK_SIZE 1000
#define THREAD_NUM 8
#define ROW_SIZE (TASK_SIZE + 1)
#define TASK_LEN (TASK_SIZE * TASK_SIZE)

#define ANSI_RED "\033[01;31m"
#define ANSI_RESET "\033[0m"

#ifndef __unused
# define __unused __attribute__((__unused__))
#endif
#ifndef __always_inline
# define __always_inline inline __attribute__((__always_inline__))
#endif

typedef double flt;

__always_inline
static flt random_float(void)
{
	flt r = (flt)random() / (flt)RAND_MAX;
	return r * (flt)2 - (flt)1;
}

__always_inline
static double time_diff(struct timespec end, struct timespec start)
{
	double sec = end.tv_sec - start.tv_sec;
	sec += (end.tv_nsec -  start.tv_nsec) * 1e-9;
	return sec;
}

__always_inline
static flt absf(flt val)
{
	if (val < 0) {
		return -val;
	}
	return val;
}

/**
 * matrix shape:
 *
 * [0, 1, ..., TASK_SIZE - 1, TASK_SIZE, TASK_SIZE + 1 ... 2 * TASK_SIZE, 2*TASK_SIZE + 1 ...
 * <----------------------->  <------->  <------------------------------> <------------->
 *      first line                b0          second line                     b1
 */
__always_inline
static void generate_matrix(flt *mat)
{
	const int len = TASK_LEN + TASK_SIZE;

	for (int i = 0; i < len; i++) {
		mat[i] = random_float();
	}
}

__unused
__always_inline
static void copy_matrix(flt *dst, flt *src)
{
	memcpy(dst, src, sizeof(flt) * (TASK_LEN + TASK_SIZE));
}

__unused
__always_inline
static void print_matrix(flt *mat)
{
	static flt prev[TASK_LEN + TASK_SIZE] = {};

	int i = 0;
	for (int row = 0; row < TASK_SIZE; row++) {
		int col;
		for (col = 0; col < TASK_SIZE; col++) {
			if (absf(prev[i] - mat[i]) > 1e-5) {
				printf(ANSI_RED);
			}
			printf("%7.3f ", mat[i]);
			prev[i] = mat[i];
			printf(ANSI_RESET);
			i++;
		}
		if (absf(prev[i] - mat[i]) > 1e-5) {
			printf(ANSI_RED);
		}
		printf(" | %7.3f\n", mat[i]);
		prev[i] = mat[i];
		printf(ANSI_RESET);
		i++;
	}
	printf("\n");
}

static flt *cpy;
static flt *mat;
static flt ret[TASK_SIZE];
static pthread_barrier_t barrier;

__always_inline
static void *solver_runner(void *thread_id)
{
	const unsigned tid = (unsigned long)thread_id;
	float frac;

	for (unsigned passage = 0; passage < TASK_SIZE; passage++) {

		for (unsigned row = passage + 1; row < TASK_SIZE; row++) {
			// printf("frac: %7.3f / %7.3f\n", mat[row * ROW_SIZE + passage], mat[passage * ROW_SIZE + passage]);
			frac = mat[row * ROW_SIZE + passage] / mat[passage * ROW_SIZE + passage];

			for (unsigned col = passage + 1 + tid; col < TASK_SIZE; col += THREAD_NUM) {
				if (col < passage + 1) {
					continue;
				}
				mat[ROW_SIZE * row + col] -= frac * mat[ROW_SIZE * passage + col];
			}
			pthread_barrier_wait(&barrier);
			if (tid == 0) {
				mat[ROW_SIZE * row + ROW_SIZE - 1] -= frac * mat[ROW_SIZE * passage + ROW_SIZE - 1];
			}

		}
		// print_matrix(mat);

	}

	for (int passage = TASK_SIZE - 1 - tid; passage >= 0; passage -= THREAD_NUM) {

		for (int row = passage - 1; row >= 0; row--) {
			frac = mat[row * ROW_SIZE + passage] / mat[ROW_SIZE * passage + passage];
			// mat[row * ROW_SIZE + passage] -= frac * mat[passage * ROW_SIZE + passage];
			// mat[row * ROW_SIZE + passage] = 0;
			mat[row * ROW_SIZE + TASK_SIZE] -= frac * mat[passage * ROW_SIZE + TASK_SIZE];
			// mat[(row - 1) * ROW_SIZE + TASK_SIZE - 1] = 0;
		}

		ret[passage] = mat[passage * ROW_SIZE + TASK_SIZE] / mat[passage * ROW_SIZE + passage];
		// print_matrix(mat);
	}

	return NULL;
}

void solve(void)
{
	pthread_t threads[THREAD_NUM];

	pthread_barrier_init(&barrier, NULL, THREAD_NUM);
	for (unsigned long i = 0; i < THREAD_NUM; i++) {
		pthread_create(&threads[i], NULL, solver_runner, (void *)i);
	}

	for (unsigned i = 0; i < THREAD_NUM; i++) {
		pthread_join(threads[0], NULL);
	}
	pthread_barrier_destroy(&barrier);
}

__unused
__always_inline
static void check_solution(flt *mat, flt *ret)
{
	for (int row = 0; row < TASK_SIZE; row++) {
		flt r = 0;

		for (int col = 0; col < TASK_SIZE; col++) {
			r += ret[col] * mat[row * ROW_SIZE + col];
		}
		r -= mat[row * ROW_SIZE + ROW_SIZE - 1];
		printf("root %d (%f) error: %f\n", row, ret[row], r);
	}
}

int main()
{
	struct timespec start, end;

	mat = malloc(sizeof(flt) * (TASK_LEN + TASK_SIZE));
	cpy = malloc(sizeof(flt) * (TASK_LEN + TASK_SIZE));

	generate_matrix(mat);
	copy_matrix(cpy, mat);

	// print_matrix(mat);
	clock_gettime(CLOCK_MONOTONIC, &start);
	solve();
	clock_gettime(CLOCK_MONOTONIC, &end);

	check_solution(cpy, ret);

	printf("time: %f\n", time_diff(end, start));
}

