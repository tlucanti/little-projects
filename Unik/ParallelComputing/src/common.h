
#include <time.h>
#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>

#ifdef RUN_MPI
#include <mpi.h>
#endif

#ifndef SIZE
# define SIZE 4096
#endif

#ifndef NR_THREADS
# define NR_THREADS 8
#endif

typedef float flt;

static inline void *call_malloc(unsigned long size)
{
	void *ret = malloc(size);

	if (ret == NULL) {
		printf("malloc error");
		abort();
	}
	return ret;
}

#ifdef RUN_MPI
void call_mpi(int ret, char *message)
{
	if (ret) {
		printf("failed to run %s: error code %d\n", message, ret);
		abort();
	}
}
#endif

static inline void init_matrix(float **array, int size)
{
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			array[i][j] = (float)rand() / (float)RAND_MAX * 10;
		}
	}
}

static inline void zero_matrix(float **array, int size)
{
	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < size; ++j) {
			array[i][j] = 0;
		}
	}
}

static inline void copy_matrix(flt **dst, flt **src, int size)
{
	for (int y = 0; y < size; y++) {
		for (int x = 0; x <= size; x++) {
			dst[y][x] = src[y][x];
		}
	}
}

static inline void print_matrix(float **array, int size)
{
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			printf("\t%f", array[i][j]);
		}
		printf("\n");
	}
}

static inline float time_diff(struct timespec *begin, struct timespec *end)
{
	float res = 0;

	res += end->tv_sec - begin->tv_sec;
	res += (end->tv_nsec - begin->tv_nsec) * 1e-9f;
	return res;
}

static inline void alloc_matrix(float ***a, int size)
{
	*a = call_malloc(sizeof(float *) * size);
	for (int i = 0; i < size; ++i) {
		(*a)[i] = call_malloc(sizeof(float) * size);
	}
}

static inline void alloc_matrix3(float ***a, float ***b, float ***c, int size)
{
	alloc_matrix(a, size);
	alloc_matrix(b, size);
	alloc_matrix(c, size);
}

