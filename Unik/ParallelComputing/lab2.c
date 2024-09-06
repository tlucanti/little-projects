
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define SIZE 10

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

static void gauss_passage(flt **mat, int pass, int size)
{
}

static void gauss_single_thread(flt **mat, int size)
{
	// forward pass
	for (int pass = 0; pass < size; pass++) {
		for (int row = pass + 1; row < size; row++) {
			float frac = mat[row][pass] / mat[pass][pass];

			for (int col = pass + 1; col <= size; col++) {
				mat[row][col] -= frac * mat[pass][col];
			}
		}

		// printf("\nforward pass %d:\n", pass);
		// print_matrix(mat, size);
	}
	printf("\nforward done:\n");
	print_matrix(mat, size);

	// backward pass
	for (int pass = 0; pass < size; pass++) {
		float frac;

		for (int row = size - 2 - pass; row >= 0; row--) {
			frac = mat[row][size - 1 - pass] / mat[size - 1 - pass][size - 1 - pass];

			mat[row][size] -= frac * mat[size - 1 - pass][size];
		}

		mat[size - pass - 1][size] /= mat[size - 1 - pass][size - 1 - pass];

		// printf("\nbackward pass %d:\n", pass);
		// print_matrix(mat, size);
	}

	printf("\nbackward done:\n");
	print_matrix(mat, size);
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

int main()
{
	flt **mat = NULL;
	flt **orig = NULL;
	struct timespec start, end;

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
	gauss_single_thread(mat, SIZE);
	clock_gettime(CLOCK_MONOTONIC, &end);

	check_solution(orig, mat, SIZE);

	printf("time: %f\n", time_diff(end, start));
}

