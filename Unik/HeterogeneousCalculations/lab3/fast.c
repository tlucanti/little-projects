
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TASK_SIZE 3
#define ROW_SIZE (TASK_SIZE + 1)
#define TASK_LEN (TASK_SIZE * TASK_SIZE)

#define ANSI_RED "\033[01;31m"
#define ANSI_RESET "\033[0m"

typedef float flt;

flt random_float(void)
{
	flt r = (flt)random() / (flt)RAND_MAX;
	return r * (flt)2 - (flt)1;
}

double time_diff(struct timespec end, struct timespec start)
{
	double sec = end.tv_sec - start.tv_sec;
	sec += (end.tv_nsec -  start.tv_nsec) * 1e-9;
	return sec;
}

flt absf(flt val)
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
void generate_matrix(flt *mat)
{
	const int len = TASK_LEN + TASK_SIZE;

	for (int i = 0; i < len; i++) {
		mat[i] = random_float();
	}
}

void copy_matrix(flt *dst, flt *src)
{
	memcpy(dst, src, sizeof(flt) * (TASK_LEN + TASK_SIZE));
}

void print_matrix(flt *mat)
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

void solve(flt *mat, flt *ret)
{
	float frac;

	for (int passage = 0; passage < TASK_SIZE; passage++) {

		for (int row = passage + 1; row < TASK_SIZE; row++) {
			// printf("frac: %7.3f / %7.3f\n", mat[row * ROW_SIZE + passage], mat[passage * ROW_SIZE + passage]);
			frac = mat[row * ROW_SIZE + passage] / mat[passage * ROW_SIZE + passage];
			mat[ROW_SIZE * row + passage] = 0;
			for (int col = passage + 1; col < ROW_SIZE; col++) {
				mat[ROW_SIZE * row + col] -= frac * mat[ROW_SIZE * passage + col];
			}

		}
		print_matrix(mat);

	}
	printf("=====================================================================================\n");

	for (int passage = TASK_SIZE - 1; passage >= 0; passage--) {

		for (int row = passage - 1; row >= 0; row--) {

			printf("frac: %7.3f / %7.3f\n", mat[row * ROW_SIZE + passage], mat[ROW_SIZE * passage + passage]);
			frac = mat[row * ROW_SIZE + passage] / mat[ROW_SIZE * passage + passage];
			mat[row * ROW_SIZE + passage] -= frac * mat[passage * ROW_SIZE + passage];
			mat[row * ROW_SIZE + TASK_SIZE] -= frac * mat[passage * ROW_SIZE + TASK_SIZE];
			// mat[(row - 1) * ROW_SIZE + TASK_SIZE - 1] = 0;
		}

		ret[passage] = mat[passage * ROW_SIZE + TASK_SIZE] / mat[passage * ROW_SIZE + passage];
		mat[passage * ROW_SIZE + passage] = 1;
		mat[passage * ROW_SIZE + TASK_SIZE] = ret[passage];
		print_matrix(mat);
	}
}

void check_solution(flt *mat, flt *ret)
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
	flt cpy[TASK_LEN + TASK_SIZE];
	flt mat[TASK_LEN + TASK_SIZE];
	flt ret[TASK_SIZE];
	struct timespec start, end;


	generate_matrix(mat);
	copy_matrix(cpy, mat);

	clock_gettime(CLOCK_MONOTONIC, &start);
	print_matrix(mat);
	solve(mat, ret);
	clock_gettime(CLOCK_MONOTONIC, &end);

	check_solution(cpy, ret);

	printf("time: %f\n", time_diff(end, start));
}

