
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>

#define DIM 1000
#ifndef __always_inline
# define __always_inline inline __attribute__((__always_inline__))
#endif

typedef float flt;

union int_flt {
	unsigned iv;
	flt fv;
};

static FILE *file;
static flt rows[DIM] = {};
static union int_flt cols[DIM] = {};
static unsigned char text[2 * DIM * DIM];
static const unsigned char *str = text;

__always_inline
static double time_diff(struct timespec end, struct timespec start)
{
        double sec = end.tv_sec - start.tv_sec;
        sec += (end.tv_nsec -  start.tv_nsec) * 1e-9;
        return sec;
}

__always_inline
static void check_answer(const char *fname, flt *vals)
{
	file = fopen(fname, "r");
	if (file == NULL) {
		printf("cannot open %s\n", fname);
		abort();
	}

	for (int i = 0; i < DIM; i++) {
		float val;
		if (fscanf(file, "%f", &val) != 1) {
			printf("%s read error", fname);
			abort();
		}

		if (fabs(val - vals[i]) > 1e-5) {
			printf("%s value error: (expected) %f != %f (real)\n",
			       fname, val, vals[i]);
			abort();
		}
	}
}

int main()
{
	struct timespec start, end;
	int fd;

	fd = open("matrix.txt", O_RDONLY);
	if (fd == -1) {
		printf("open matrix.txt error\n");
		abort();
	}

	clock_gettime(CLOCK_MONOTONIC, &start);

	int rd = 0;
	rd = read(fd, text, 2 * DIM * DIM - 1);
	if (rd != 2 * DIM * DIM - 1) {
		abort();
	}

	#pragma omp parallel for
	for (unsigned row = 0; row < DIM; row++) {
		unsigned row_sum = 0;
		for (unsigned col = 0; col < DIM; col++) {
			unsigned val;

			val = *str; // sub all '0' at the end at once
			str += 2;

			cols[col].iv += val;
			row_sum += val;
		}

		rows[row] = (flt)(row_sum - '0' * DIM) / (flt)DIM;
	}

	for (int col = 0; col < DIM; col++) {
		cols[col].fv = (flt)(cols[col].iv - '0' * DIM) / (flt)DIM;
	}

	clock_gettime(CLOCK_MONOTONIC, &end);
	printf("time: %f\n", time_diff(end, start));

	check_answer("result_rows.txt", rows);
	check_answer("result_cols.txt", (flt *)cols);
}

