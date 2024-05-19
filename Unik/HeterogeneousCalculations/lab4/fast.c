
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define DIM 1000

typedef float flt;

union int_flt {
	int iv;
	flt fv;
};

static FILE *file;
static flt rows[DIM] = {};
static union int_flt cols[DIM] = {};

static double time_diff(struct timespec end, struct timespec start)
{
        double sec = end.tv_sec - start.tv_sec;
        sec += (end.tv_nsec -  start.tv_nsec) * 1e-9;
        return sec;
}

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

		if (fabs(val - vals[i]) > 1e-6) {
			printf("%s value error: (expected) %f != %f (real)\n",
			       fname, val, vals[i]);
			abort();
		}
	}
}

int main()
{
	struct timespec start, end;

	file = fopen("matrix.txt", "r");
	if (file == NULL) {
		printf("cannot open 'matrix.txt'\n");
		abort();
	}

	clock_gettime(CLOCK_MONOTONIC, &start);

	for (int row = 0; row < DIM; row++) {
		int row_sum = 0;
		for (int col = 0; col < DIM; col++) {
			int val;

			if (fscanf(file, "%d", &val) != 1) {
				abort();
			}

			cols[col].iv += val;
			row_sum += val;
		}

		rows[row] = (flt)row_sum / (flt)DIM;
	}

	for (int col = 0; col < DIM; col++) {
		cols[col].fv = (flt)cols[col].iv / (flt)DIM;
	}

	clock_gettime(CLOCK_MONOTONIC, &end);

	fclose(file);
	printf("time: %f\n", time_diff(end, start));

	check_answer("result_rows.txt", rows);
	check_answer("result_cols.txt", (flt *)cols);
}

