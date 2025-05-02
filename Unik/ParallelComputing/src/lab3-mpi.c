
#define RUN_MPI
#define FLOAT_TYPE double
#include "common.h"

static flt function(flt x)
{
	return (100 * x * logf(x + 1)) / (x + 100 * square(cosf(0.1 * x)));
}

static flt integral_mpi(flt a, flt b, flt h)
{
	int rank, size;
	call_mpi(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "rank");
	call_mpi(MPI_Comm_size(MPI_COMM_WORLD, &size), "size");

	int n = (b - a) / h;
	flt local_sum = 0.0;

	for (int i = rank; i < n; i += size) {
		local_sum += function(a + i * h);
	}

	flt global_sum = 0.0;
	call_mpi(MPI_Reduce(&local_sum, &global_sum, 1, FLOAT_TYPE_MPI, MPI_SUM,
			    0, MPI_COMM_WORLD),
		 "reduce");

	if (rank == 0) {
		global_sum -= (function(a) + function(b)) * 0.5f;
		global_sum *= h;
	}

	return global_sum;
}

int main(int argc, char **argv)
{
	struct timespec begin, end;
	int rank, num_procs;
	flt ans;

	call_mpi(MPI_Init(&argc, &argv), "mpi init");
	call_mpi(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "comm rank");
	call_mpi(MPI_Comm_size(MPI_COMM_WORLD, &num_procs), "comm size");

	clock_gettime(CLOCK_MONOTONIC, &begin);
	ans = integral_mpi(INT_START, INT_END, INT_H);
	clock_gettime(CLOCK_MONOTONIC, &end);

	if (rank == 0) {
		double err = fabs(1 - INT_TRUE_ANSWER / ans) * 100;
		printf("answer: %f (relative error: %f%%)\n", ans, err);
		printf("time: %fs\n", time_diff(&begin, &end));
	}

	call_mpi(MPI_Finalize(), "mpi finalize");
	return 0;
}
