
#define RUN_MPI
#include "common.h"

static void multiply_mpi(float **A, float **B, float **C, int size)
{
	int rank, num_procs;
	call_mpi(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "comm rank");
	call_mpi(MPI_Comm_size(MPI_COMM_WORLD, &num_procs), "comm size");

	for (int i = rank; i < size; i += num_procs) {
		for (int k = 0; k < size; k++) {
			for (int j = 0; j < size; j++) {
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}

	if (rank != 0) {
		for (int i = rank; i < SIZE; i += num_procs) {
			call_mpi(MPI_Send(C[i], size, MPI_FLOAT, 0, i, MPI_COMM_WORLD),
				 "send");
		}
	} else {
		for (int p = 1; p < num_procs; p++) {
			for (int i = p; i < SIZE; i += num_procs) {
				call_mpi(MPI_Recv(C[i], size, MPI_FLOAT, p, i,
						  MPI_COMM_WORLD, MPI_STATUS_IGNORE),
					 "recv");
			}
		}
	}
}

int main(int argc, char **argv)
{
	float **A, **B, **C;
	struct timespec begin, end;
	int rank, num_procs;

	call_mpi(MPI_Init(&argc, &argv), "mpi init");
	call_mpi(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "comm rank");
	call_mpi(MPI_Comm_size(MPI_COMM_WORLD, &num_procs), "comm size");

	alloc_matrix3(&A, &B, &C, SIZE);
	zero_matrix(C, SIZE);

	if (rank == 0) {
		srand(123);
		init_matrix(A, SIZE);
		init_matrix(B, SIZE);
	}

	clock_gettime(CLOCK_MONOTONIC, &begin);
	for (int i = 0; i < SIZE; i++) {
		call_mpi(MPI_Bcast(A[i], SIZE, MPI_FLOAT, 0, MPI_COMM_WORLD),
			 "broadcast a");
		call_mpi(MPI_Bcast(B[i], SIZE, MPI_FLOAT, 0, MPI_COMM_WORLD),
			 "broadcast b");
	}
	multiply_mpi(A, B, C, SIZE);
	clock_gettime(CLOCK_MONOTONIC, &end);

	if (rank == 0 && SIZE <= 20) {
		print_matrix(C, SIZE);
	}

	call_mpi(MPI_Finalize(), "mpi finalize");

	if (rank == 0) {
		printf("%fs\n", time_diff(&begin, &end));
	}

	return 0;
}
