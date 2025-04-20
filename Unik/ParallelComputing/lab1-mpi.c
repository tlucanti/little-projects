
#define RUN_MPI
#include "common.h"

void call_mpi(int ret, char *message)
{
	if (ret) {
		printf("failed to run %s: error code %d\n", message, ret);
		abort();
	}
}

static void multiply_mpi(float **A, float **B, float **C, int size)
{
	int rank, num_procs;
	call_mpi(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "comm rank");
	call_mpi(MPI_Comm_size(MPI_COMM_WORLD, &num_procs), "comm size");

	// Calculate workload distribution
	int rows_per_proc = size / num_procs;
	int remainder = size % num_procs;

	// Adjust rows for this process to handle uneven division
	int local_rows = (rank < remainder) ? rows_per_proc + 1 : rows_per_proc;
	int start_row;

	if (rank < remainder) {
		start_row = rank * (rows_per_proc + 1);
	} else {
		start_row = remainder * (rows_per_proc + 1) +
			    (rank - remainder) * rows_per_proc;
	}

	int end_row = start_row + local_rows;

	// Perform matrix multiplication for assigned rows
	for (int i = start_row; i < end_row; i++) {
		for (int j = 0; j < size; j++) {
			C[i][j] = 0.0; // Initialize result cell to zero
			for (int k = 0; k < size; k++) {
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}

	// Gather results from all processes to process 0
	if (rank != 0) {
		// Send computed rows to process 0
		for (int i = start_row; i < end_row; i++) {
			call_mpi(MPI_Send(C[i], size, MPI_FLOAT, 0, i,
					  MPI_COMM_WORLD),
				 "send c[i]");
		}
	} else {
		// Process 0 receives results from other processes
		for (int p = 1; p < num_procs; p++) {
			int p_rows = (p < remainder) ? rows_per_proc + 1 :
						       rows_per_proc;
			int p_start_row;

			if (p < remainder) {
				p_start_row = p * (rows_per_proc + 1);
			} else {
				p_start_row = remainder * (rows_per_proc + 1) +
					      (p - remainder) * rows_per_proc;
			}

			int p_end_row = p_start_row + p_rows;

			for (int i = p_start_row; i < p_end_row; i++) {
				call_mpi(MPI_Recv(C[i], size, MPI_FLOAT, p, i,
						  MPI_COMM_WORLD,
						  MPI_STATUS_IGNORE),
					 "send c[i]");
			}
		}
	}
}

int main(int argc, char **argv)
{
	float **A, **B, **C;
	int rank, num_procs;

	// Initialize MPI environment
	call_mpi(MPI_Init(&argc, &argv), "mpi init");
	call_mpi(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "comm rank");
	call_mpi(MPI_Comm_size(MPI_COMM_WORLD, &num_procs), "comm size");

	// Allocate matrices on all processes
	alloc_matrix(&A, &B, &C, SIZE);

	// Only the root process initializes the matrices
	srand(123);
	if (rank == 0) {
		init_matrix(A, SIZE);
		init_matrix(B, SIZE);
		print_matrix(A, SIZE);
		print_matrix(B, SIZE);
		printf("==============================================\n");
	}

	// Broadcast matrices A and B to all processes
	for (int i = 0; i < SIZE; i++) {
		call_mpi(MPI_Bcast(A[i], SIZE, MPI_FLOAT, 0, MPI_COMM_WORLD),
			 "broadcast a[i]");
		call_mpi(MPI_Bcast(B[i], SIZE, MPI_FLOAT, 0, MPI_COMM_WORLD),
			 "broadcast b[i]");
	}

	// Perform parallel matrix multiplication
	multiply_mpi(A, B, C, SIZE);
	if (rank == 0) {
		print_matrix(C, SIZE);
	}

	// Free memory
	for (int i = 0; i < SIZE; i++) {
		free(A[i]);
		free(B[i]);
		free(C[i]);
	}
	free(A);
	free(B);
	free(C);

	// Finalize MPI environment
	call_mpi(MPI_Finalize(), "mpi finalize");
	return 0;
}
