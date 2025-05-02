#include <stdio.h>
#include <mpi.h>

int main()
{
	// Initialize the MPI environment
	MPI_Init(NULL, NULL);

	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	printf("rank: %d\n", world_rank);

	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	printf("size: %d\n", world_size);

	int number;
	if (world_rank == 0) {
		number = -1;
		MPI_Send(&number, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
	} else if (world_rank == 1) {
		MPI_Recv(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,
			 MPI_STATUS_IGNORE);
		printf("Process 1 received number %d from process 0\n", number);
	}

	// Finalize the MPI environment.
	MPI_Finalize();
}
