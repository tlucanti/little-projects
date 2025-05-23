#include <stdio.h>
#include <mpi.h>

#define PING_PONG_LIMIT 100

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

	int ping_pong_count = 0;
	int partner_rank = (world_rank + 1) % 2;
	while (ping_pong_count < PING_PONG_LIMIT) {
		if (world_rank == ping_pong_count % 2) {
			// Increment the ping pong count before you send it
			ping_pong_count++;
			MPI_Send(&ping_pong_count, 1, MPI_INT, partner_rank, 0,
				 MPI_COMM_WORLD);
			printf("%d sent and incremented ping_pong_count "
			       "%d to %d\n",
			       world_rank, ping_pong_count, partner_rank);
		} else {
			MPI_Recv(&ping_pong_count, 1, MPI_INT, partner_rank, 0,
				 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			printf("%d received ping_pong_count %d from %d\n",
			       world_rank, ping_pong_count, partner_rank);
		}
	}
	// Finalize the MPI environment.
	MPI_Finalize();
}
