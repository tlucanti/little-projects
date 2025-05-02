
#define RUN_MPI
#include <unistd.h>
#include "common.h"

/* helper function to find next closest number from @start that will give
 * reminder @rank by modulo @num_procs
 */
static inline int __fm_begin(int begin, int rank, int num_procs)
{
	while (begin % num_procs != rank) {
		begin++;
	}
	return begin;
	/*                                     num_procs
	 * Xprev           begin         X<------------------>Xnext
	 *
	 * Xprev, X, Xnext will give @rank in reminder when devided over
	 * @num_procs
	 *
	 * so we need to find next closest value that will give @rank when
	 * divided over @num_procs - that is value X
	 *
	 * to find X we need to find distance between @begin and X and add this
	 * value to begin
	 */

	if (begin % num_procs == 0) {
		return begin;
	}

	/* sice Xprev is equivalent to @rank by mudulo @num_procs, to get
	 * distance from Xprev and begin we can just subtruct them by modilo
	 * @num_procs
	 *
	 *          dist
	 * Xprev<--------->begin         X                    Xnext
	 */
	int dist = (begin - rank) % num_procs;

	/* now the distaance betweem @begin and X is just @num_procs - @dist
	 */
	int shift = num_procs - dist;

	/* if begin is already equals X: @dist will be equal to 0 and @shift
	 * will be equal to @num_procs, so we need to take modulo here again
	 * to cover this case
	 */
	return begin + shift;
}

/* for_mpi iterates through numbers from @begin to @end but only includes
 * numbers where: @var % @num_procs == @rank
 *
 * example of resulting sequence per rank:
 * with begin = 0, end = 14, num_procs = 4
 *
         iterations
 * rank \
 *  0     0  4  8  12
 *  1     1  5  9  13
 *  2     2  6  10
 *  3     3  7  11
 */

#define for_mpi(var, begin, end, rank, num_procs)                     \
	for (int var = __fm_begin(begin, rank, num_procs); var < end; \
	     var += num_procs)

static void gauss_mpi(flt **mat, int size)
{
	int rank, num_procs;

	call_mpi(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "comm rank");
	call_mpi(MPI_Comm_size(MPI_COMM_WORLD, &num_procs), "comm size");

	/* forward pass */
	for (int pass = 0; pass < size - 1; pass++) {
		/* we assign each process to set of rows,
		 * so processes can write only to this set,
		 * and read only this set and pivot row
		 *
		 * pivot row is the row that will be subtracted from other rows
		 */

		for_mpi(row, pass + 1, size, rank, num_procs) {
			flt frac = mat[row][pass] / mat[pass][pass];

			for (int col = pass + 1; col <= size; col++) {
				mat[row][col] -= frac * mat[pass][col];
			}
		}

		/* broadcast pivot row to other processes
		 *
		 * but we do not broadcast whole row, only part that will be
		 * used in substruction in following passes
		 */
		call_mpi(MPI_Bcast(mat[pass + 1] + pass + 1, size - pass,
				   FLOAT_TYPE_MPI, (pass + 1) % num_procs,
				   MPI_COMM_WORLD),
			 "broadcas pivot row");
	}

	/* backward pass */
	for (int pass = 0; pass < size; pass++) {
		/* broadcast free coefficient for following operation from
		 * process that owns this row
		 */
		call_mpi(MPI_Bcast(&mat[size - 1 - pass][size], 1, FLOAT_TYPE_MPI,
				   (size - 1 - pass) % num_procs,
				   MPI_COMM_WORLD),
			 "broadcas pivot row");

		for_mpi(row, 0, size - 1 - pass, rank, num_procs) {
			flt frac = mat[row][size - 1 - pass] /
				   mat[size - 1 - pass][size - 1 - pass];

			mat[row][size] -= frac * mat[size - 1 - pass][size];
		}

		if ((size - pass - 1) % num_procs == rank) {
			/* reduce free coefficient after subtructing it from
			 * other rows
			 */
			mat[size - pass - 1][size] /=
				mat[size - 1 - pass][size - 1 - pass];
		}
	}

	if (rank != 0) {
		/* send free coefficients back to root process */
		for_mpi(row, 0, size, rank, num_procs) {
			call_mpi(MPI_Send(&mat[row][size], 1, FLOAT_TYPE_MPI, 0, row,
					  MPI_COMM_WORLD),
				 "send");
		}
	} else {
		/* receive free coefficients in root process */
		for (int row = 0; row < size; row++) {
			if (row % num_procs == 0) {
				continue;
			}
			call_mpi(MPI_Recv(&mat[row][size], 1, FLOAT_TYPE_MPI,
					  row % num_procs, row, MPI_COMM_WORLD,
					  MPI_STATUS_IGNORE),
				 "recv");
		}
	}
}

int main(int argc, char **argv)
{
	flt **mat, **orig;
	struct timespec begin, end;
	int rank, num_procs;

	call_mpi(MPI_Init(&argc, &argv), "mpi init");
	call_mpi(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "comm rank");
	call_mpi(MPI_Comm_size(MPI_COMM_WORLD, &num_procs), "comm size");

	alloc_matrix_gauss(&mat, SIZE);

	if (rank == 0) {
		srand(123);
		init_matrix_gauss(mat, SIZE);

		if (SIZE < 20) {
			alloc_matrix_gauss(&orig, SIZE);
			copy_matrix_gauss(orig, mat, SIZE);
			printf("original matrix\n");
			print_matrix_gauss(orig, SIZE);
		}
	}

	clock_gettime(CLOCK_MONOTONIC, &begin);
	for (int i = 0; i < SIZE; i++) {
		call_mpi(MPI_Bcast(mat[i], SIZE + 1, FLOAT_TYPE_MPI, 0,
				   MPI_COMM_WORLD),
			 "broadcast mat");
	}
	gauss_mpi(mat, SIZE);
	clock_gettime(CLOCK_MONOTONIC, &end);

	if (rank == 0) {
		printf("time: %fs\n", time_diff(&begin, &end));

		if (SIZE < 20) {
			check_solution_gauss(orig, mat, SIZE);
		}
	}

	call_mpi(MPI_Finalize(), "mpi finalize");
	return 0;
}

