
#define RUN_MPI
#include "common.h"

static void gauss_mpi(flt **mat, int size)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Calculate row distribution
    int rows_per_proc = size / num_procs;
    int remainder = size % num_procs;

    // Calculate start and end rows for this process
    int start_row = rank * rows_per_proc + (rank < remainder ? rank : remainder);
    int end_row = start_row + rows_per_proc + (rank < remainder ? 1 : 0);

    // Temporary buffer for pivot row
    flt *pivot_row = malloc((size + 1) * sizeof(flt));

    // Forward pass
    for (int pass = 0; pass < size - 1; pass++) {
        // Determine which process has the current pivot row
        int pivot_owner = 0;
        while (pass >= start_row && pass < end_row) {
            // This process owns the pivot row

            // Send pivot row to all other processes
            for (int col = 0; col <= size; col++) {
                pivot_row[col] = mat[pass][col];
            }

            MPI_Bcast(pivot_row, size + 1, MPI_FLOAT, rank, MPI_COMM_WORLD);
            break;
        }

        // All processes wait for the pivot row
        MPI_Bcast(pivot_row, size + 1, MPI_FLOAT, pivot_owner, MPI_COMM_WORLD);

        // Each process processes its assigned rows
        for (int row = start_row; row < end_row; row++) {
            if (row > pass) {
                flt frac = mat[row][pass] / pivot_row[pass];

                for (int col = pass + 1; col <= size; col++) {
                    mat[row][col] -= frac * pivot_row[col];
                }
            }
        }

        // Wait for all processes to finish this pass
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Backward pass
    for (int pass = 0; pass < size; pass++) {
        int curr_row = size - 1 - pass;

        // Determine which process has the current row
        int row_owner = 0;
        while (curr_row >= start_row && curr_row < end_row) {
            // This process owns the current row

            // Normalize the row
            mat[curr_row][size] /= mat[curr_row][curr_row];

            // Create broadcast data
            pivot_row[0] = mat[curr_row][size];

            MPI_Bcast(pivot_row, 1, MPI_FLOAT, rank, MPI_COMM_WORLD);
            break;
        }

        // All processes wait for the result
        MPI_Bcast(pivot_row, 1, MPI_FLOAT, row_owner, MPI_COMM_WORLD);

        // If this process doesn't own the row, update the value
        if (curr_row >= start_row && curr_row < end_row && rank != row_owner) {
            mat[curr_row][size] = pivot_row[0];
        }

        // Each process updates its earlier rows
        for (int row = start_row; row < end_row && row < curr_row; row++) {
            flt frac = mat[row][curr_row] / mat[curr_row][curr_row];
            mat[row][size] -= frac * pivot_row[0];
        }

        // Wait for all processes to finish this pass
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Gather results back to rank 0
    for (int i = 0; i < size; i++) {
        // Determine which process has this row
        int row_owner = 0;
        for (int p = 0; p < num_procs; p++) {
            int proc_start = p * rows_per_proc + (p < remainder ? p : remainder);
            int proc_end = proc_start + rows_per_proc + (p < remainder ? 1 : 0);

            if (i >= proc_start && i < proc_end) {
                row_owner = p;
                break;
            }
        }

        if (rank == row_owner && rank != 0) {
            // Send result to rank 0
            MPI_Send(&mat[i][size], 1, MPI_FLOAT, 0, i, MPI_COMM_WORLD);
        } else if (rank == 0 && row_owner != 0) {
            // Receive result from the owner
            MPI_Recv(&mat[i][size], 1, MPI_FLOAT, row_owner, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    free(pivot_row);
}
