
#ifndef BOARD_SIZE
# define BOARD_SIZE 8u
#endif

#include <stdio.h>
#include <time.h>
#include <pthread.h>

#ifndef thread_local
#define thread_local _Thread_local
#endif

#define round(x) ((x) + (8 - (x) % 8))

static unsigned char cols[BOARD_SIZE] = {};
static unsigned char main_diags[round(BOARD_SIZE * 2)] = {};
static unsigned char side_diags[round(BOARD_SIZE * 2)] = {};
static unsigned long ans = 0;

static void nqueens_dfs(unsigned char row)
{
        for (unsigned char col = 0; col < BOARD_SIZE; col++) {
                unsigned char main, side;

                if (cols[col]) {
                        continue;
                }

                main = col + row;
                if (main_diags[main]) {
                        continue;
                }

                side = BOARD_SIZE - 1 + col - row;
                if (side_diags[side]) {
                        continue;
                }

                if (row == BOARD_SIZE - 1) {
                        ans++;
                        continue;
                }

                cols[col] = 1;
                main_diags[main] = 1;
                side_diags[side] = 1;
                nqueens_dfs(row + 1);
                cols[col] = 0;
                main_diags[main] = 0;
                side_diags[side] = 0;
        }
}

unsigned long nqueens(void)
{
        nqueens_dfs(0);
        return ans;
}

int main()
{
        struct timespec start, end;

        clock_gettime(CLOCK_MONOTONIC, &start);
        unsigned long ans = nqueens();
        clock_gettime(CLOCK_MONOTONIC, &end);

        double time = (double)(end.tv_sec - start.tv_sec) +
                      (double)(end.tv_nsec - start.tv_nsec) * 1e-9;
        printf("ans: %lu, time: %f\n", ans, time);
}

