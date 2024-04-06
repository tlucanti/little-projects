
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

static thread_local unsigned char cols[BOARD_SIZE] = {};
static thread_local unsigned char main_diags[round(BOARD_SIZE * 2)] = {};
static thread_local unsigned char side_diags[round(BOARD_SIZE * 2)] = {};
static thread_local unsigned long ans = 0;

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

static void *nqueens_runner(void *colp)
{
        unsigned long col = (unsigned long)colp;

        cols[col] = 1;
        main_diags[col] = 1;
        side_diags[BOARD_SIZE - 1 + col] = 1;

        nqueens_dfs(1);
        return (void *)ans;
}

unsigned long nqueens(void)
{
        pthread_t threads[BOARD_SIZE / 2 + BOARD_SIZE % 2];
        unsigned long final_ans = 0;

        for (unsigned long i = 0; i < BOARD_SIZE / 2 + BOARD_SIZE % 2; i++) {
                pthread_create(&threads[i], NULL, nqueens_runner, (void *)i);
        }

        for (unsigned long i = 0; i < BOARD_SIZE / 2; i++) {
                unsigned long ans;
                pthread_join(threads[i], (void **)&ans);
                final_ans += ans * 2;
        }
        if (BOARD_SIZE % 2) {
                unsigned long ans;
                pthread_join(threads[BOARD_SIZE / 2], (void **)&ans);
                final_ans += ans;
        }

        return final_ans;
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

