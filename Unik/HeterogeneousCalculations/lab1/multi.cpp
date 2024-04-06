
#ifndef BOARD_SIZE
# define BOARD_SIZE 8u
#endif

#include <stdio.h>
#include <chrono>
#include <thread>
#include <future>

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

static void nqueens_runner(int col, unsigned long *ans_ref)
{
        cols[col] = 1;
        main_diags[col] = 1;
        side_diags[BOARD_SIZE - 1 + col] = 1;

        nqueens_dfs(1);
        *ans_ref = ans;
}

unsigned long nqueens(void)
{
        unsigned long anses[BOARD_SIZE / 2 + BOARD_SIZE % 2] = {};
        std::thread threads[BOARD_SIZE / 2 + BOARD_SIZE % 2] = {};
        unsigned long final_ans = 0;

        for (unsigned long i = 0; i < BOARD_SIZE / 2 + BOARD_SIZE % 2; i++) {
                threads[i] = std::thread(nqueens_runner, i, &anses[i]);
        }

        for (unsigned long i = 0; i < BOARD_SIZE / 2; i++) {
                threads[i].join();
                final_ans += anses[i] * 2;
        }
        if (BOARD_SIZE % 2) {
                threads[BOARD_SIZE / 2].join();
                final_ans += anses[BOARD_SIZE / 2];
        }

        return final_ans;
}



int main()
{
        auto start = std::chrono::high_resolution_clock::now();
        unsigned long ans = nqueens();
        auto end = std::chrono::high_resolution_clock::now();

        double time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1e9;

        printf("ans: %lu, time: %f\n", ans, time);
}

