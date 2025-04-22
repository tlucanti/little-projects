
CC = clang
MPICC = mpicc
CFLAGS_MPI = -Wall -Wextra -O3 -ffast-math
CFLAGS = -Wall -Wextra -O3 -ffast-math -fopenmp

all: lab1 lab2 lab3 lab1-mpi

lab1:
	$(CC) $(CFLAGS) lab1.c -o mmult-compare.elf

lab2:
	$(CC) $(CFLAGS) lab2.c -o gauss-compare.elf

lab3:
	$(CC) $(CFLAGS) lab3.c -lm -o integral-compare.elf

lab1-mpi:
	$(MPICC) $(CFLAGS_MPI) lab1-mpi.c -o mmult-mpi.elf
