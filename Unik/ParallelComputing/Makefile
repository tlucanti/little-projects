
CC = clang
MPICC = mpicc
CFLAGS_MPI = -Wall -Wextra -O3 -ffast-math
CFLAGS = -Wall -Wextra -O3 -ffast-math -fopenmp

all: lab1 lab2 lab3 lab1-mpi lab2-mpi lab3-mpi

lab1:
	$(CC) $(CFLAGS) src/lab1.c -o mmult-compare.elf

lab2:
	$(CC) $(CFLAGS) src/lab2.c -o gauss-compare.elf

lab3:
	$(CC) $(CFLAGS) src/lab3.c -lm -o integral-compare.elf

lab1-mpi:
	$(MPICC) $(CFLAGS_MPI) src/lab1-mpi.c -o mmult-mpi.elf

lab2-mpi:
	$(MPICC) $(CFLAGS_MPI) src/lab2-mpi.c -o gauss-mpi.elf

lab3-mpi:
	$(MPICC) $(CFLAGS_MPI) src/lab3-mpi.c -lm -o integral-mpi.elf
