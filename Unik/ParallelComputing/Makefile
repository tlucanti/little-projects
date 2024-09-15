
CC = clang
CFLAGS = -Wall -Wextra -O3 -ffast-math -fopenmp

all: lab1 lab2 lab3

lab1:
	$(CC) $(CFLAGS) lab1.c -o mmult-compare.elf

lab2:
	$(CC) $(CFLAGS) lab2.c -o gauss-compare.elf

lab3:
	$(CC) $(CFLAGS) -lm lab3.c -o integral-compare.elf

