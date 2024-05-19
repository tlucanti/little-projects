
CXX = clang++
CC = clang

CFLAGS =
CFLAGS += -Wall -Wextra
CFLAGS += -fdiagnostics-color=always
CFLAGS += -O3 -ffast-math -fopenmp
CXXFLAGS += $(CFLAGS) -std=c++17

all: slow fast

slow:
	$(CXX) $(CXXFLAGS) slow.cpp -o slow.elf

fast:
	$(CC) $(CFLAGS) fast.c -o fast.elf

