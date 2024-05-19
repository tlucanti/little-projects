
CXXFLAGS =
CFLAGS =

CFLAGS += -Wall -Wextra
CFLAGS += -fdiagnostics-color=always
# CFLAGS += -O0 -g3
CFLAGS += -O3 -ffast-math

CXXFLAGS += $(CFLAGS) -std=c++17

CXX = g++
CC = gcc

all: slow fast
.PHONY: all

slow:
	$(CXX) $(CXXFLAGS) slow.cpp -o slow.elf

fast:
	$(CC) $(CFLAGS) single.c -o single.elf
	$(CC) $(CFLAGS) fast.c -o fast.elf

