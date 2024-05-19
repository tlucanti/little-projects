
CXXFLAGS =
CFLAGS =

CFLAGS += -Wall -Wextra
CFLAGS += -fdiagnostics-color=always
CFLAGS += -O0 -g3

CXXFLAGS += $(CFLAGS) -std=c++17

CXX = g++
CC = gcc

all: slow fast
.PHONY: all

slow:
	$(CXX) $(CXXFLAGS) slow.cpp -o slow.elf

fast:
	$(CC) $(CFLAGS) fast.c -o fast.elf

