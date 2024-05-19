
CXX = clang++
CC = clang

CFLAGS =
CFLAGS += -Wall -Wextra
CFLAGS += -fdiagnostics-color=always
CFLAGS += -O3 -ffast-math
# CFLAGS += -O0 -g3
CFLAGS += -D THREAD_NUM=`nproc --all`
CXXFLAGS += $(CFLAGS) -std=c++17

all: slow fast

slow:
	$(CXX) $(CXXFLAGS) slow.cpp -o slow.elf

fast:
	$(CC) $(CFLAGS) fast.c -o fast.elf
	$(CC) $(CFLAGS) single.c -o single.elf

