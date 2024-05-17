
CXXFLAGS =

CXXFLAGS += -Wall -Wextra
CXXFLAGS += -std=c++17
CXXFLAGS += -O0 -g3
CXXFLAGS += -fdiagnostics-color=always

CC = g++

all: slow fast

slow:
	$(CC) $(CXXFLAGS) slow.cpp -o slow

fast:
	$(CC) $(CXXFLAGS) fast.cpp -o fast

