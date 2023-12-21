
LAB1_OBJ =	build/Impl.o	\
		build/DNF.o	\
		build/lab1.o

LAB2_OBJ =	build/Impl.o		\
		build/DNF.o		\
		build/TableItem.o	\
		build/Table.o		\
		build/Automat.o		\
		build/lab2.o

CXX_FLAGS =	-fdiagnostics-color=always	\
		-Wall -Wextra -Werror		\
		-O0 -g3				\
		-std=c++20			\
		-I include

CC = 	g++
LD =	$(CC)

all: run-lab2

build/%.o: src/%.cpp build Makefile
	$(CC) $(CXX_FLAGS) -c $< -o $@

build/%.o: %.cpp build Makefile
	$(CC) $(CXX_FLAGS) -c $< -o $@

build:
	mkdir -p build

build-lab1: $(LAB1_OBJ)
	$(LD) $(LDFLAGS) $(LAB1_OBJ) -o lab1

build-lab2: $(LAB2_OBJ)
	$(LD) $(LDFLAGS) $(LAB2_OBJ) -o lab2

run-lab1: build-lab1
	./lab1 1111101-11-01000

run-lab2: build-lab2
	./lab2 transition.txt output.txt

clean:
	rm -rf build

.PHONY: build-lab1 build-lab2 run-lab1 run-lab2

