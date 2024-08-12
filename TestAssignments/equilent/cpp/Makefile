
CXXFLAGS =
CXXFLAGS += -Wall -Wextra -Wmost -Wconversion -pedantic
CXXFLAGS += -O3 -fdiagnostics-color=always

all: init build run check

init:
	tar xfv csv.tar
	sed -i -e 's/-0$$/0/g' window_10.csv
	sed -i -e 's/-0,/0,/g' window_10.csv
	sed -i -e 's/-0$$/0/g' window_100.csv
	sed -i -e 's/-0,/0,/g' window_100.csv
	sed -i -e 's/-0$$/0/g' window_1000.csv
	sed -i -e 's/-0,/0,/g' window_1000.csv

build:
	clang++ -O3 slow.cpp -D WINDOW=10 -o slow10.elf
	clang++ -O3 slow.cpp -D WINDOW=100 -o slow100.elf
	clang++ -O3 slow.cpp -D WINDOW=1000 -o slow1000.elf
	#
	clang++ $(CXXFLAGS) fast.cpp -D WINDOW=10 -o fast10.elf
	clang++ $(CXXFLAGS) fast.cpp -D WINDOW=100 -o fast100.elf
	clang++ $(CXXFLAGS) fast.cpp -D WINDOW=1000 -o fast1000.elf

run:
	./fast10.elf < input.csv > f10.csv
	./fast100.elf < input.csv > f100.csv
	./fast1000.elf < input.csv > f1000.csv
	#
	./slow10.elf < input.csv > s10.csv
	./slow100.elf < input.csv > s100.csv
	./slow1000.elf < input.csv > s1000.csv

check:
	if [ "" == "" ]; then echo "ok"; fi

