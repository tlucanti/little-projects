
all:
	tar xfv csv.tar
	sed -i -e 's/-0$$/0/g' window_10.csv
	sed -i -e 's/-0,/0,/g' window_10.csv
	sed -i -e 's/-0$$/0/g' window_100.csv
	sed -i -e 's/-0,/0,/g' window_100.csv
	sed -i -e 's/-0$$/0/g' window_1000.csv
	sed -i -e 's/-0,/0,/g' window_1000.csv
	#
	g++ -O3 task.cpp -D WINDOW=10 -o task10.elf
	g++ -O3 task.cpp -D WINDOW=100 -o task100.elf
	g++ -O3 task.cpp -D WINDOW=1000 -o task1000.elf
	#
	./task10.elf < input.csv > w10.csv
	./task100.elf < input.csv > w100.csv
	./task1000.elf < input.csv > w1000.csv
