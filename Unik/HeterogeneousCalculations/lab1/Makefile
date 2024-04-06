
CFLAGS = -Wall -Wextra -O3
CXX = g++
CC = clang

all:
	@ for i in $$(seq 4 15); do \
		echo "CC\tmy_$$i.elf"; \
		$(CC) $(CFLAGS) multi.c -D BOARD_SIZE=$$i -o multi_$$i.elf; \
		echo "CC\tmulti_$$i.elf"; \
		$(CC) $(CFLAGS) my.c -D BOARD_SIZE=$$i -o my_$$i.elf; \
	done

	@echo "CC\tintel.elf"
	@$(CXX) $(CFLAGS) intel.cpp -o intel.elf

clean:
	rm -f *elf
	#$(CC) $(CFLAGS) my.cpp -o my

