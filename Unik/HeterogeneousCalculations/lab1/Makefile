
all: run_simple run_cache run_block
.PHONY: all

clean:
	rm -f run_simple.out run_cache.out run_block.out
.PHONY: clean

re: clean all
.PHONY: re

run_simple:
	gcc -Wall -Wextra -D MULT_TYPE=1 mmult.c -o mult_simple.out

run_cache:
	gcc -Wall -Wextra -D MULT_TYPE=2 mmult.c -o mult_cache.out

run_block:
	gcc -Wall -Wextra -D MULT_TYPE=3 mmult.c -o mult_block.out


