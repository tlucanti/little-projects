
CFLAGS = -Wall -Wextra -O3

CC = clang

all: build_simple build_cache build_block build_ultimate
.PHONY: all

clean:
	rm -f run_simple.out run_cache.out run_block.out
.PHONY: clean

re: clean all
.PHONY: re

build_simple:
	@echo "[CC]\tsimple"
	@$(CC) $(CFLAGS) -D MULT_TYPE=1 mmult.c -o mult_simple.out

build_cache:
	@echo "[CC]\tcache-friendly"
	@$(CC) $(CFLAGS) -D MULT_TYPE=2 mmult.c -o mult_cache.out

build_block:
	@echo "[CC]\tblock"
	@$(CC) $(CFLAGS) -D MULT_TYPE=3 mmult.c -o mult_block.out

build_ultimate:
	@echo "[CC]\tultimate"
	@$(CC) $(CFLAGS) -D MULT_TYPE=4 mmult.c -o mult_ultimate.out

run: build_simple build_cache build_block build_ultimate
	@echo -n "running simple multiplication:\t\t"
	@./mult_simple.out
	@echo -n "running cache-friendly multiplication:\t"
	@./mult_cache.out
	@echo -n "running block multiplication:\t\t"
	@./mult_block.out
	@echo -n "running ultimate multiplication:\t"
	@./mult_ultimate.out
