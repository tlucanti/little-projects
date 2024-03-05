
CFLAGS = -Wall -Wextra -O3

CC = clang

all: build_simple build_split build_multisplit

build_simple:
	@echo "[CC]\tsimple"
	@$(CC) $(CFLAGS) aos_soa.c -D TYPE=1 -o aos_simple.elf -lm

build_split:
	@echo "[CC]\tarray split"
	@$(CC) $(CFLAGS) aos_soa.c -D TYPE=2 -o aos_split.elf -lm

build_multisplit:
	@echo "[CC]\tarray multisplit"
	@$(CC) $(CFLAGS) aos_soa.c -D TYPE=3 -o aos_multisplit.elf -lm

run: all
	@echo -n "running simple comparison:\t\t"
	@./aos_simple.elf
	@echo -n "running split comparison:\t\t"
	@./aos_split.elf
	@echo -n "running multisplit comparison:\t\t"
	@./aos_multisplit.elf


