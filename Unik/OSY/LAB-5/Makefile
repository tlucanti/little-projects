obj-m += driver.o
KDIR = /lib/modules/$(shell uname -r)/build
SRC = programm
all:
	cc $(SRC).c -o $(SRC)
	make -C $(KDIR) M=$(shell pwd) modules
clean:
	make -C $(KDIR) M=$(shell pwd) clean

fclean: clean
	rm -rf $(SRC)

re: fclean all

start:	all
	sudo insmod driver.ko
	sudo dmesg | tail -n 1

stop: 
	sudo rmmod driver.ko
	sudo dmesg | tail -n 1

