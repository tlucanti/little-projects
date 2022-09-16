#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
char write_buf[1024];
char read_buf[1024];
int main()
{
	int fd, var;
	fd = open("/dev/MyLab5Module", O_RDWR);
	if(fd < 0) {
		printf("Cannot open device file...\n");
		return 0;
	}
	while(1) {
		printf("Please Enter the Option\n");
		printf("\n");
		printf(" 1. Write \n");
		printf(" 2. Read \n");
		printf(" 3. Exit \n");
		printf("\n");
		printf("Your Option = \n");
		scanf("%d", &var);
		switch(var) {
			case 1:
				printf("Enter the string to write into driver :");
				scanf("%s", write_buf);
				printf("Data Writing ...");
				write(fd, write_buf, strlen(write_buf)+1);
				printf("Done!\n");
				break;
			case 2:
				printf("Data Reading ...");
				read(fd, read_buf, 1024);
				printf("Done!\n\n");
				printf("Data = %s\n\n", read_buf);
				break;
			case 3:
				close(fd);
				exit(1);
				break;
			default:
				printf("Enter Valid option = %d\n",var);
				break;
		}

	}
	close(fd);
}
