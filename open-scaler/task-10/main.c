
#include <unistd.h>
#include <stdint.h>
#include <stdio.h>
#include <arpa/inet.h>
#include <stdbool.h>
#include <stdlib.h>

int main()
{
	uint16_t net;
	int16_t host;
	int res;
	int pos = 0;
	int neg = 0;

	if (sizeof(uint16_t) != 2) {
		printf("short it not 2 bytes long\n");
	}

	while (true) {
		res = read(1, &net, sizeof(uint16_t));

		if (res == 0) {
			printf("eof\n");
		} else if (res == -1) {
			perror("read");
			return 1;
		}

		host = ntohs(net);

		if (host > 0) {
			++pos;
		} else {
			++neg;
		}

		if (res == 0) {
			break;
		}
	}

	printf("positive: %d\n", pos);
	printf("negative: %d\n", neg);
	printf("difference: %d\n", abs(pos - neg));
}
