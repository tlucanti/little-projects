
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
	long long res;
	long long pos = 0;
	long long neg = 0;
	int i = 0;

	if (sizeof(uint16_t) != 2) {
		printf("short it not 2 bytes long\n");
	}

	while (true) {
		++i;
		res = read(0, &net, sizeof(uint16_t));

		if (i % 100000 == 0) {
			printf("read %d numbers\n", i);
		}
		if (res == 0) {
			printf("eof\n");
		} else if (res == -1) {
			perror("read");
			return 1;
		}

		host = (int16_t)ntohs(net);

		if (host > 0) {
			++pos;
		} else {
			++neg;
		}

		if (res == 0) {
			break;
		}
	}

	printf("positive: %lld\n", pos);
	printf("negative: %lld\n", neg);
	printf("difference: %lld\n", pos - neg);
}
