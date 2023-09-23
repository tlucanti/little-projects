
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>

int main()
{
	const char *path = "/dev/random";
	int dev;
	int ret;
	unsigned char c;
	unsigned long long cnt[256];
	int ans = 0;

	bzero(cnt, sizeof(cnt));

	dev = open(path, O_RDONLY /* | O_DIRECT */);
	if (dev == -1) {
		perror("open");
		return 1;
	}

	for (unsigned long long i = 0; i < 10000000; ++i) {
		ret = read(dev, &c, 1);

		if (i % 100000 == 0) {
			printf("read %llu bytes\n", i);
		}

		if (ret == 0) {
			printf("eof\n");
			break;
		} else if (ret == -1) {
			perror("read");
			return 1;
		} else {
			cnt[c]++;
		}
	}

	for (int i = 1; i < 256; ++i) {
		if (cnt[i] > cnt[ans]) {
			ans = i;
		}
	}
	printf("highest frequency char %d (%llu times)\n", ans, cnt[ans]);
}
