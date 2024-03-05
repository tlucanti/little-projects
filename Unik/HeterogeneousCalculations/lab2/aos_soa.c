
#include <malloc.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

#ifndef ARRAY_SIZE
# define ARRAY_SIZE 50000
#endif

#ifndef STRING_SIZE
# define STRING_SIZE 1000
#endif

#ifndef __always_inline
# define __always_inline inline __attribute__((__always_inline__))
#endif

#ifndef TYPE
# define TYPE 1
#endif

static __always_inline float maxf(float a, float b)
{
	return a > b ? a : b;
}

struct Object {
	char opts_a[STRING_SIZE];
	int opts_b[3];
	float opts_c[3];
};

struct Object1 {
	char (*op_a)[STRING_SIZE];
	int (*op_b)[3];
	float (*op_c)[3];
};

struct Object2 {
	char (*op_a)[STRING_SIZE];
	int (*op_b)[3];
	float *op_c1;
	float *op_c2;
	float *op_c3;
};

// [ ---  a --- ][b][c][  ---- a ---][b][c]
//
//
// [   a1    ][   a2    ]...
// [bbb][bbb ...
// [c11c12c13][c21c22c23][ccc][ccc][ccc]
//
// [c11c32c31]
// [c12c22c32]
// [c13c23c33]
//

static void init_objects(int size, struct Object *array)
{
	for (int i = 0; i < size; i++) {
		strcpy(array[i].opts_a, "some name");
		array[i].opts_b[0] = rand() % 1000;
		array[i].opts_b[1] = rand() % 1000;
		array[i].opts_b[2] = rand() % 1000;
		array[i].opts_c[0] = (float)(random() % 1000);
		array[i].opts_c[1] = (float)(random() % 1000);
		array[i].opts_c[2] = (float)(random() % 1000);
	}
}

static void init_objects1(int size, struct Object1 *objs)
{
	for (int i = 0; i < size; i++) {
		strcpy(objs->op_a[i], "some name");
		objs->op_b[i][0] = rand() % 1000;
		objs->op_b[i][1] = rand() % 1000;
		objs->op_b[i][2] = rand() % 1000;
		objs->op_c[i][0] = (float)(random() % 1000);
		objs->op_c[i][1] = (float)(random() % 1000);
		objs->op_c[i][2] = (float)(random() % 1000);
	}
}

static void init_objects2(int size, struct Object2 *objs)
{
	for (int i = 0; i < size; i++) {
		strcpy(objs->op_a[i], "some name");
		objs->op_b[i][0] = rand() % 1000;
		objs->op_b[i][1] = rand() % 1000;
		objs->op_b[i][2] = rand() % 1000;
		objs->op_c1[i] = (float)(random() % 1000);
		objs->op_c2[i] = (float)(random() % 1000);
		objs->op_c3[i] = (float)(random() % 1000);
	}
}

static float compare_objects(int size, struct Object *array)
{
	float max_dist = -1.f;

	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			float d_0 = array[i].opts_c[0] - array[j].opts_c[0];
			float d_1 = array[i].opts_c[1] - array[j].opts_c[1];
			float d_2 = array[i].opts_c[2] - array[j].opts_c[2];

			max_dist = maxf(max_dist, sqrtf(d_0 * d_0 + d_1 * d_1 + d_2 * d_2));
		}
	}
	return max_dist;
}

static float compare_objects1(int size, struct Object1 *objs)
{
	float max_dist = -1.f;

	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			float d_0 = objs->op_c[i][0] - objs->op_c[j][0];
			float d_1 = objs->op_c[i][1] - objs->op_c[j][1];
			float d_2 = objs->op_c[i][2] - objs->op_c[j][2];

			max_dist = maxf(max_dist, sqrtf(d_0 * d_0 + d_1 * d_1 + d_2 * d_2));
		}
	}
	return max_dist;
}

static float compare_objects2(int size, struct Object2 *objs)
{
	float max_dist = -1.f;

	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			float d_0 = objs->op_c1[i] - objs->op_c1[j];
			float d_1 = objs->op_c2[i] - objs->op_c2[j];
			float d_2 = objs->op_c3[i] - objs->op_c3[j];

			max_dist = maxf(max_dist, sqrtf(d_0 * d_0 + d_1 * d_1 + d_2 * d_2));
		}
	}
	return max_dist;
}

static float timer_diff(struct timespec *begin, struct timespec *end)
{
	float res = 0;

	res += end->tv_sec - begin->tv_sec;
	res += (end->tv_nsec - begin->tv_nsec) * 1e-9f;
	return res;
}

void o0(void)
{
	struct Object *objs = malloc(sizeof (struct Object) * ARRAY_SIZE);
	struct timespec begin, end;
	int ret;

	if (objs == NULL) {
		printf("no memory\n");
		abort();
	}
	init_objects(ARRAY_SIZE, objs);

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &begin);
	ret = compare_objects(ARRAY_SIZE, objs);
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);

	write(-1, &ret, sizeof(float));

	printf("%fs\n", timer_diff(&begin, &end));
}

void o1(void)
{
	struct Object1 objs;
	struct timespec begin, end;
	int ret;

	objs.op_a = malloc(sizeof(objs.op_a[0]) * ARRAY_SIZE);
	objs.op_b = malloc(sizeof(objs.op_b[0]) * ARRAY_SIZE);
	objs.op_c = malloc(sizeof(objs.op_c[0]) * ARRAY_SIZE);
	if (objs.op_a == NULL || objs.op_b == NULL || objs.op_c == NULL) {
		printf("no memory\n");
		abort();
	}
	init_objects1(ARRAY_SIZE, &objs);

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &begin);
	ret = compare_objects1(ARRAY_SIZE, &objs);
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);

	write(-1, &ret, sizeof(float));

	printf("%fs\n", timer_diff(&begin, &end));
}

void o2(void)
{
	struct Object2 objs;
	struct timespec begin, end;
	int ret;

	objs.op_a = malloc(sizeof(objs.op_a[0]) * ARRAY_SIZE);
	objs.op_b = malloc(sizeof(objs.op_b[0]) * ARRAY_SIZE);
	objs.op_c1 = malloc(sizeof(float) * ARRAY_SIZE);
	objs.op_c2 = malloc(sizeof(float) * ARRAY_SIZE);
	objs.op_c3 = malloc(sizeof(float) * ARRAY_SIZE);
	if (objs.op_a == NULL || objs.op_b == NULL ||
	    objs.op_c1 == NULL || objs.op_c2 == NULL || objs.op_c3 == NULL) {
		printf("no memory\n");
		abort();
	}
	init_objects2(ARRAY_SIZE, &objs);

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &begin);
	ret = compare_objects2(ARRAY_SIZE, &objs);
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);

	write(-1, &ret, sizeof(float));

	printf("%fs\n", timer_diff(&begin, &end));
}

int main()
{
	switch (TYPE) {
	case 1:
		o0();
		break;
	case 2:
		o1();
		break;
	case 3:
		o2();
		break;
	default:
		printf("invalid type\n");
		abort();
	}
}

