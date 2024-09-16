
#include <math.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdio.h>
#include <time.h>

#define START 0
#define END 1000000
#define H 0.01
#define TRUE_ANSWER_1K 522761.6108534604
#define TRUE_ANSWER_1M 1281128564.241842

#define TRUE_ANSWER TRUE_ANSWER_1M

#define NR_THREADS 20

typedef double flt;

static double time_diff(struct timespec end, struct timespec start)
{
	double sec = end.tv_sec - start.tv_sec;
	sec += (end.tv_nsec -  start.tv_nsec) * 1e-9;
	return sec;
}

static flt squaref(flt x)
{
	return x * x;
}

static flt function(flt x)
{
	return (100 * x * logf(x + 1)) / (x + 100 * squaref(cosf(0.1 * x)));
}

/**
 * function uses trapezoidal rule to compute integral.
 * formula is the follows:
 *
 * sum(x = a to b with step h): (f(x) + f(x + h)) / 2 * h
 *
 * we can move constant factors 0.5 and h out of sum:
 *
 * h * 0.5 * sum(x = a,h,b):  f(x) + f(x + h)
 *
 * if we expand sum we will get:
 *
 * (f(a) + f(a+h)) + (f(a+h) + f(a+2h)) + ... + (f(b-h) + f(b))
 *
 * as we can see - all terms except first and last are included 2 times, so, we
 * can simplify full formula to:
 *
 * h * 0.5 * [ 2 * ( f(a) + f(a + h) + ... + f(b - h) + f(b) )  - f(a) - f(b) ] =
 *
 * = h * [ f(a) + f(a+h) + ... + f(b)  -  0.5 (f(a) + f(b)) ]
 */
static flt integral_single_thread(flt a, flt b, flt h)
{
	const int n = (b - a) / h;
	flt ans = (function(a) + function(b)) * (flt)0.5;

	a += h;
	for (int i = 0; i < n - 2; i++) {
		ans += function(a);
		a += h;
	}

	return ans * h;
}

static flt integral_omp(flt a, flt b, flt h)
{
	const int n = (b - a) / h;
	flt ans = (function(a) + function(b)) * (flt)0.5;

	a += h;
	#pragma omp parallel for reduction(+:ans)
	for (int i = 0; i < n - 2; i++) {
		ans += function(a + h * i);
	}

	return ans * h;
}

struct thr {
	flt res;
	int tid;
	pthread_t thread;
};

static void *pthread_worker(struct thr *args)
{
	const int tid = args->tid;
	flt ans = 0;
	flt a = START + H * tid;
	flt b = END;

	while (a <= b) {
		ans += function(a);
		a += H * NR_THREADS;
	}

	args->res = ans;
	return NULL;
}

static flt integral_pthread(flt a, flt b, flt h)
{
	struct thr threads[NR_THREADS];
	flt ans = 0;

	for (unsigned long i = 0; i < NR_THREADS; i++) {
		pthread_create(&threads[i].thread, NULL, (void *)pthread_worker, (void *)&threads[i]);
	}

	for (int i = 0; i < NR_THREADS; i++) {
		pthread_join(threads[i].thread, NULL);
		ans += threads[i].res;
	}

	ans -= (function(a) + function(b)) * 0.5f;
	return ans * h;
}

int main(int argc, char **argv)
{
	struct timespec start, end;
	flt ans;

	if (argc != 2)
		return 1;

	clock_gettime(CLOCK_MONOTONIC, &start);
	switch (argv[1][0]) {
	case 's':
		ans = integral_single_thread(START, END, H);
		break;
	case 'o':
		ans = integral_omp(START, END, H);
		break;
	case 'p':
		ans = integral_pthread(START, END, H);
		break;
	default:
		return 1;
	}
	clock_gettime(CLOCK_MONOTONIC, &end);

	double err = fabs(1 - TRUE_ANSWER / ans) * 100;
	printf("answer: %f (relative error: %f%%)\n", ans, err);
	printf("time: %fs\n", time_diff(end, start));
}
