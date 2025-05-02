
#define FLOAT_TYPE double
#include "common.h"

static flt function(flt x)
{
	return (100 * x * logf(x + 1)) / (x + 100 * square(cosf(0.1 * x)));
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
#pragma omp parallel for reduction(+ : ans)
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
	const int n = (INT_END - INT_START) / INT_H;
	flt ans = 0;
	// flt c = 0;
	flt a = INT_START;

	for (int i = tid; i < n; i += NR_THREADS) {
		ans += function(a + i * INT_H);

		// flt y = function(a) - c;
		// flt t = ans + y;
		// c = (t - ans) - y ;
		// ans = t;
		// a += H * NR_THREADS;
	}

	args->res = ans;
	return NULL;
}

static flt integral_pthread(flt a, flt b, flt h)
{
	struct thr threads[NR_THREADS];
	flt ans = 0;

	for (unsigned long i = 0; i < NR_THREADS; i++) {
		threads[i].tid = i;
		pthread_create(&threads[i].thread, NULL, (void *)pthread_worker,
			       (void *)&threads[i]);
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
	struct timespec begin, end;
	flt ans;

	if (argc != 2)
		goto args;

	clock_gettime(CLOCK_MONOTONIC, &begin);
	switch (argv[1][0]) {
	case 's':
		ans = integral_single_thread(INT_START, INT_END, INT_H);
		break;
	case 'o':
		ans = integral_omp(INT_START, INT_END, INT_H);
		break;
	case 'p':
		ans = integral_pthread(INT_START, INT_END, INT_H);
		break;
	default:
		goto args;
	}
	clock_gettime(CLOCK_MONOTONIC, &end);

	double err = fabs(1 - INT_TRUE_ANSWER / ans) * 100;
	printf("answer: %f (relative error: %f%%)\n", ans, err);
	printf("time: %fs\n", time_diff(&begin, &end));

	return 0;

args:
	printf("invalid args, expected 's', 'o', 'p'\n"
	       "\ts: single thread execution\n"
	       "\tp: pthread multithreaded execution\n"
	       "\to: OMP multithreaded execution\n");
	return 1;
}
