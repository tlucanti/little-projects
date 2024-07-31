
#include <stdio.h>
#include <math.h>

typedef double float_type;

float_type sin_taylor(float_type x, int n)
{
	float_type ans = x;
	float_type power = x;
	float_type factorial = 1;
	int fac = 1;
	int i = 0;

	while (--n) {
		++i;

		power *= x * x;

		factorial *= (fac + 1) * (fac + 2);
		fac += 2;

		if (i % 2 == 0) {
			ans += power / factorial;
		} else {
			ans -= power / factorial;
		}
	}

	return ans;
}

float_type cos_taylor(float_type x, int n)
{
	float_type ans = 1;
	float_type power = 1;
	float_type factorial = 1;
	int fac = 0;
	int i = 0;

	while (--n) {
		++i;

		power *= x * x;

		factorial *= (fac + 1) * (fac + 2);
		fac += 2;

		if (i % 2 == 0) {
			ans += power / factorial;
		} else {
			ans -= power / factorial;
		}
	}

	return ans;
}

float_type tan_taylor(float_type x, int n)
{
	return sin_taylor(x, n) / cos_taylor(x, n);
}

int main()
{
	float_type x = 0.5;
	int n = 10;
	float_type taylor_res = tan_taylor(x, n);

	if (sizeof(float_type) == sizeof(float)) {
		double libc_res = tanf(x);

		printf("taylor tan(%f) with n = %4d terms: %.100f\n", (double)x, n, (double)taylor_res);
		printf("libc tan(%f):                       %.100f\n", (double)x, (double)libc_res);
		printf("error: %.100f\n", libc_res - (double)taylor_res);
	} else if (sizeof(float_type) == sizeof(double)) {
		double libc_res = tan(x);

		printf("taylor tan(%f) with n = %4d terms: %.100f\n", (double)x, n, (double)taylor_res);
		printf("libc tan(%f):                       %.100f\n", (double)x, (double)libc_res);
		printf("error: %.100f\n", libc_res - (double)taylor_res);
	} else if (sizeof(float_type) == sizeof(long double)) {
		long double libc_res = tanl(x);

		printf("taylor tan(%Lf) with n = %4d terms: %.100Lf\n", (long double)x, n, (long double)taylor_res);
		printf("libc tan(%Lf):                       %.100Lf\n", (long double)x, (long double)libc_res);
		printf("error: %.100Lf\n", libc_res - (long double)taylor_res);
	}

}

