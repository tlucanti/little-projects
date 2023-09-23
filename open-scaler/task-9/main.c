
#include <stdio.h>
#include <math.h>

double sin_taylor(double x, int n)
{
	double ans = x;
	double power = x;
	double factorial = 1;
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

double cos_taylor(double x, int n)
{
	double ans = 1;
	double power = 1;
	double factorial = 1;
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

double tan_taylor(double x, int n)
{
	return sin_taylor(x, n) / cos_taylor(x, n);
}

int main()
{
	double x = 0.5;
	int n = 10;
	double res = tan_taylor(x, n);

	printf("tan(%f) with n = %d ternms: %.18f\n", x, n, res);
	printf("error: %.18f\n", fabs(tan(x) - res));
}

