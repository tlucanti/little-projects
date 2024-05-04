// lab1.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//

#include <iostream>
#include <cmath>
#include <ctime>
#include <thread>

#include <chrono>

#define ARRAY_SIZE(x) (sizeof(x) / sizeof(x[0]))
double refer = 2. * M_PI / 3.;

typedef float T;

#define Q 100000 * 1000
#define NR_THREADS 8

#if 0
T integral(T h) {
    T sum = 0;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < Q; i++) {
        sum += (4 / sqrtf(4.f - powf((h * i + h / 2.f), 2.f))) * h;
    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = (t2 - t1);
    std::cout << "Duration is: " << duration.count() << " seconds\n";
    return sum;
}
#endif

static void integral_runner(T h, int step, float *res) {
    T sum = 0;

    for (int i = step; i < Q; i += NR_THREADS) {
        sum += (4 / sqrtf(4.f - powf((h * i + h / 2.f), 2.f))) * h;
    }

    *res = sum;
}

static T integral_fast(T h)
{
	std::thread threads[NR_THREADS] = {};
	float ans[NR_THREADS] = {};
	float final_ans = 0;

std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < NR_THREADS; i++) {
		threads[i] = std::thread(integral_runner, h, i, &ans[i]);
	}

	for (int i = 0; i < NR_THREADS; i++) {
		threads[i].join();
		final_ans += ans[i];
	}
std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> duration = (t2 - t1);
	std::cout << "Duration is: " << duration.count() << " seconds\n";

	return final_ans;
}

double integral_govna(int q, double h) {
    double sum = 0;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < q; i++) {
        sum = sum + (4 / std::sqrt(4 - std::pow((h * i + h / 2), 2))) * h;
    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = (t2 - t1);
    std::cout << "Duration is: " << duration.count() << " seconds\n";
    return sum;
}

int main()
{
    double a = 0;
    double b = 1;
    double res;
    int   q[] = { /* 10000, 100000, 1000000, 10000000, */ Q };
    double h[ARRAY_SIZE(q)];
    for (unsigned i = 0; i < ARRAY_SIZE(q); i++) {
        h[i] = (b - a) / (double)(q[i]);
        printf("q = %i \n", q[i]);
        printf("h = %f \n", h[i]);

#ifdef GOVNO
	res = integral_govna(q[i], h[i]);
#else
        res = integral_fast(h[i]);
#endif
        printf("res = %f \n", res);
        printf("err = %e \n\n\n", std::abs(res - refer));
    }
}

