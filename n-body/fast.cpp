
#include <cmath>
#include <iostream>
#include <ctime>

#ifndef DRAW
# define DRAW true
#endif

#if DRAW
# include <guilib.h>
# include <stdguilib.h>

static gui_window *window = nullptr;
#endif

#ifndef __always_inline
# define __always_inline inline __attribute__((__always_inline__))
#endif

#define BODIES 3000
#define TIME_STEPS 10
#define time_step (flt)0.005
#define CONST_G (flt)6.6743015e-11

typedef float flt;

__always_inline
static double time_diff(struct timespec end, struct timespec start)
{
	double sec = end.tv_sec - start.tv_sec;
	sec += (end.tv_nsec - start.tv_nsec) * 1e-9;
	return sec;
}

namespace bruh {
	template <class T>
	struct is_float : std::false_type {};

	template <>
	struct is_float<float> : std::true_type {};

	template <class T>
	struct is_double : std::false_type {};

	template <>
	struct is_double<double> : std::true_type {};

	template <class T>
	struct is_long_double : std::false_type {};

	template <>
	struct is_long_double<long double> : std::true_type {};

	template <class T>
	__always_inline
	static T sqrt(T x)
	{
		if constexpr (is_float<T>())
			return ::sqrtf(x);
		else if constexpr (is_double<T>())
			return ::sqrt(x);
		else if constexpr (is_long_double<T>())
			return ::sqrtl(x);
		else
			static_assert(std::is_floating_point<T>(), "");
	}
}

struct vec3 {
	flt x;
	flt y;
	flt z;

	__always_inline
	vec3(flt x, flt y, flt z)
		: x(x), y(y), z(z)
	{}

	__always_inline
	vec3(void)
		: x(0), y(0), z(0)
	{}

	__always_inline
	flt abs(void) const
	{
		return x * x + y * y + z * z;
	}

	__always_inline
	flt length(void) const
	{
		return bruh::sqrt<flt>(abs());
	}

	__always_inline
	vec3 operator +(const vec3 &other) const
	{
		return vec3(x + other.x, y + other.y, z + other.z);
	}

	__always_inline
	vec3 operator -(const vec3 &other) const
	{
		return vec3(x - other.x, y - other.y, z - other.z);
	}

	__always_inline
	vec3 operator *(flt v) const
	{
		return vec3(x * v, y * v, z * v);
	}

	__always_inline
	void operator +=(const vec3 &other)
	{
		x += other.x;
		y += other.y;
		z += other.z;
	}

	__always_inline
	void operator -=(const vec3 &other)
	{
		x -= other.x;
		y -= other.y;
		z -= other.z;
	}

	__always_inline
	void operator *=(const flt v)
	{
		x *= v;
		y *= v;
		z *= v;
	}

	friend std::ostream &operator <<(std::ostream &out, const vec3 &v);
};


__always_inline
static flt square(flt x)
{
	return x * x;
}

__always_inline
static flt random_float(void)
{
	return (flt)rand() * ((flt)1 / (flt)RAND_MAX);
}

__always_inline
static flt random_float_neg(void)
{
	return random_float() * (flt)2 - 1;
}

__always_inline
static vec3 random_unit(void)
{
	return vec3(random_float_neg(), random_float_neg(), random_float_neg());
};

struct body_container {
	struct body {
		vec3 pos;
		vec3 vel;
		vec3 acc;
		flt mass;
	};

	body bv[BODIES];
	int n;

	body_container(void) {}

	vec3 &pos(int i)
	{
		return bv[i].pos;
	}

	const vec3 &pos(int i) const
	{
		return bv[i].pos;
	}

	vec3 &vel(int i)
	{
		return bv[i].vel;
	}

	const vec3 &vel(int i) const
	{
		return bv[i].vel;
	}

	vec3 &acc(int i)
	{
		return bv[i].acc;
	}

	const vec3 &acc(int i) const
	{
		return bv[i].acc;
	}

	flt &mass(int i)
	{
		return bv[i].mass;
	}

	flt mass(int i) const
	{
		return bv[i].mass;
	}
};

static body_container bodies;

__always_inline
static void make_bodies(void)
{
	bodies.pos(0) = vec3(0, 0, 0);
	bodies.vel(0) = vec3(0, 0, 0);
	bodies.acc(0) = vec3(0, 0, 0);
	bodies.mass(0) = random_float();
	if constexpr (DRAW) {
		bodies.mass(0) *= 2e11;
	}

	for (int i = 1; i < BODIES; i++) {
		bodies.pos(i) = random_unit();
		bodies.vel(i) = random_unit() * 1e1;
		bodies.acc(i) = vec3(0, 0, 0);
		bodies.mass(i) = random_float();
		if constexpr (DRAW) {
			bodies.mass(i) *= 1e10;
		}
	}
}

__always_inline
static vec3 compute_acc_once(int cur)
{
	vec3 acc = vec3(0, 0, 0);
	vec3 tmp_vel;
	vec3 tmp_pos;

	for (int other = 0; other < BODIES; other++) {
		if (cur == other) {
			continue;;
		}

		vec3 dist = bodies.pos(other) - bodies.pos(cur);
		flt r = std::pow<flt>(dist.abs(), (flt)-1.5);
		flt norm = CONST_G * bodies.mass(other) * r;

		vec3 k1 = (bodies.pos(other) - bodies.pos(cur)) * norm;

		tmp_vel = bodies.vel(cur) + k1 * (flt)0.5;
		tmp_pos = bodies.pos(cur) + tmp_vel * (flt)0.5 * time_step;
		vec3 k2 = (bodies.pos(other) - tmp_pos) * norm;

		tmp_vel = bodies.vel(cur) + k2 * (flt)0.5;
		tmp_pos = bodies.pos(cur) + tmp_vel * (flt)0.5 * time_step;
		vec3 k3 = (bodies.pos(other) - tmp_pos) * norm;

		tmp_vel = bodies.vel(cur) + k3;
		tmp_pos = bodies.pos(cur) + tmp_vel * (flt)time_step;
		vec3 k4 = (bodies.pos(other) - tmp_pos) * norm;

		acc += (k1 + k2 * 2 + k3 * 2 + k4) * ((flt)1 / 6);
	}

	return acc;
}

__always_inline
static flt get_energy(void)
{
	 flt energy = 0;

	 for (int i = 0; i < BODIES; i++) {
		 for (int j = i + 1; j < BODIES; j++) {
			 flt distance = (bodies.pos(i) - bodies.pos(j)).length();
			 energy -= bodies.mass(i) * bodies.mass(j) * CONST_G / distance;
		 }
	 }

	 for (int i = 0; i < BODIES; i++) {
		 energy += bodies.mass(i) * square(bodies.vel(i).length()) / 2;
	 }

	 return energy;
}

#if DRAW
__always_inline
static void draw_bodies(void)
{
	static gui_window *window = nullptr;
	static constexpr int w = 3000;
	static constexpr int h = 2000;

	if (window == nullptr) {
		window = gui_alloc();
		gui_create(window, w, h);
	}

	static std::vector<int> prev_x;
	static std::vector<int> prev_y;

	if (prev_x.empty()) {
		prev_x.resize(bodies.size());
		prev_y.resize(bodies.size());
	}

	for (int i = 0; i < (int)bodies.size(); i++) {
		int x = (bodies.pos(i).x + 10) / 20 * w;
		int y = (bodies.pos(i).y + 10) / 20 * h;
		int old_x = prev_x.at(i);
		int old_y = prev_y.at(i);

		flt r = bodies.mass(i) * 1e-9;
		gui_draw_circle(window, old_x, old_y, r, COLOR_BLACK);
		gui_draw_circle(window, x, y, r, COLOR_GREEN);

		prev_x.at(i) = x;
		prev_y.at(i) = y;
	}
	gui_draw(window);
}
#endif

__always_inline
static void loop_iter(void)
{
	// update_pos
	for (int i = 0; i < BODIES; i++) {
		bodies.pos(i) += bodies.vel(i) * time_step;
		bodies.acc(i) = compute_acc_once(i);
		bodies.vel(i) += bodies.acc(i) * time_step;
	}
}

void run(void)
{
	struct timespec start, end;
	flt energy = get_energy();

#ifdef __linux__
	clock_gettime(CLOCK_MONOTONIC, &start);
#endif

	for (unsigned i = 0; i < TIME_STEPS; i++) {
	#if DRAW
		if (gui_closed(window)) {
			return;
		}
	#endif

		loop_iter();

		std::cout << "energy error: " << std::abs((get_energy() - energy) / energy) * 100 << "%\r\n";
	#if DRAW
		// for (int i = 0; i < bodies.size(); i++) {
		// 	std::cout << "body pos: " << bodies.pos(i) << ", vel: " << bodies.vel(i) << ", acc: " << bodies.acc(i) << "\r\n";
		// }
		// std::cout << "\r\n";
		draw_bodies();
	#endif
	}
#ifdef __linux__
	clock_gettime(CLOCK_MONOTONIC, &end);
#endif

	std::cout << "energy error: " << std::abs((get_energy() - energy) / energy) * 100 << "\r\n";
	std::cout << "elapsed time: " << time_diff(end, start) << "\r\n";
}

std::string print_float(flt val)
{
	char buf[128];
	sprintf(buf, "%7.4Lf", (long double)val);
	return std::string(buf);
}

std::ostream &operator <<(std::ostream &out, const vec3 &v)
{
	out << '(' << print_float(v.x) << ", " <<
		      print_float(v.y) << ", " <<
		      print_float(v.z) << ')';
	return out;
}

int main()
{
	make_bodies();
	run();
}

