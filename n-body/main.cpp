
#include <cmath>
#include <iostream>
#include <vector>
#include <ctime>

#ifndef DRAW
# define DRAW true
#endif

#if DRAW
# include <guilib.h>
# include <stdguilib.h>
#endif

#ifndef ALGO
# define ALGO RK2
#endif

#define SCREEN_W 1920
#define SCREEN_H 1080
#define ZOOM (flt)20

#define MIN_SIMULATED_DIST (flt)1e-15

typedef double flt;

enum algo {
	EULER,
	HEUN,
	RK3,
	RK4,
	RK1 = EULER,
	RK2 = HEUN,
};

double time_diff(struct timespec end, struct timespec start)
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
	T sqrt(T x)
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

class NBody {
	struct vec3 {
		flt x;
		flt y;
		flt z;

		vec3(flt x, flt y, flt z)
			: x(x), y(y), z(z)
		{}

		vec3(void)
			: x(0), y(0), z(0)
		{}

		flt abs(void) const
		{
			return x * x + y * y + z * z;
		}

		flt length(void) const
		{
			return bruh::sqrt<flt>(abs());
		}

		vec3 operator +(const vec3 &other) const
		{
			return vec3(x + other.x, y + other.y, z + other.z);
		}

		vec3 operator -(const vec3 &other) const
		{
			return vec3(x - other.x, y - other.y, z - other.z);
		}

		vec3 operator *(flt v) const
		{
			return vec3(x * v, y * v, z * v);
		}

		vec3 operator /(flt v) const
		{
			v = (flt)1 / v;
			return vec3(x * v, y * v, z * v);
		}

		void operator +=(const vec3 &other)
		{
			x += other.x;
			y += other.y;
			z += other.z;
		}

		void operator -=(const vec3 &other)
		{
			x -= other.x;
			y -= other.y;
			z -= other.z;
		}

		void operator *=(const flt v)
		{
			x *= v;
			y *= v;
			z *= v;
		}
	};

	friend std::ostream &operator <<(std::ostream &out, const NBody::vec3 &v);

	struct body {
		vec3 pos;
		vec3 vel;
		vec3 acc;
		flt mass;
	};

	flt square(flt x)
	{
		return x * x;
	}

	flt random_float(void)
	{
		return (flt)rand() / (flt)RAND_MAX;
	}

	flt random_float_neg(void)
	{
		return random_float() * (flt)2 - 1;
	}

	vec3 random_unit(void)
	{
		return vec3(random_float_neg(), random_float_neg(), random_float_neg());
	};

	void make_bodies(int cnt)
	{
		bodies.setsize(cnt);

		for (int i = 0; i < cnt; i++) {
			bodies.pos(i) = random_unit() * 1.5;
			bodies.vel(i) = random_unit() * 1e1;
			bodies.acc(i) = vec3(0, 0, 0);
			bodies.mass(i) = random_float() * 3e6;
			if (DRAW) {
				bodies.mass(i) *= 1;
			}
		}
	}

	void update_pos(void)
	{
		for (int i = 0; i < bodies.size(); i++) {
			bodies.pos(i) += bodies.vel(i) * time_step;
		}
	}

	vec3 compute_acc_once(int cur)
	{
		vec3 acc = vec3(0, 0, 0);
		vec3 tmp_vel;
		vec3 tmp_pos;
		vec3 k1, k2, k3, k4;

		for (int other = 0; other < bodies.size(); other++) {
			if (cur == other) {
				continue;;
			}


			vec3 dist = bodies.pos(other) - bodies.pos(cur);
			flt r = dist.abs();
			r = std::pow<flt>(bruh::sqrt<flt>(r), 3);
			if (r < MIN_SIMULATED_DIST) {
				r = MIN_SIMULATED_DIST;
			}
			flt norm = CONST_G * bodies.mass(other) * bodies.mass(cur) / r;
			k1 = (bodies.pos(other) - bodies.pos(cur)) * norm;
			if (ALGO == EULER) {
				acc += k1 * time_step;
				continue;
			}

			tmp_vel = bodies.vel(cur) + k1 * time_step * (flt)0.5;
			tmp_pos = bodies.pos(cur) + tmp_vel * time_step * (flt)0.5;
			k2 = (bodies.pos(other) - tmp_pos) * norm;
			if (ALGO == HEUN) {
				// acc += (k1 + k2) * 0.5 * time_step;
				acc += k2 * time_step;
				continue;
			}

			tmp_vel = bodies.vel(cur) + k2 * (flt)0.5;
			tmp_pos = bodies.pos(cur) + tmp_vel * (flt)0.5 * time_step;
			k3 = (bodies.pos(other) - tmp_pos) * norm;

			tmp_vel = bodies.vel(cur) + k3;
			tmp_pos = bodies.pos(cur) + tmp_vel * time_step;
			k4 = (bodies.pos(other) - tmp_pos) * norm;

			acc += (k1 + k2 * 2 + k3 * 3 + k4) / (flt)6 * time_step;
		}

		return acc;
	}


	void update_acc(void)
	{
		for (int i = 0; i < bodies.size(); i++) {
			bodies.acc(i) = compute_acc_once(i);
		}
	}

	void update_vel(void)
	{
		for (int i = 0; i < bodies.size(); i++) {
			bodies.vel(i) += bodies.acc(i) * time_step;
		}
	}

	flt get_energy(void)
	{
		 flt energy = 0;

		 for (int i = 0; i < bodies.size(); i++) {
			 for (int j = i + 1; j < bodies.size(); j++) {
				 flt distance = (bodies.pos(i) - bodies.pos(j)).length();
				 energy -= bodies.mass(i) * bodies.mass(j) * CONST_G / distance;
			 }
		 }

		 for (int i = 0; i < bodies.size(); i++) {
			 energy += bodies.mass(i) * square(bodies.vel(i).length()) / 2;
		 }

		 return energy;
	}

	unsigned get_color(flt x)
	{
		unsigned r, g;

		x *= 0.01;
		r = 255 * (1 - exp(-x));
		g = 255 * (1 - exp(-2/x));

		return r | (g << 8);
	}

	void draw_bodies(void)
	{
#if DRAW
		static std::vector<int> prev_x;
		static std::vector<int> prev_y;
		static flt max_mass = 0;

		if (prev_x.empty()) {
			prev_x.resize(bodies.size());
			prev_y.resize(bodies.size());
			for (int i = 0; i < (int)bodies.size(); i++) {
				max_mass = std::max(max_mass, bodies.mass(i));
			}
		}

		for (int i = 0; i < (int)bodies.size(); i++) {
			int x = (bodies.pos(i).x + ZOOM / 2) / ZOOM * SCREEN_W;
			int y = (bodies.pos(i).y + ZOOM / 2) / ZOOM * SCREEN_H;
			int old_x = prev_x.at(i);
			int old_y = prev_y.at(i);
			unsigned color;

			if (old_x == 0 && old_y == 0) {
				old_x = x;
				old_y = y;
			}

			flt r = bodies.mass(i) / max_mass * 20;
			r = std::min(r, (flt)5);
			//gui_draw_circle(window, old_x, old_y, r, COLOR_BLACK);
			if (std::abs(x) < SCREEN_W && std::abs(y) < SCREEN_H) {
				gui_draw_line(window, old_x, old_y, x, y, 0xFF0000);
			}
			color = get_color(bodies.vel(i).abs());
			gui_draw_circle(window, x, y, r, color);

			prev_x.at(i) = x;
			prev_y.at(i) = y;

			if (1) {
				std::cout << x << ' ' << y << ' ';
				std::cout << "pos: " << bodies.pos(i)
					  << ", vel: " << bodies.vel(i)
					  << ", acc: " << bodies.acc(i)
					  << ", mass: " << bodies.mass(i)
					  << '\n';
			}
		}
		gui_draw(window);
#endif
	}

	struct body_container {
		std::vector<body> bv;
		int n;

		body_container(void) {}

		void setsize(int n) {
			this->n = n;
			bv.resize(n);
		}

		int size(void) {
			return n;
		}

		vec3 &pos(int i) {
			return bv[i].pos;
		}

		vec3 &vel(int i) {
			return bv[i].vel;
		}

		vec3 &acc(int i) {
			return bv[i].acc;
		}

		flt &mass(int i) {
			return bv[i].mass;
		}
	};

	vec3 computeAcc(int cur)
	{
		vec3 acc = { 0, 0, 0 };

		for (int other = 0; other < bodies.size(); other++) {
			if (other == cur)
				continue;
			vec3 dr = bodies.pos(cur) - bodies.pos(other);
			acc -= dr * CONST_G * bodies.mass(cur) * bodies.mass(other) / std::pow(dr.length(), 3);
		}

		return acc;
	}

	void updatePlanets(void)
	{
		for (int i = 0; i < bodies.size(); i++) {
			bodies.acc(i) = computeAcc(i);
		}

		for (int i = 0; i < bodies.size(); i++) {
			vec3 pk1 = bodies.vel(i) * time_step;
			bodies.pos(i) += pk1; continue;

			vec3 pk2 = (bodies.vel(i) + pk1 * 0.5) * time_step;
			vec3 pk3 = (bodies.vel(i) + pk2 * 0.5) * time_step;
			vec3 pk4 = (bodies.vel(i) + pk3) * time_step;

			bodies.pos(i) += (pk1 + pk2 * 2 + pk3 * 2 + pk4) * (flt)1/6;
		}

		for (int i = 0; i < bodies.size(); i++) {
			vec3 vk1 = bodies.acc(i) * time_step;
			bodies.vel(i) += vk1; continue;

			vec3 vk2 = (bodies.acc(i) + vk1 * 0.5) * time_step;
			vec3 vk3 = (bodies.acc(i) + vk2 * 0.5) * time_step;
			vec3 vk4 = (bodies.acc(i) + vk3) * time_step;

			bodies.vel(i) += (vk1 + vk2 * 2 + vk3 * 2 + vk4) * (flt)1/6;
		}
	}

	body_container bodies;
	static constexpr flt CONST_G = 6.6743015e-11L;
	flt time_step;
#if DRAW
	gui_window *window;
#endif

public:
	NBody(int cnt, flt time_step)
		: time_step(time_step)
	{
		srand(time(nullptr));
		// srand(4);

#if DRAW
		gui_bootstrap();
		window = gui_alloc();
		if (window == nullptr)
			throw std::bad_alloc();
		if (gui_create(window, SCREEN_W, SCREEN_H))
			throw std::runtime_error("gui create fail");
#endif

		make_bodies(cnt);
	}

	~NBody()
	{
#if DRAW
		gui_destroy(window);
		gui_finalize();
#endif
	}

	void run(int nr_steps)
	{
		struct timespec start, end;
		flt energy = get_energy();

#if DRAW
		draw_bodies();
#endif

		clock_gettime(CLOCK_MONOTONIC, &start);
		while (nr_steps-- > 0) {
#if DRAW
			if (gui_closed(window)) {
				return;
			}
#endif

			updatePlanets();
			// update_pos();
			// update_acc();
			// update_vel();

			if (DRAW) {
				draw_bodies();
			}

			printf("energy error: %.10f\n", std::abs((get_energy() - energy) / energy));
		}
		clock_gettime(CLOCK_MONOTONIC, &end);

		std::cout << "elapsed time: " << time_diff(end, start) << "\r\n";
	}
};

std::string print_float(flt val)
{
	char buf[128];
	if (std::abs(val) > 1e10) {
		sprintf(buf, "%Lg", (long double)val);
	} else {
		sprintf(buf, "%8.4Lf", (long double)val);
	}
	return std::string(buf);
}

std::ostream &operator <<(std::ostream &out, const NBody::vec3 &v)
{
	out << '(' << print_float(v.x) << ", " <<
		      print_float(v.y) << ", " <<
		      print_float(v.z) << ')';
	return out;
}

int main(int argc, char **argv)
{
	if (argc != 3) {
		return 1;
	}
#ifdef RK4
	std::cout << "RK4" << std::endl;
#elif defined(HEUN)
	std::cout << "HEUN" << std::endl;
#endif

	int nr_bodies = atoi(argv[1]);
	int nr_steps = atoi(argv[2]);
	NBody sim(nr_bodies, 0.005);

	sim.run(nr_steps);
}

