
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <guilib.h>
#include <stdguilib.h>

#ifndef DRAW
# define DRAW true
#endif

typedef float flt;

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

		bodies.pos(0) = vec3(0, 0, 0);
		bodies.vel(0) = vec3(0, 0, 0);
		bodies.acc(0) = vec3(0, 0, 0);
		bodies.mass(0) = 2e12;

		for (int i = 1; i < cnt; i++) {
			bodies.pos(i) = random_unit();
			bodies.vel(i) = random_unit() * 1e1;
			bodies.acc(i) = vec3(0, 0, 0);
			bodies.mass(i) = random_float() * 1e11;
		}
	}

	void update_pos(void)
	{
		for (int i = 0; i < bodies.size(); i++) {
			bodies.pos(i) += bodies.vel(i) * time_step;
		}
	}

	vec3 compute_acc_once(int cur, vec3 init=vec3(0, 0, 0))
	{
		vec3 acc = init;

		for (int other = 0; other < bodies.size(); other++) {
			if (cur == other) {
				continue;;
			}

			vec3 dr = bodies.pos(other) - bodies.pos(cur);
			flt norm = std::pow<flt>(dr.abs(), (flt)-1.5);
			acc += dr * (bodies.mass(other) * norm);
		}

		acc *= CONST_G;
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

	void draw_bodies(void)
	{
		if (not DRAW) {
			return;
		}

		for (int i = 0; i < (int)bodies.size(); i++) {
			int x = (bodies.pos(i).x - bodies.pos(0).x * 0.9 + 2) / 4 * w;
			int y = (bodies.pos(i).y - bodies.pos(0).y * 0.9 + 2) / 4 * h;

			// gui_draw_circle(window, old_x, old_y, 10, COLOR_BLACK);
			flt r = bodies.mass(i) * 1e-10;
			gui_draw_circle(window, x, y, r, COLOR_GREEN);
		}
		gui_draw(window);
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

	body_container bodies;
	static constexpr flt CONST_G = 6.6743015e-11L;
	static constexpr int w = 800;
	static constexpr int h = 600;
	flt time_step;
	gui_window *window;

public:
	NBody(int cnt, flt time_step)
		: time_step(time_step), window(nullptr)
	{
		srand(time(nullptr));

		if (DRAW) {
			gui_bootstrap();
			window = gui_alloc();
			if (window == nullptr)
				throw std::bad_alloc();
			if (gui_create(window, w, h))
				throw std::runtime_error("gui create fail");
		}

		make_bodies(cnt);
	}

	~NBody()
	{
		if (DRAW) {
			gui_destroy(window);
			gui_finalize();
		}
	}

	void run(flt max_time)
	{
		flt time = 0;
		flt energy = get_energy();

		while (time < max_time) {
			if (DRAW && gui_closed(window)) {
				return;
			}

			time += time_step;

			update_pos();
			update_acc();
			update_vel();

			std::cout << "energy error: " << abs((get_energy() - energy) / energy) * 100 << "%\r\n";
			for (int i = 0; i < bodies.size(); i++) {
				std::cout << "body pos: " << bodies.pos(i) << ", vel: " << bodies.vel(i) << ", acc: " << bodies.acc(i) << "\r\n";
			}
			std::cout << "\r\n";
			draw_bodies();
		}
	}
};

std::string print_float(flt val)
{
	char buf[128];
	sprintf(buf, "%7.4Lf", (long double)val);
	return std::string(buf);
}

std::ostream &operator <<(std::ostream &out, const NBody::vec3 &v)
{
	out << '(' << print_float(v.x) << ", " <<
		      print_float(v.y) << ", " <<
		      print_float(v.z) << ')';
	return out;
}

int main()
{
	NBody sim(3, 0.005);

	sim.run(10);
}


