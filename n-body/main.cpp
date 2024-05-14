
#include <cmath>
#include <iostream>
#include <vector>

float sqrt(float x)
{
	return sqrtf(x);
}

double sqrt(double x)
{
	return sqrt(x);
}

long double sqrt(long double x)
{
	return sqrtl(x);
}

template <class flt>
class NBody {
	static constexpr flt CONST_G = 6.6743015e-11L;

	template <class T>
	struct vec3 {
		T x;
		T y;
		T z;

		vec3(T x, T y, T z)
			: x(x), y(y), z(z)
		{}

		T abs(void) const
		{
			return x * x + y * y + z * z;
		}

		T length(void) const
		{
			return sqrt(abs());
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
			v = 1 / v;
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
	};

	struct body {
		vec3 pos;
		vec3 vel;
		vec3 acc;
		flt mass;
	};

	std::vector<body> bodies;
	flt time_step;

	flt square(flt x)
	{
		return x * x;
	}

	flt random_float(void)
	{
		return (flt)rand() / (flt)RAND_MAX;
	}

	vec3 random_unit(void)
	{
		return vec3(random_float(), random_float(), random_float());
	};

	void make_bodies(int cnt)
	{
		bodies.resize(cnt);

		for (int i = 0; i < cnt; i++) {
			bodies[i].pos = random_unit();
			bodies[i].vel = random_unit();
			bodies[i].acc = vec3(0, 0, 0);
		}
	}

	void update_pos(void)
	{
		for (body &b : bodies) {
			b.pos += b.vel * time_step;
		}
	}

	void update_acc(void)
	{
		const flt one_and_half = (flt)3 / (flt)2;

		for (body &b : bodies) {
			b.acc = vec3(0, 0, 0);

			for (const body &other : bodies) {
				if (&b == &other) {
					continue;;
				}

				vec3 dr = other.pos - b.pos;
				flt norm = std::pow<flt>(dr.abs(), -one_and_half);
				b.acc += dr * (other.mass * norm);
			}

		}
	}

	void update_vel(void)
	{
		for (body &b : bodies) {
			b.vel += b.acc * time_step;
		}
	}

	flt get_energy(void)
	{
		 flt energy = 0;

		 for (int i = 0; i < (int)bodies.size(); i++) {
			 for (int j = i + 1; j < bodies.size(); i++) {
				 flt distance = (bodies[i].pos - bodies[j].pos).length();
				 energy -= bodies[i].mass * bodies[j].mass * CONST_G / distance;
			 }
		 }

		 for (const body &b : bodies) {
			 energy += b.mass * square(b.vel.length()) / 2;
		 }

		 return energy;
	}

public:
	NBody(int cnt, flt time_step)
		: time_step(time_step)
	{
		make_bodies(cnt);
	}


	void run(flt max_time)
	{
		flt time = 0;
		flt energy = get_energy();

		while (time < max_time) {
			time += time_step;

			update_pos();
			update_acc();
			update_vel();

			std::cout << "energy error: " << get_energy() - energy << '\n';
		}
	}
};

int main()
{
	NBody<float> sim(3, 0.01);

	sim.run(1);
}


