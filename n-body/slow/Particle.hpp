#ifndef _PARTICLE_HPP
#define _PARTICLE_HPP
#include <cmath>
#include <stdexcept>
#include "types.hpp"

#define DEFINE_OP(op) Point operator op(Point b) { return { x op b.x, y op b.y, z op b.z }; }
#define DEFINE_SCALAR_OP(op) Point operator op(double b) { return { x op b, y op b, z op b }; }

struct Point {
	double x;
	double y;
	double z;

	DEFINE_OP(+)
	DEFINE_OP(-)
	DEFINE_OP(*)
	DEFINE_OP(/)
	DEFINE_SCALAR_OP(*)
	DEFINE_SCALAR_OP(/)

	double abs(void)
	{
		return x*x + y*y + z*z;
	}

	double &operator [](long i)
	{
		switch (i) {
		case 0: return x;
		case 1: return y;
		case 2: return z;
		default: std::out_of_range("Point::operator [] out of range");
		}
		std::abort();
	}
};

struct Particle
{
public:
  Particle() { init(); }
  void init()
  {
	  pos = {0, 0, 0};
	  vel = {0, 0, 0};
	  acc = {0, 0, 0};
	  mass = 0;
  }
  Point pos;
  Point vel;
  Point acc;
  real_type mass;
};

#endif
