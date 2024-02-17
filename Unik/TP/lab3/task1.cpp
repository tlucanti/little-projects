
#include <string>
#include <stdexcept>
#include <cmath>
#include <numbers>
#include <iostream>

class Figure {
protected:
	const std::string _name;
	float a;
	float b;

public:
	Figure(const std::string &_name, float a=0, float b=0)
		: _name(_name), a(a), b(b)
	{}

	const std::string &name() const
	{
		return _name;
	}
	virtual float volume() const = 0;
	virtual ~Figure() {}
};

class TruncPyramid : public Figure {
private:
	float h;

public:
	TruncPyramid(float lower_width, float upper_width, float height)
		: Figure("truncated pyramid", lower_width, upper_width),
		  h(height)
	{
		if (lower_width < 0 || upper_width < 0 || height < 0) {
			throw std::invalid_argument("negative values");
		}
		if (lower_width > upper_width) {
			throw std::invalid_argument("lower_width > upper_width");
		}
	}

	float volume() const override
	{
		float s1 = a * a;
		float s2 = b * b;

		return (s1 + sqrt(s1 * s2) + s2) * h / 3;
	}
};

class TrianglePrism : public Figure {
private:
	float c;
	float h;

public:
	TrianglePrism(float a, float b, float c, float height)
		: Figure("triangle prism", a, b),
		  c(c), h(height)
	{
		if (a < 0 || b < 0 || c < 0 || height < 0) {
			throw std::invalid_argument("negative values");
		}
		if (a + b <= c || b + c <= a || a + c <= b) {
			throw std::invalid_argument("singular triangle");
		}
	}

	float volume() const override
	{
		float p = (a + b + c) / 2;
		float s = sqrt(p * (p - a) * (p - b) * (p - c));

		return s * h;
	}
};

class HollowSphere : public Figure {
public:
	HollowSphere(float r1, float r2)
		: Figure("hollow sphere")
	{
		if (r1 < 0 || r2 < 0) {
			throw std::invalid_argument("negative values");
		}
		a = std::max(r1, r2);
		b = std::min(r1, r2);
	}

	float volume() const override
	{
		float v1 = std::numbers::pi * cube(a) * 4 / 3;
		float v2 = std::numbers::pi * cube(b) * 4 / 3;

		return v1 - v2;
	}

private:
	float cube(float x) const
	{
		return x * x * x;
	}
};

Figure *getRandomFig(float a, float b, float c, float d)
{
	switch (random() % 3) {
	case 0: return new TruncPyramid(a, b, c);
	case 1: return new TrianglePrism(a, b, c, d);
	case 2: return new HollowSphere(a, b);
	default: abort();
	}
}

int main()
{
	TruncPyramid py(4, 5, 3);
	TrianglePrism pr(3, 4, 5, 10);
	HollowSphere sp(1, 2);

	srand(time(NULL));

	std::cout << py.name() << ": " << py.volume() << '\n';
	std::cout << pr.name() << ": " << pr.volume() << '\n';
	std::cout << sp.name() << ": " << sp.volume() << '\n';

	Figure *fig = getRandomFig(6, 7, 8, 9);
	if (typeid(*fig) == typeid(TruncPyramid)) {
		std::cout << "TruncPyramid\n";
	} else if (typeid(*fig) == typeid(TrianglePrism)) {
		std::cout << "TrianglePrism\n";
	} else if (typeid(*fig) == typeid(HollowSphere)) {
		std::cout << "HollowSphere\n";
	} else {
		abort();
	}
}





