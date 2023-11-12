
#include <fstream>
#include <string>
#include <iostream>
#include <vector>
#include <random>
#include <numbers>

#include <cstdlib>
#include <cmath>
#include <ctime>

#define CONFIG_IMAGE_WIDTH 512
#define CONFIG_IMAGE_HEIGHT 512
#define CONFIG_NR_STARS 1000
#define CONFIG_HUBBLE_CROSS true

__attribute__((__noreturn__, __cold__))
static void panic(const std::string &reason)
{
	std::cout << "[panic]: " << reason << std::endl;
	std::abort();
}

static unsigned int rand_in_range(unsigned int start, unsigned int end)
{
	return std::rand() % (end - start + 1) + start;
}

static float rand_in_range(float start, float end)
{
	return static_cast<float>(std::rand()) / (RAND_MAX / (end - start + 1)) + start;
}

struct Point {
	float x;
	float y;

	Point(void)
		: x(0), y(0)
	{ }

	Point(float x, float y)
		: x(x), y(y)
	{ }
};

std::ostream &operator <<(std::ostream &out, const Point &p)
{
	return out << '(' << p.x << ", " << p.y << ')';
}

static Point random_unit(void)
{
	float angle = rand_in_range(0.f, 2 * std::numbers::pi);
	std::cerr << angle << ' ';
	return Point(std::cos(angle), std::sin(angle));
}

static float dot(const Point &a, const Point &b)
{
	return a.x * b.x + a.y * b.y;
}

static float length(const Point &p)
{
	return std::sqrt(dot(p, p));
}

Point normalize(Point p)
{
	float x = 1.f / length(p);
	p.x *= x;
	p.y *= x;
	return p;
}

template <class T>
class Matrix {
protected:
	unsigned int w;
	unsigned int h;
	std::vector<T> data;

public:
	Matrix(unsigned int w, unsigned int h)
		: w(w), h(h)
	{
		data.resize(w * h);
	}

	T &get(unsigned int x, unsigned int y)
	{
		if (x >= w or y >= h) {
			panic("Matrix::get: out of bounds");
		}

		return data.at(y * w + x);
	}

	void clear(const T &v)
	{
		data.assign(data.size(), v);
	}

	unsigned int get_w(void) const
	{
		return w;
	}

	unsigned int get_h(void) const
	{
		return h;
	}
};

class Image : private Matrix<unsigned int> {
public:
	static constexpr unsigned char white = 255u;
	static constexpr unsigned char black = 0u;

private:

public:
	Image(unsigned int w, unsigned int h)
		: Matrix(w, h)
	{ }

	void set_pix(unsigned int px, unsigned int py, unsigned char intensity)
	{
		if (px >= w or py >= h) {
			panic("Image::set_pix: out of bounds");
		}

		Matrix::get(px, py) = intensity;
	}

	void set_pix_safe(unsigned int px, unsigned int py, unsigned int intensity)
	{
		if (px >= w or py >= h) {
			return;
		}

		Matrix::get(px, py) = intensity;
	}

	void add_pix(unsigned int px, unsigned int py, unsigned int intensity)
	{
		if (px >= w or py >= h) {
			panic("Image::set_pix: out of bounds");
		}

		Matrix::get(px, py) += intensity;
	}

	void add_pix_safe(unsigned int px, unsigned int py, unsigned int intensity)
	{
		if (px >= w or py >= h) {
			return;
		}

		Matrix::get(px, py) += intensity;
	}


	unsigned int get_pix(unsigned int px, unsigned int py)
	{
		if (px >= w or py >= h) {
			panic("Image::get_pix: out of bounds");
		}

		return Matrix::get(px, py);
	}

	unsigned int get_w(void) const
	{
		return Matrix::get_w();
	}

	unsigned int get_h(void) const
	{
		return Matrix::get_h();
	}

	void clear(unsigned char c=Image::black)
	{
		Matrix::clear(c);
	}

	void to_pgm(std::ostream &out, const std::string &comment)
	{
		print_header(out, comment);
		print_data(out);
	}

	void to_pgm(std::ostream &out)
	{
		to_pgm(out, "");
	}

private:
	void print_header(std::ostream &out, const std::string &comment)
	{
		out << "P5\n";
		if (!comment.empty()) {
			out << '#' << comment << '\n';
		}
		out << w << ' ' << h << '\n';
		out << static_cast<int>(Image::white) << '\n';
	}

	void print_data(std::ostream &out)
	{
		unsigned char c;
		for (unsigned int y = 0; y < h; ++y) {
			for (unsigned int x = 0; x < w; ++x) {
				c = std::min(255u, get_pix(x, y));
				out << c;
			}
		}
	}
};

static float hypot(unsigned int x0, unsigned int y0, unsigned int x1,
		   unsigned int y1)
{
	return std::sqrt((x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1));
}

static float uniform_cos(unsigned int x, unsigned int max, float p)
{
	float xv, c;

	xv = static_cast<float>(x) / static_cast<float>(max);
	c = std::cos((2.f * xv - 1) * std::numbers::pi);
	return std::pow((c + 1.f) / 2.f, p);
}

static void draw_line(Image &im, int x0, int y0, int x1, int y1, float intensity)
{
	int init_x, init_y;
	int dx, dy;
	int sx, sy;
	int err;
	float length, dist, alpha;

	init_x = x0;
	init_y = y0;

	dx = std::abs(x1 - x0);
	dy = -std::abs(y1 - y0);

	length = std::sqrt(dx * dx + dy * dy);

	sx = x0 < x1 ? 1 : -1;
	sy = y0 < y1 ? 1 : -1;

	err = dx + dy;

	while (true) {
		dist = hypot(init_x, init_y, x0, y0);
		alpha = uniform_cos(dist, length, 4);
		im.add_pix_safe(x0, y0, static_cast<unsigned int>(255.f * intensity * alpha));

		if (x0 == x1 and y0 == y1) {
			break;
		}

		if (err * 2 > dy) {
			if (x0 == x1) {
				break;
			}
			err += dy;
			x0 += sx;
		}
		if (err * 2 <= dx) {
			if (y0 == y1) {
				break;
			}
			err += dx;
			y0 += sy;
		}
	}
}

static void hubble_cross(Image &im, unsigned int x, unsigned int y,
			 unsigned int r, float intensity)
{
	unsigned int x0, y0, x1, y1;

	if (not CONFIG_HUBBLE_CROSS) {
		return;
	}

	x0 = x - r / 2;
	y0 = y - r / 2;
	x1 = x + r / 2;
	y1 = y + r / 2;

	if (hypot(x0, y0, x1, y1) > 0.03f * std::max(im.get_w(), im.get_h())) {
		draw_line(im, x - r / 2, y - r / 2, x + r / 2, y + r / 2, intensity);
		draw_line(im, x + r / 2, y - r / 2, x - r / 2, y + r / 2, intensity);
	}
}

static void create_star(Image &im, unsigned int x, unsigned int y,
			unsigned int r, float intensity)
{
	float cx, cy, c;
	unsigned int px, py;

	for (unsigned int h = 0; h < r; ++h) {
		for (unsigned int w = 0; w < r; ++w) {
			cx = uniform_cos(w, r, 3);
			cy = uniform_cos(h, r, 3);
			c = cx * cy * 255. * intensity;

			px = x + w - r / 2;
			py = y + h - r / 2;

			im.add_pix_safe(px, py, static_cast<unsigned char>(c));
		}
	}
	hubble_cross(im, x, y, r * 3, intensity);
}

static void create_stars(Image &im, int nr_stars)
{
	unsigned int star_x, star_y, star_r;

	for (int i = 0; i < nr_stars; ++i) {
		star_x = rand_in_range(0, im.get_w() - 1);
		star_y = rand_in_range(0, im.get_h() - 1);
		star_r = rand_in_range(1, std::min(im.get_h(), im.get_w()));

		if (i * 100.f / nr_stars < 95.f) {
			star_r /= 100;
		} else if (i * 100.f / nr_stars < 98.f) {
			star_r /= 30;
		} else {
			star_r /= 20;
		}

		create_star(im, star_x, star_y, star_r, 1.f);
	}
}

__attribute__((__used__))
static void create_dust(Image &im, float density)
{
	float intensity;
	int type;

	for (unsigned y = 0; y < im.get_h(); ++y) {
		for (unsigned x = 0; x < im.get_w(); ++x) {
			if (rand_in_range(0.f, 1.f) < density) {
				type = rand_in_range(0u, 100u);

				if (type < 80) {
					intensity = rand_in_range(1u, 30u);
				} else if (type < 95) {
					intensity = rand_in_range(1u, 100u);
				} else {
					intensity = rand_in_range(1u, 255u);
				}
				im.add_pix(x, y, intensity);
			}
		}
	}
}

__attribute__((__used__))
static void perlin(Image &im, int period)
{
	Matrix<Point> mat(period + 1, period + 1);
	Point d1, d2, d3, d4;
	float v1, v2, v3, v4;
	float v12, v34, v;
	float dx, dy;
	int px, py;
	int grid_x, grid_y;
	unsigned char c;

	const int delta_x = im.get_h() / period;
	const int delta_y = im.get_w() / period;

	for (unsigned int y = 0; y <= static_cast<unsigned>(period); ++y) {
		for (unsigned int x = 0; x <= static_cast<unsigned>(period); ++x) {
			mat.get(x, y) = random_unit();
			std::cerr << mat.get(x, y) << '\n';
		}
	}

	for (int y = 0; y < static_cast<int>(im.get_h()); ++y) {
		for (int x = 0; x < static_cast<int>(im.get_w()); ++x) {
			px = x * period / im.get_w();
			py = y * period / im.get_h();

			grid_x = px * delta_x;
			grid_y = py * delta_x;

			d1 = Point(x - grid_x, y - grid_y);
			d2 = Point(x - (grid_x + delta_x), y - grid_y);
			d3 = Point(x - grid_x, y - (grid_y + delta_y));
			d4 = Point(x - (grid_x + delta_x), y - (grid_y + delta_y));

			v1 = dot(mat.get(px, py), normalize(d1));
			v2 = dot(mat.get(px + 1, py), normalize(d2));
			v3 = dot(mat.get(px, py + 1), normalize(d3));
			v4 = dot(mat.get(px + 1, py + 1), normalize(d4));

			dx = static_cast<float>(x % delta_x) / delta_x;
			dy = static_cast<float>(y % delta_y) / delta_y;
			v12 = std::lerp(v1, v2, dx);
			v34 = std::lerp(v3, v4, dx);
			v = std::lerp(v12, v34, dy);

			v = (v + 1) * 255 / 2;
			c = static_cast<unsigned char>(v);
			im.set_pix_safe(x, y, c);

			if (y == 4 && x == 49) {
				std::cerr << x << ' ' << y << ' ' << px << '\n';
				std::cerr << d2 << ' ' << v2 << '\n';
			}
			if (y == 4 && x == 50) {
				std::cerr << x << ' ' << y << ' ' << px << '\n';
				std::cerr << d1 << ' ' << v1 << '\n';
			}

			// std::cerr << '(' << x << ',' << y << ")\n";
			// std::cerr << px << ' ' << py << '\n';
			// std::cerr << grid_x << ' ' << grid_y << '\n';
			// std::cerr << mat.get(px, py) << 'x' << d1 << '\n';
			// std::cerr << v1 << "\n============\n";

			// std::cerr << '(' << x << ',' << y << ")\n";
			// std::cestd::cerr << px + 1 << ' ' << py << '\n';
			// std::cestd::cerr << grid_x + delta_x << ' ' << grid_y << '\n';
			// std::cestd::cerr << mat.get(px + 1, py) << 'x' << d2 << '\n';
			// std::cestd::cerr << v2 << "\n============\n";
		}
	}
}

int main()
{
	Image im(CONFIG_IMAGE_WIDTH, CONFIG_IMAGE_HEIGHT);

	std::srand(std::time(nullptr));
	create_stars(im, CONFIG_NR_STARS);
	create_dust(im, 0.1f);
	//perlin(im, 3);

	im.to_pgm(std::cout);
}
