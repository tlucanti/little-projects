
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
	return random() % (end - start + 1) + start;
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

	unsigned int get_w(void)
	{
		return w;
	}

	unsigned int get_h(void)
	{
		return h;
	}

	void clear(unsigned char c=Image::black)
	{
		Matrix::clear(this->black);
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

static void draw_line(Image &im, int x0, int y0, int x1, int y1)
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
		im.add_pix_safe(x0, y0, static_cast<unsigned int>(255 * alpha));

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
	float c, alpha;
	unsigned int px, py;
	unsigned int x0, y0, x1, y1;

	if (not CONFIG_HUBBLE_CROSS) {
		return;
	}

	x0 = x - r / 2;
	y0 = y - r / 2;
	x1 = x + r / 2;
	y1 = y + r / 2;

	if (hypot(x0, y0, x1, y1) > 0.03f * std::max(im.get_w(), im.get_h())) {
		draw_line(im, x - r / 2, y - r / 2, x + r / 2, y + r / 2);
		draw_line(im, x + r / 2, y - r / 2, x - r / 2, y + r / 2);
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
			c = cx * cy * intensity;

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

		create_star(im, star_x, star_y, star_r, 255.f);
	}
}

static void perlin(Image &im, float freq)
{

}

int main()
{
	Image im(CONFIG_IMAGE_WIDTH, CONFIG_IMAGE_HEIGHT);

	std::srand(std::time(nullptr));
	create_stars(im, CONFIG_NR_STARS);
	im.to_pgm(std::cout);
}
