
#include <fstream>
#include <string>
#include <iostream>
#include <vector>
#include <random>
#include <numbers>

#include <cstdlib>
#include <cmath>

#define CONFIG_IMAGE_WIDTH 512
#define CONFIG_IMAGE_HEIGHT 512
#define CONFIG_NR_STARS 100
#define CONFIG_HUBBLE_CROSS true

__attribute__((__noreturn__, __cold__))
void panic(const std::string &reason)
{
	std::cout << "[panic]: " << reason << std::endl;
	std::abort();
}

unsigned int rand_in_range(unsigned int start, unsigned int end)
{
	return rand() % (end - start + 1) + start;
}

class Image {
public:
	static constexpr unsigned char white = 255u;
	static constexpr unsigned char black = 0u;

private:
	unsigned int w;
	unsigned int h;
	std::vector<unsigned int> data;

public:
	Image(unsigned int w, unsigned int h)
		: w(w), h(h)
	{
		data.resize(w * h);
	}

	void set_pix(unsigned int px, unsigned int py, unsigned char intensity)
	{
		if (px >= w or py >= h) {
			panic("Image::set_pix: out of bounds");
		}

		data.at(py * w + px) = intensity;
	}

	void set_pix_safe(unsigned int px, unsigned int py, unsigned int intensity)
	{
		if (px >= w or py >= h) {
			return;
		}

		data.at(py * w + px) = intensity;
	}

	void add_pix_safe(unsigned int px, unsigned int py, unsigned int intensity)
	{
		if (px >= w or py >= h) {
			return;
		}

		data.at(py * w + px) += intensity;
	}


	unsigned int get_pix(unsigned int px, unsigned int py)
	{
		if (px >= w or py >= h) {
			panic("Image::get_pix: out of bounds");
		}

		return data.at(py * w + px);

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
		data.assign(data.size(), c);
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

float uniform_cos(unsigned int x, unsigned int max, float p)
{
	float xv = static_cast<float>(x) / static_cast<float>(max);
	float c = std::cos((2.f * xv - 1) * std::numbers::pi);
	return std::pow((c + 1.f) / 2.f, p);
}

void intensify_pixel(Image &im, int x, int y, float intensity, float alpha)
{
	float c = 255.f * (1.f - std::abs(intensity)) * alpha;
	std::cerr << intensity << ' ';
	im.add_pix_safe(x, y, static_cast<unsigned char>(c));
}

void draw_line(Image &im, unsigned int x1, unsigned int y1, unsigned int x2, unsigned int y2)
{
	int x, y, dx, dy, d, D;
	float length, sin, cos, l2, r;

	x = x1;
	y = y1;
	dx = x2 - x1;
	dy = y2 - y1;
	d = 2 * dy - dx;

	D = 0;
	l2 = dx * dx + dy * dy;
	length = std::sqrt(l2);

	sin = dx / length;
	cos = dy / length;

	while (x < x2) {
		r = (x - x1) * (x - x1) + (y - y1) * (y - y1);
		r = uniform_cos(r, l2, 2);
		r = 1.0f;

		//intensify_pixel(im, x, y - 1, D + cos, r);
		intensify_pixel(im, x, y, D, r);
		//intensify_pixel(im, x, y + 1, D - cos, r);

		x += 1;
		if (d <= 0) {
			D += sin;
			d += 2 * dy;
		} else {
			d += sin - cos;
			d += 2 * (dy - dx);
			y += 1;
		}
	}
}

void hubble_cross(Image &im, unsigned int x, unsigned int y, unsigned int r, float intensity)
{
	float c, alpha;
	unsigned int px, py;

	if (not CONFIG_HUBBLE_CROSS) {
		return;
	}

	draw_line(im, x - r / 2, y - r / 2, x + r / 2, y + r / 2);
	return;

	for (unsigned int i = 0; i < r; ++i) {
		alpha = uniform_cos(i, r, 3);
		c = intensity * alpha;

		px = x + i - r / 2;
		py = y + i - r / 2;

		im.add_pix_safe(x, py, static_cast<unsigned char>(c));
		im.add_pix_safe(px, y, static_cast<unsigned char>(c));
	}
}

void create_star(Image &im, unsigned int x, unsigned int y, unsigned int r, float intensity)
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
	hubble_cross(im, x, y, r * 2, intensity);
}

void create_sky(Image &im, int nr_stars)
{
	unsigned int star_x, star_y, star_r;

	for (int i = 0; i < nr_stars; ++i) {
		star_x = rand_in_range(0, im.get_w() - 1);
		star_y = rand_in_range(0, im.get_h() - 1);
		star_r = rand_in_range(1, std::min(im.get_h(), im.get_w()));

		if (i * 100 / nr_stars < 80) {
			star_r /= 30;
		} else if (i * 100 / nr_stars < 90) {
			star_r /= 20;
		} else {
			star_r /= 10;
		}

		std::cerr << "placing star at " << star_x << ' ' << star_y << '\n';
		create_star(im, star_x, star_y, star_r, 255.f);
	}
}

int main()
{
	Image im(CONFIG_IMAGE_WIDTH, CONFIG_IMAGE_HEIGHT);

	create_sky(im, CONFIG_NR_STARS);
	im.to_pgm(std::cout);
}

