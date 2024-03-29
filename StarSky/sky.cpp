
#include <fstream>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <numbers>

#include <cstdlib>
#include <cmath>
#include <ctime>

#ifndef __noret
# define __noret [[noreturn]]
#endif
#ifndef __cold
# define __cold __attribute__((__cold__))
#endif
#ifndef __used
# define __used [[maybe_unused]]
#endif
#ifndef likely
# define lineky(expr) __builtind_expect(!!(expr), true)
#endif
#ifndef unlikely
# define unlikely(expr) __builtin_expect(!!(expr), false)
#endif
#ifndef unreachable
# define unreachable() __builtin_unreachable()
#endif

static bool g_enable_hubble = true;
static constexpr char g_help_message[] =
	"help for StarSky\n"
	"\n"
	"usage:\n"
	"./sky (width) (height) (nr_stars) [options]\n"
	"\n"
	"  --help\t\tprint this help message and quit\n"
	"  --file=<filename>\tsave output image to file, instead prints binary\n"
	"                   \toutput content to stdout\n"
	"  --no-dust\t\tidisable dust drawing (small, distant, dim stars)\n"
	"  --no-hubble\t\tdisable hubble diffraction crosses drawing (white \n"
	"             \t\tcrossees around stars)\n"
	"  --no-milkyway\t\tdisable milkyway galaxy trail drawing\n"
	"\n"
	"(c) tlucanti\t:\tgithub.com/tlucanti\n";

__noret __cold
static void panic(const std::string &reason)
{
	std::cerr << "[panic]: " << reason << std::endl;
	std::cerr.flush();
	std::abort();
}

static unsigned int rand_in_range(unsigned int start, unsigned int end)
{
	return std::rand() % (end - start + 1) + start;
}

static float rand_in_range(float start, float end)
{
	return static_cast<float>(std::rand()) / RAND_MAX * (end - start) + start;
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

__used
std::ostream &operator <<(std::ostream &out, const Point &p)
{
	return out << '(' << p.x << ", " << p.y << ')';
}

static Point random_unit(float from, float to)
{
	float angle = rand_in_range(from, to);
	return Point(std::cos(angle), std::sin(angle));
}

static Point random_unit(void)
{
	return random_unit(0.f, 2 * std::numbers::pi);
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
		if (unlikely(x >= w or y >= h)) {
			panic("Matrix::get: out of bounds");
			unreachable();
		}

		return data.at(y * w + x);
	}

	const T &get(unsigned int x, unsigned int y) const
	{
		if (unlikely(x >= w or y >= h)) {
			panic("Matrix::get: out of bounds");
			unreachable();
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
		if (unlikely(px >= w or py >= h)) {
			panic("Image::set_pix: out of bounds");
			unreachable();
		}

		Matrix::get(px, py) = intensity;
	}

	void set_pix_safe(unsigned int px, unsigned int py, unsigned int intensity)
	{
		if (unlikely(px >= w or py >= h)) {
			return;
		}

		Matrix::get(px, py) = intensity;
	}

	void add_pix(unsigned int px, unsigned int py, unsigned int intensity)
	{
		if (unlikely(px >= w or py >= h)) {
			panic("Image::set_pix: out of bounds");
			unreachable();
		}

		Matrix::get(px, py) += intensity;
	}

	void add_pix_safe(unsigned int px, unsigned int py, unsigned int intensity)
	{
		if (unlikely(px >= w or py >= h)) {
			return;
		}

		Matrix::get(px, py) += intensity;
	}


	unsigned int get_pix(unsigned int px, unsigned int py)
	{
		if (unlikely(px >= w or py >= h)) {
			panic("Image::get_pix: out of bounds");
			unreachable();
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

static float catet(float hypot, float catet)
{
	return std::sqrt(hypot * hypot - catet * catet);
}

static float uniform_cos(unsigned int x, unsigned int max, float p)
{
	float xv, c;

	xv = static_cast<float>(x) / static_cast<float>(max);
	c = std::cos((2.f * xv - 1) * std::numbers::pi);
	return std::pow((c + 1.f) / 2.f, p);
}

static float norm(float x, float w)
{
	return std::exp(-w * x * x);
}

static float square(float x)
{
	return x * x;
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
	if (g_enable_hubble) {
		hubble_cross(im, x, y, r * 3, intensity);
	}
}

static void create_stars(Image &im, int nr_stars)
{
	unsigned int star_x, star_y, star_r;
	int type;

	for (int i = 0; i < nr_stars; ++i) {
		star_x = rand_in_range(0, im.get_w() - 1);
		star_y = rand_in_range(0, im.get_h() - 1);
		star_r = rand_in_range(1, std::min(im.get_h(), im.get_w()));

		type = i * 100.f / nr_stars;

		if (type < 95.f) {
			star_r /= 100;
		} else if (type < 98.f) {
			star_r /= 30;
		} else {
			star_r /= 20;
		}

		create_star(im, star_x, star_y, star_r, 1.f);
	}
}

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

static float perlin_dot(const Matrix<Point> &grid, int ix, int iy, float x, float y)
{
	Point g = grid.get(ix, iy);
	Point p(x - ix, y - iy);

	return dot(g, p);
}

static float perlin_pix(const Matrix<Point> &grid, float x, float y)
{
	Point p(x, y);
	int x0 = static_cast<int>(floor(x));
	int x1 = x0 + 1;
	int y0 = static_cast<int>(floor(y));
	int y1 = y0 + 1;
	float v1, v2, v3, v4, v12, v34;

	v1 = perlin_dot(grid, x0, y0, x, y);
	v2 = perlin_dot(grid, x1, y0, x, y);
	v3 = perlin_dot(grid, x0, y1, x, y);
	v4 = perlin_dot(grid, x1, y1, x, y);

	v12 = std::lerp(v1, v2, x - x0);
	v34 = std::lerp(v3, v4, x - x0);

	return std::lerp(v12, v34, y - y0);
}

static void perlin_octave(Matrix<float> &im, unsigned frequency, float amplitude)
{
	Matrix<Point> grid(frequency + 1, frequency + 1);
	float fx, fy, c;

	for (unsigned int y = 0; y <= frequency; ++y) {
		for (unsigned int x = 0; x <= frequency; ++x) {
			grid.get(x, y) = random_unit();
		}
	}

	for (unsigned y = 0; y < im.get_h(); ++y) {
		for (unsigned x = 0; x < im.get_w(); ++x) {
			fx = static_cast<float>(x) * frequency / im.get_w();
			fy = static_cast<float>(y) * frequency / im.get_h();
			if (im.get_w() > im.get_h()) {
				fy /= static_cast<float>(im.get_w() / im.get_h());
			} else {
				fx /= static_cast<float>(im.get_h() / im.get_w());
			}

			c = (perlin_pix(grid, fx, fy) + 1.f) / 2.f * amplitude;
			im.get(x, y) += c;
		}
	}

}

static void perlin(Matrix<float> &im, unsigned octaves)
{
	unsigned frequency = 8;
	float amplitude = 0.5f;
	Point direction;

	if (unlikely(octaves > 7)) {
		panic("create_milkyway: too many octaves");
		unreachable();
	}

	while (octaves--) {
		perlin_octave(im, frequency, amplitude);
		frequency *= 2.f;
		amplitude /= 2;
	}
}

static void create_milkyway(Image &im, unsigned octaves, float alpha)
{
	Matrix<float> milkyway(im.get_w(), im.get_h());
	Point direction, center, delta;
	float c, du, dh, r;
	const float pi = std::numbers::pi;

	perlin(milkyway, octaves);

	center.x = milkyway.get_w() / 2.f;
	center.y = milkyway.get_h() / 2.f;
	direction = random_unit(pi / 6.f, pi / 4.f);

	for (unsigned y = 0; y < milkyway.get_h(); ++y) {
		for (unsigned x = 0; x < milkyway.get_w(); ++x) {
			delta.x = x - center.x;
			delta.y = y - center.y;

			du = dot(delta, direction);
			dh = catet(length(delta), du);

			du /= std::max(im.get_w(), im.get_h());
			dh /= std::max(im.get_w(), im.get_h());

			r = norm(du, 2.f) * norm(dh, 32.f) * alpha;
			c = r * square(milkyway.get(x, y));

			im.add_pix(x, y, static_cast<unsigned char>(c));
		}
	}
}

struct config {
	unsigned width;
	unsigned height;

	unsigned nr_stars;

	bool enable_hubble;
	bool enable_dust;
	bool enable_milkyway;

	std::ostream *out;
	bool stdout;
};

static int to_int(const std::string &s)
{
	std::stringstream ss(s);
	int i;

	ss >> i;
	return i;
}

static void parse_argv(int argc, char **argv, struct config &config)
{
	std::string s;
	int i;

	for (i = 1; i < argc; ++i) {
		if (std::string(argv[i]) == "--help") {
			std::cerr << g_help_message << std::endl;
			exit(0);
		}
	}

	if (argc < 4) {
		std::cerr << "expected arguments, use '--help' for help\n";
		std::exit(1);
	}

	config.enable_hubble = true;
	config.enable_dust = true;
	config.enable_milkyway = true;
	config.out = &std::cout;
	config.stdout = true;

	config.width = to_int(argv[1]);
	config.height = to_int(argv[2]);
	config.nr_stars = to_int(argv[3]);

	for (i = 4; i < argc; ++i) {
		s = argv[i];

		if (s.starts_with("--file=")) {
			std::string file = s.substr(7);
			auto out = new std::ofstream(file);
			config.stdout = false;
			config.out = out;
		} else if (s == "--no-hubble") {
			config.enable_hubble = false;
			g_enable_hubble = false;
		} else if (s == "--no-dust") {
			config.enable_dust = false;
		} else if (s == "--no-milkyway") {
			config.enable_milkyway = false;
		} else {
			std::cerr << "unknown option '" << s <<
				"', try '--help' for help" << std::endl;
			std::exit(1);
		}
	}
}

int main(int argc, char **argv)
{
	struct config config;

	parse_argv(argc, argv, config);
	Image im(config.width, config.height);

	std::srand(std::time(nullptr));
	create_stars(im, config.nr_stars);
	if (config.enable_dust) {
		create_dust(im, 0.05f);
	}
	if (config.enable_milkyway) {
		create_milkyway(im, 4, 255.f);
	}

	im.to_pgm(*config.out);

	if (not config.stdout) {
		delete config.out;
	}
}

