
#include <fstream>
#include <string>
#include <cstdlib>

#define CONFIG_IMAGE_WIDTH 128
#define CONFIG_IMAGE_HEIGHT 128

__attribute__((__noreturn__, __cold__))
void panic(const std::string &reason)
{
	std::cout << "[panic]: " << reason << endl;
	std::abort();
}

class Image {
private:
	unsigned int w;
	unsigned int h;
	std::vector<unsigned char> data;

public:
	Image(unsigned int w, unsigned int h)
		: x(x), y(y)
	{
		data.resize(x * y);
	}

	void set_pix(unsigned int px, unsigned int py, unsigned char intensity)
	{
		if (px >= w or py >= h) {
			panic("Image::set_pix: out of bounds");
		}

		data.at(py * w + px) = intensity;
	}

	unsigned char get_pix(unsigned int px, unsigned int py)
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
		out << "255\n";
	}

	void print_data(std::ostream &out)
	{
		for (unsigned int y = 0; y < h; ++y) {
			for (unsigned int x = 0; x < w; ++x) {
				out << get_pix(x, y);
			}
			out << '\n';
		}
	}
};

int main()
{
	Image im(CONFIG_IMAGE_WIDTH, CONFIG_IMAGE_HEIGHT);

	for (int y = 0; y < im.get_h(); ++y) {
		for (int x = 0; x < im.get_w(); ++x) {
			im.set_pix(x, y, random());
		}
	}

	im.to_pgm(std::cout);
}

