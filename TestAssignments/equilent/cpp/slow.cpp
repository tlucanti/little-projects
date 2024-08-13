
#include <cmath>
#include <iostream>
#include <vector>

#ifndef SIZE
# define SIZE 100000
#endif

#ifndef WINDOW
# define WINDOW 10
#endif


struct Angles {
	double a;
	double b;
};

struct argm {
	double val;
	int idx;
};

struct win_state {
	struct argm max;
	struct argm min;
};

/**
 * Intuitive implementation:
 * For each sliding window compute angles of all points from this window and
 * chose maximum for upper line and minimum for lower line.
 * Time complexity:
 *  - rolling window O(n), n = len(input)
 *  - each window takes O(w) iterations, w = window width
 * total: O(n * w)
 */
static void calc(const std::vector<double> &inputs, std::vector<Angles> &outputs)
{
	struct win_state window;

	outputs.at(0) = {0, 0};

	for (int i = 1; i < (int)inputs.size(); i++) {
		// update sliding window max and min
		int t = std::max(i - WINDOW + 1, 0);

		window.max = { inputs.at(t), t };
		window.min = { inputs.at(t), t };

		for (t++; t < i; t++) {
			if (std::atan2(inputs.at(t) - inputs.at(i), i - t) > std::atan2(window.max.val - inputs.at(i), i - window.max.idx)) {
				window.max = { inputs.at(t), t };
			}
			if (std::atan2(inputs.at(t) - inputs.at(i), i - t) < std::atan2(window.min.val - inputs.at(i), i - window.min.idx)) {
				window.min = { inputs.at(t), t };
			}
		}

		// compute angles for this index
		double dy, dx;

		dy = window.max.val - inputs.at(i);
		dx = i - window.max.idx;
		outputs.at(i).a = std::atan2(dy, dx);

		dy = window.min.val - inputs.at(i);
		dx = i - window.min.idx;
		outputs.at(i).b = std::atan2(dy, dx);
	}
}

int main()
{
	std::vector<double> inputs(SIZE);
	std::vector<Angles> outputs(SIZE);

	for (int i = 0; i < SIZE; i++) {
		std::cin >> inputs.at(i);
	}

	calc(inputs, outputs);

	for (const auto &ang : outputs) {
		std::cout << ang.a << ',' << ang.b << '\n';
	}
}

