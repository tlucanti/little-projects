
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <cmath>

template <class T>
class ProdCounter {
	T prod;
	int n;
public:
	ProdCounter() { prod = {}; n = 0; ++prod; }
	void operator()(const T &t) {
		T i = {};
		while (i * i <= t) {
			if (i * i == t) {
				n++;
				prod *= t;
				return;
			}
			++i;
		}
	}
	bool ok(void) const { return n != 0; }
	float compute(void) const { return std::pow((float)prod, 1.f / n); }
};

int main()
{
	srandom(time(NULL));

	std::vector<int> v(random() % 100);
	std::generate(v.begin(), v.end(), [](){ return rand() % 200 - 100; });
	std::cout << "vector: ";
	std::copy(v.begin(), v.end(), std::ostream_iterator<int>(std::cout, " "));

	int prod = 1;
	int n = 0;
	std::for_each(v.begin(), v.end(),
		[&prod, &n](int x) {
			int i = 0;
			while (i * i <= x) {
				if (i * i == x) {
					prod *= x;
					n++;
					return;
				}
				++i;
			}
		});

	if (n == 0) {
		std::cout << "\nno square numbers\n";
	} else {
		std::cout << "\ncomputed: " << std::pow((float)prod, 1.f / n) << '\n';
	}

	ProdCounter<int> c;
	c = std::for_each(v.begin(), v.end(), c);
	if (!c.ok()) {
		std::cout << "no square numbers\n";
	} else {
		std::cout << "computed: " << c.compute() << '\n';
	}


}

