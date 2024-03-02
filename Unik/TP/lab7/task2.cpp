
#include <vector>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <iostream>

int main()
{
	int n = 10;
	int x0 = 1, y0 = 2, z0 = 3;
	std::vector<int> x(n), y(n), z(n);

	std::generate(x.begin(), x.end(), [](){ return rand() % 200 - 100; });
	std::generate(y.begin(), y.end(), [](){ return rand() % 200 - 100; });
	std::generate(z.begin(), z.end(), [](){ return rand() % 200 - 100; });
	std::copy(x.begin(), x.end(), std::ostream_iterator<int>(std::cout, " "));
	std::cout << '\n';
	std::copy(y.begin(), y.end(), std::ostream_iterator<int>(std::cout, " "));
	std::cout << '\n';
	std::copy(z.begin(), z.end(), std::ostream_iterator<int>(std::cout, " "));
	std::cout << "\n\n";


	x.insert(x.begin(), x0);
	y.insert(y.begin(), y0);
	z.insert(z.begin(), z0);
	std::vector<int> resx(x.size());
	std::vector<int> resy(x.size());
	std::vector<int> resz(x.size());

	std::partial_sum(x.begin(), x.end(), resx.begin());
	std::partial_sum(y.begin(), y.end(), resy.begin());
	std::partial_sum(z.begin(), z.end(), resz.begin());

	std::cout << x0 << ' ' << y0 << ' ' << z0 << '\n';

	std::copy(++resx.begin(), resx.end(), std::ostream_iterator<int>(std::cout, " "));
	std::cout << '\n';
	std::copy(++resy.begin(), resy.end(), std::ostream_iterator<int>(std::cout, " "));
	std::cout << '\n';
	std::copy(++resz.begin(), resz.end(), std::ostream_iterator<int>(std::cout, " "));
	std::cout << '\n';
}
