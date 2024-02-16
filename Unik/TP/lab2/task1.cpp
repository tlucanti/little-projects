
#include <fstream>
#include <vector>
#include <iterator>
#include <cstdlib>
#include <ctime>
#include <algorithm>

int main()
{
	std::srand(std::time(NULL));
	std::vector<int> v(rand() % 1000);
	std::ofstream f("out.txt");
	for (auto &i : v) {
		i = rand() % 2000 - 1000;
	}

	if (not f.is_open()) {
		abort();
	}
	std::sort(v.begin(), v.end());
	std::copy(v.begin(), v.end(), std::ostream_iterator<int>(f, " "));
	f.close();
}
