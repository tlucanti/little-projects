
#include <vector>
#include <numeric>
#include <algorithm>
#include <iterator>
#include <iostream>

int main()
{
	std::vector<char> v(13);
	std::iota(v.begin(), v.end(), 'a' - v.size() + 1);
	std::reverse(v.begin(), v.end());

	std::copy(v.begin(), v.end(), std::ostream_iterator<char>(std::cout, " "));
}
