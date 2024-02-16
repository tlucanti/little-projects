
#include <vector>
#include <list>
#include <iostream>

template <class T>
void bubble_sort(T begin, T end)
{
    if (begin == end) {
	    return ;
    }

    std::size_t d = std::distance(begin, end);
    for (std::size_t i = 0; i < d - 1; ++i) {
	    for (std::size_t j = 0; j < d - i - 1; ++j) {
		    T a(begin);
		    std::advance(a, j);
		    T b = std::next(a);

		    if (*a > *b) {
			    std::swap(*a, *b);
		    }
	    }
    }
}

int main() {
	int a[] = {5, 4, 3, 2, 1};
	std::vector<int> b = {5, 4, 3, 2, 1};
	std::list<int> c = {5, 4, 3, 2, 1};

	bubble_sort(a, a + 5);
	for (int i = 0; i < 5; ++i) {
		std::cout << a[i] << ' ';
	}
	std::cout << '\n';

	bubble_sort(b.begin(), b.end());
	for (auto i : b) {
		std::cout << i << ' ';
	}
	std::cout << '\n';

	bubble_sort(c.begin(), c.end());
	for (auto i : c) {
		std::cout << i << ' ';
	}
	std::cout << '\n';
}

