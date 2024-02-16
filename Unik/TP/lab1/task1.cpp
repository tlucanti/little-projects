
#include <vector>
#include <list>
#include <iostream>

template <typename BidirectionalIterator>
void bubble_sort(BidirectionalIterator begin, BidirectionalIterator end) {
    if (begin == end) return;

    bool swapped = true;

    while(begin != end-- && swapped) {
        swapped = false;
        for(auto i = begin; i != end; ++i) {
            if(*std::next(i) < *i) {
                std::swap(*i, *std::next(i));
                swapped = true;
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

