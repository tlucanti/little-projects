
#include <Table.hpp>

Table::Table(const std::vector<std::vector<int>> &transition)
{
	unsigned val, maxw, v;

	rows = transition.size();
	cols = transition.front().size();
	for (size_t r = 0; r < rows; ++r) {
		if (cols != transition.at(r).size()) {
			panic("invalid input shape");
		}

		for (size_t c = 0; c < cols; ++c) {
			v = transition.at(r).at(c);
			maxw = std::max(maxw, v);
		}
	}
	maxw = ilog2(maxw);

	table.resize(transition.size());
	for (size_t r = 0; r < rows; ++r) {
		table.at(r).reserve(cols);

		for (size_t c = 0; c < cols; ++c) {
			val = transition.at(r).at(c);
			table.at(r).emplace_back(val, maxw);
		}
	}
}

void Table::dump(void)
{
	for (size_t r = 0; r < rows; ++r) {
		for (size_t c = 0; c < cols; ++c) {
			std::cout << table.at(r).at(c).val.val << ' ';
		}
		std::cout << '\n';
	}
	std::cout << '\n';
}

