
#include <Table.hpp>

Table::Table()
{}

Table::Table(const std::vector<std::vector<int>> &transition)
{
	int val, maxw, v;

	rows = transition.size();
	cols = transition.front().size();
	for (size_t r = 0; r < rows; ++r) {
		if (cols != transition.at(r).size()) {
			panic("invalid input shape");
		}

		for (size_t c = 0; c < cols; ++c) {
			v = transition.at(r).at(c);
			if (v < 0) {
				continue;
			}
			maxw = std::max(maxw, v);
		}
	}
	std::cout << "MAXW: " << maxw << ' ' << ilog2(maxw) << '\n';
	if (maxw > 100) {
		std::cout << "err\n";
	}
	maxw = ilog2(maxw) + 1;

	table.resize(transition.size());
	for (size_t r = 0; r < rows; ++r) {
		table.at(r).reserve(cols);

		for (size_t c = 0; c < cols; ++c) {
			val = transition.at(r).at(c);
			table.at(r).emplace_back(val, maxw);
		}
	}
}

void Table::dump(bool binary) const
{
	for (size_t r = 0; r < rows; ++r) {
		for (size_t c = 0; c < cols; ++c) {
			TableItem v = table.at(r).at(c);
			if (binary) {
				std::cerr << v.bin(get_w());
			} else {
				std::cerr << v;
			}
			std::cerr << ' ';
		}
		std::cerr << '\n';
	}
	std::cerr << '\n';
}

size_t Table::get_rows(void) const
{
	return this->rows;
}

size_t Table::get_cols(void) const
{
	return this->cols;
}

int Table::get_w(void) const
{
	return table.at(0).at(0).get_w();
}

const TableItem &Table::at(size_t r, size_t c) const
{
	return table.at(r).at(c);
}

