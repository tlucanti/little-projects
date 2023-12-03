
#ifndef TABLE_HPP
#define TABLE_HPP

#include <TableItem.hpp>
#include <vector>

class Table {
public:
	Table(const std::vector<std::vector<int>> &transition);
	Table();

	void dump(void) const;
	size_t get_rows(void) const;
	size_t get_cols(void) const;

private:
	std::vector<std::vector<TableItem>> table;
	size_t rows;
	size_t cols;
};

#endif /* TABLE_HPP */
