
#ifndef TABLE_HPP
#define TABLE_HPP

#include <TableItem.hpp>

#include <vector>

class Table {
public:
	Table(const std::vector<std::vector<int>> &transition);

	void dump(void);

private:
	std::vector<std::vector<TableItem>> table;
	size_t rows;
	size_t cols;
};

#endif /* TABLE_HPP */
