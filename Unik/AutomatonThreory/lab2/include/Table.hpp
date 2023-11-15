
#ifndef TABLE_HPP
#define TABLE_HPP

#include <TableItem.hpp>

#include <vector>

class Table {
	std::vector<std::vector<TableItem>> table;

	Table(const std::vector<std::vector<int> > &transition,
	      const std::vector<std::vector<int> > &exit);
};

#endif /* TABLE_HPP */
