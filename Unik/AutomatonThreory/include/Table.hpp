
#ifndef TABLE_HPP
#define TABLE_HPP

#include <TableItem.hpp>
#include <vector>

class Table {
public:
    Table(const std::vector<std::vector<int> > &transition);
    Table();

    void dump(bool binary = false) const;
    size_t get_rows(void) const;
    size_t get_cols(void) const;
    int get_w(void) const;
    const TableItem &at(size_t r, size_t c) const;

private:
	std::string _bin(unsigned int v, unsigned maxw);

	std::vector<std::vector<TableItem>> table;
	size_t rows;
	size_t cols;
};

#endif /* TABLE_HPP */
