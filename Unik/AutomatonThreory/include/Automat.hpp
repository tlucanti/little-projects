
#ifndef TABLE_HPP
#define TABLE_HPP

#include <Table.hpp>
#include <Impl.hpp>

class Automat {
public:
	Automat(const Table &transition, const std::vector<Impl> &states);
	Automat(const Table &transition, const Table &states);

private:
	Table transition;
	Table states;
};

#endif /* TABLE_HPP */

