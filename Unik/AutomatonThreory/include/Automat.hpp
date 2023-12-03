
#ifndef AUTOMAT_HPP
#define AUTOMAT_HPP

#include <Table.hpp>

class Automat {
public:
	Automat(const Table &transition, const std::vector<int> &output);
	Automat(const Table &transition, const Table &output);

private:
	void init_automat(const Table &trnsition, const Table &output);

	Table transition;
	Table output;
};

#endif /* AUTOMAT_HPP */

