
#ifndef DNF_HPP
#define DNF_HPP

#include <Impl.hpp>
#include <vector>

class DNF
{
private:
	std::string input;
	std::vector<Impl> data;

	int step();

public:
	DNF(const std::string &input);
	~DNF();

	void minimize(std::ostream &out=std::cout);
	void dump(void);
	void table(void);
};

#endif /* DNF_HPP */

