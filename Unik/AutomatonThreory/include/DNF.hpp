
#ifndef DNF_HPP
#define DNF_HPP

#include <Impl.hpp>
#include <vector>

class DNF
{
private:
	std::vector<Impl> data;

	void step();

public:
	DNF(const std::string &input);
	~DNF();

	void minimize(std::ostream &out);
};

#endif /* DNF_HPP */

