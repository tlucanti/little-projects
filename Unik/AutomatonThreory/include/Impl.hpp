
#ifndef IMPL_HPP
#define IMPL_HPP

#include <iostream>

class Impl
{
private:
	unsigned int num;
	unsigned int ind;
	unsigned int p;
	bool pw;
	bool inf;

public:
	Impl(int n);
	~Impl();
	bool can_patch(const Impl &a, const Impl &b);
	Impl patch(Impl &a, Impl &b);

friend std::ostream &operator <<(std::ostream &out, const Impl &i);
};

std::ostream &operator <<(std::ostream &out, const Impl &i)
{
	out << i.num << std::endl;
	return out;
}

#endif /* IMPL_HPP */

