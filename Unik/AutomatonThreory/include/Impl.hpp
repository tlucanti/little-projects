
#ifndef IMPL_HPP
#define IMPL_HPP

#include <iostream>

struct Impl
{
	unsigned int num;
	unsigned int ind;
	unsigned int p;
	bool pw;
	bool inf;

	Impl(unsigned int n, bool inf);
	Impl(unsigned int n, unsigned int ind, unsigned int p, bool pw, bool inf);
	~Impl();
	static bool can_patch(const Impl &a, const Impl &b);
	static Impl patch(Impl &a, Impl &b);
};

inline std::ostream &operator <<(std::ostream &out, const Impl &i)
{
	out << i.num << std::endl;
	return out;
}

#endif /* IMPL_HPP */

