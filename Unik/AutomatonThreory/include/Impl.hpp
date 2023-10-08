
#ifndef IMPL_HPP
#define IMPL_HPP

#include <iostream>

class Impl
{
public:
	unsigned int num;
	unsigned int ind;
	unsigned int p;
	bool pw;
	bool inf;

	Impl(unsigned int num, bool inf);
	Impl(unsigned int num, unsigned int ind, unsigned int p, bool pw, bool inf);
	bool operator ==(const Impl &b);
	~Impl();
	static bool can_patch(const Impl &a, const Impl &b);
	static Impl patch(Impl &a, Impl &b);
};

inline std::ostream &operator <<(std::ostream &out, const Impl &i)
{
	out << i.num;
	return out;
}

#endif /* IMPL_HPP */

