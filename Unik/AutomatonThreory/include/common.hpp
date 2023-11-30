
#ifndef COMMON_HPP
#define COMMON_HPP

#include <string>
#include <iostream>
#include <iomanip>
#include <cstdlib>

__attribute__((__cold__, __noreturn__))
static inline void panic(const std::string &s)
{
	std::cerr << "PANIC: " << s << std::endl;
	std::exit(1);
}

static inline unsigned int powerof2(unsigned int x)
{
	return !(x & (x - 1));
}

static inline unsigned int ilog2(unsigned int x)
{
	unsigned ret = 0;

	if (x == 0) {
		panic("log of zero");
	}

	while (x >>= 1) {
		++ret;
	}
	return ret;
}

#endif /* COMMON_HPP */

