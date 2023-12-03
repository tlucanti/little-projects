
#ifndef COMMON_HPP
#define COMMON_HPP

#include <string>
#include <iostream>
#include <iomanip>
#include <cstdlib>

#define panic(msg) __panic(msg, __FILE__, __LINE__)

__attribute__((__cold__, __noreturn__))
static inline void __panic(const std::string &s, const std::string &file, size_t line)
{
	std::cout << std::endl;
	std::cerr << file << ':' << line << "\n";
	std::cerr << "PANIC: " << s << std::endl;
	std::abort();
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

