
#ifndef COMMON_HPP
#define COMMON_HPP

#include <string>
#include <iostream>
#include <cstdlib>

[[cold, noreturn]]
static inline panic(const std::string &s)
{
	std::cerr << "PANIC: " << s << std::endl;
	std::abort();
}

#endif /* COMMON_HPP */

