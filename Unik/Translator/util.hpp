
#ifndef _UTIL_HPP_
#define _UTIL_HPP_

#include <iostream>

#define panic_on(expr, msg)                               \
	do {                                              \
		if (expr) [[unlikely]] {                  \
			__panic(msg, __FILE__, __LINE__); \
		}                                         \
	} while (false);

#define panic(msg) panic_on(true, msg)

[[noreturn]]
static void __panic(const std::string &s, const std::string &file, unsigned long line)
{
	std::cerr << "\n[PANIC]: " << file << ':' << line << ": reason:\n"
		  << s << std::endl;
	std::abort();
}

#endif /* _UTIL_HPP_ */

