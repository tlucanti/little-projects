
#ifndef _UTIL_HPP_
#define _UTIL_HPP_

#include <iostream>
#include <stdexcept>

using namespace std::literals;

#define panic_on(expr, msg)                               \
	do {                                              \
		if (expr) [[unlikely]] {                  \
			__panic(msg, __FILE__, __LINE__); \
		}                                         \
	} while (false);

#define panic(msg) panic_on(true, msg)

[[noreturn]]
static inline void __panic(const std::string &s, const std::string &file, unsigned long line)
{
	std::cerr << "\n[PANIC]: " << file << ':' << line << ": reason:\n"
		  << s << std::endl;
	std::abort();
}

class TranslationError : public std::runtime_error {
public:
	TranslationError(const std::string &s) : std::runtime_error(s) {}
};

class TranslationEOF : public std::exception {
public:
	TranslationEOF() : std::exception() {}
};

static inline void err_on(bool expr, const std::string &msg)
{
	if (expr) [[unlikely]] {
		throw TranslationError(msg);
	}
}

static inline void err(const std::string &msg)
{
	err_on(true, msg);
}

#endif /* _UTIL_HPP_ */

