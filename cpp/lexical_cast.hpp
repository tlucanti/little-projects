/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   lexical_cast.hpp                                   :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: tlucanti <tlucanti@student.42.fr>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2022/02/19 17:09:44 by tlucanti          #+#    #+#             */
/*   Updated: 2022/02/20 18:06:26 by tlucanti         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef LEXICAL_CAST_HPP
# define LEXICAL_CAST_HPP

# include <string>
# include <climits>
# include "defs.h"

# define _NAMESPACE_START	namespace tlucanti {
# define _NAMESPACE_END		} /* tlucanti */

# ifndef ULONG_LONG_MAX
#  define ULONG_LONG_MAX ((unsigned long long)(-1))
# endif
# ifndef LONG_LONG_MAX
#  define LONG_LONG_MAX (ULONG_LONG_MAX / 2)
# endif
# ifndef LONG_LONG_MIN
#  define LONG_LONG_MIN (LONG_LONG_MAX + 1)
# endif

# define __INT_STR(__x) #__x
# define __INT_STR_INTERNAL(__x) __INT_STR(__x)
# define __MIN_LL_STR __INT_STR_INTERNAL(LONG_LONG_MIN)

_NAMESPACE_START

	class bad_lexical_cast : public std::bad_cast
	{
	public:
		bad_lexical_cast(const std::string &message) noexcept
				: _message(message) {}

		__WUR inline const char *what() const noexcept override
		{ return _message.c_str(); }

	protected:
		std::string _message;
	};

	class lexical_bad_symbol : public bad_lexical_cast
	{
	public:
		lexical_bad_symbol() noexcept
				: bad_lexical_cast("invalid integer literal") {}
	protected:
		lexical_bad_symbol(const std::string &message)
				: bad_lexical_cast(message) {}
	};

	class lexical_bad_base_symbol : public lexical_bad_symbol
	{
	public:
		lexical_bad_base_symbol() noexcept
				: lexical_bad_symbol("invalid base digit") {}
	};

	class lexical_unsigned_negative : public lexical_bad_symbol
	{
	public:
		lexical_unsigned_negative() noexcept
				: lexical_bad_symbol("unsigned value cannot be negative") {}
	};

	class lexical_overflow : public bad_lexical_cast
	{
	public:
		lexical_overflow() noexcept
				: bad_lexical_cast("value too big for target type") {}
	protected:
		lexical_overflow(const std::string &message)
				: bad_lexical_cast(message) {}
	};

	class lexical_underflow : public lexical_overflow
	{
	public:
		lexical_underflow() noexcept
				: lexical_overflow("value too low for target type") {}
	};

	enum lexical_cast_errors
	{
		ok				= 0b0000,
		bad_symbol		= 0b0110,
		bad_base_symbol	= 0b0111,
		neg_unsigned	= 0b0100,
		overflow		= 0b1000,
		underflow		= 0b1001,
	};

	__WUR inline
	long long int
	s2ll(const char *str, lexical_cast_errors *error_ptr, int base)
	/*
		string to integer converter
	*/
	{
		long long int	ret_val	= 0;
		unsigned int	sign	= 0;

		assert(base >= 2 and base <= 32);
		if (error_ptr)
			*error_ptr = ok;
		while (isspace(*str))
			++str;
		while (*str == '+' or *str == '-')
			if (*str++ == '-')
				sign ^= 1;
		while (isalnum(*str))
		{
			long long int i;
			if (isdigit(*str))
				i = *str - 48; // '0' = 48
			else if (islower(*str))
				i = *str - 87; // 'a' = 87 + 0xA = 97
			else
				i = *str - 55; // 'A' = 55 + 0xA = 65
			if (i >= base)
			{
				if (error_ptr)
					*error_ptr = bad_base_symbol;
				return 0;
			}
			ret_val = ret_val * base + i;
			if (ret_val < 0)
			{
				if (error_ptr)
					*error_ptr = (lexical_cast_errors)(overflow | sign); // overflow, or underflow if sign == 1
				return LONG_LONG_MAX + sign; // MAX if sign == 1 else MAX + 1 = MIN
			}
			++str;
		}
		if (*str == 0)
			return ret_val * (((1 - sign) << 1) - 1);
		else if (error_ptr)
			*error_ptr = bad_symbol;
		return 0;
	}

	__WUR inline
	unsigned long long int
	s2ull(const char *str, lexical_cast_errors *error_ptr, int base)
	/*
		string to unsigned integer converter
	*/
	{
		unsigned long long int ret_val = 0;
		unsigned long long int old_value;

		assert(base >= 2 and base <= 32);
		if (error_ptr)
			*error_ptr = ok;
		while (isspace(*str))
			++str;
		while (*str == '+')
			++str;
		if (*str == '-')
		{
			if (error_ptr)
				*error_ptr = neg_unsigned;
			return 0;
		}
		while (isalnum(*str))
		{
			unsigned long long int i;
			if (isdigit(*str))
				i = *str - 48; // '0' = 48
			else if (islower(*str))
				i = *str - 87; // 'a' = 87 + 0xA = 97
			else
				i = *str - 55; // 'A' = 55 + 0xA = 65
			if (i >= base)
			{
				if (error_ptr)
					*error_ptr = bad_base_symbol;
				return 0;
			}
			old_value = ret_val;
			ret_val = ret_val * base + i;
			if ((ret_val - i) / base != old_value)
			{
				if (error_ptr)
					*error_ptr = overflow;
				return ULONG_LONG_MAX;
			}
			++str;
		}
		if (*str == 0)
			return ret_val;
		else if (error_ptr)
			*error_ptr = bad_symbol;
		return 0;
	}


	template <typename target_T, int base=10, typename source_T>
	__WUR inline
	target_T
	lexical_cast(const source_T &_source, bool no_overflow=false)
	/*
		convert from string to integer
	*/
	{
		static_assert(base >= 2 and base <= 32, "base should be in range [2:32]");
		static_assert(
				// signed integer
				tlucanti::is_same<target_T, char>::value or
				tlucanti::is_same<target_T, short int>::value or
				tlucanti::is_same<target_T, int>::value or
				tlucanti::is_same<target_T, long int>::value or
				tlucanti::is_same<target_T, long long int>::value or
				// unsigned integer
				tlucanti::is_same<target_T, unsigned char>::value or
				tlucanti::is_same<target_T, unsigned short int>::value or
				tlucanti::is_same<target_T, unsigned int>::value or
				tlucanti::is_same<target_T, unsigned long int>::value or
				tlucanti::is_same<target_T, unsigned long long int>::value
				, "type not supported");

		lexical_cast_errors error = ok;
		target_T ret;
		std::string source = _source;
		if (tlucanti::is_same<target_T, char>::value or
			tlucanti::is_same<target_T, short int>::value or
			tlucanti::is_same<target_T, int>::value or
			tlucanti::is_same<target_T, long int>::value or
			tlucanti::is_same<target_T, long long int>::value)
			/*
				signed integet cast
			*/
		{
			long long int ret_val = s2ll(source.c_str(), &error, base);
			if (ret_val > std::numeric_limits<target_T>::max())
			{
				error = overflow;
				ret = std::numeric_limits<target_T>::max();
			}
			else if (ret_val < std::numeric_limits<target_T>::min())
			{
				error = underflow;
				ret = std::numeric_limits<target_T>::min();
			}
			else
				ret = static_cast<target_T>(ret_val);
		}
		else
			/*
				unsigned integer
			*/
		{
			unsigned long long int ret_val = s2ull(source.c_str(), &error, base);
			if (ret_val > std::numeric_limits<target_T>::max())
			{
				error = overflow;
				ret = std::numeric_limits<target_T>::max();
			}
			else
				ret = static_cast<target_T>(ret_val);
		}
		if (error == ok)
			return ret;
		else if (error == bad_symbol)
			throw lexical_bad_symbol();
		else if (error == bad_base_symbol)
			throw lexical_bad_base_symbol();
		else if (error == overflow and not no_overflow)
			throw lexical_overflow();
		else if (error == underflow and not no_overflow)
			throw lexical_underflow();
		else if (error == neg_unsigned)
			throw lexical_unsigned_negative();
		throw bad_lexical_cast("[bad_cast_error]: unknown error");
	}

	inline
	char *
	ull2s(unsigned long long int val, char *dest, int base, int upper)
	{
		assert(base >= 2 and base <= 32);
		static const char *_lower = "0123456789abcdefghijklmnopqrstuvwxyz";
		static const char *_upper = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
		const char *alpha = _lower;

		if (val == 0)
		{
			*dest = '0';
			dest[1] = 0;
			return dest;
		}
		char buf[65];
		int i = 64;
		buf[i] = 0;
		if (upper)
			alpha = _upper;
		while (val)
		{
			buf[--i] = alpha[val % base];
			val /= base;
		}
		return (char *)memmove(dest, buf + i, 65 - i);
	}

	inline
	char *
	ll2s(long long int val, char *dest, int base, int upper)
	{
		assert(base >= 2 and base <= 32);

		if (val == LONG_LONG_MIN)
			return strcpy(dest, __MIN_LL_STR);
		if (val < 0)
		{
			*dest = '-';
			ull2s(-val, dest + 1, base, upper);
			return dest;
		}
		return ull2s(val, dest, base, upper);
	}

	template <typename target_T, int base=10, bool upper=true, typename source_T>
	__WUR inline
	target_T
	numeric_cast(const source_T &source, bool no_overflow=false)
	/*
		convert from string to integer
	*/
	{
		static_assert(base >= 2 and base <= 32, "base should be in range [2:32]");
		static_assert(
				// signed integer
				tlucanti::is_same<source_T, char>::value or
				tlucanti::is_same<source_T, short int>::value or
				tlucanti::is_same<source_T, int>::value or
				tlucanti::is_same<source_T, long int>::value or
				tlucanti::is_same<source_T, long long int>::value or
				// unsigned integer
				tlucanti::is_same<source_T, unsigned char>::value or
				tlucanti::is_same<source_T, unsigned short int>::value or
				tlucanti::is_same<source_T, unsigned int>::value or
				tlucanti::is_same<source_T, unsigned long int>::value or
				tlucanti::is_same<source_T, unsigned long long int>::value
				, "type not supported");

		char *buf[66];
		std::string ret;
		if (tlucanti::is_same<source_T, char>::value or
			tlucanti::is_same<source_T, short int>::value or
			tlucanti::is_same<source_T, int>::value or
			tlucanti::is_same<source_T, long int>::value or
			tlucanti::is_same<source_T, long long int>::value)
		/*
			signed integer cast
		*/
		{
			long long s = source;
			ll2s(s, reinterpret_cast<char *>(buf), base, upper);
		}
		else
		/*
			unsigned integer cast
		*/
		{
			unsigned long long s = source;
			ull2s(s, reinterpret_cast<char *>(buf), base, upper);
		}
		return reinterpret_cast<char *>(buf);
	}

_NAMESPACE_END
#endif /* LEXICAL_CAST_HPP */

