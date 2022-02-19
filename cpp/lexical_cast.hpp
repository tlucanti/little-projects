/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   lexical_cast.hpp                                   :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: tlucanti <tlucanti@student.42.fr>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2022/02/19 17:09:44 by tlucanti          #+#    #+#             */
/*   Updated: 2022/02/19 20:18:11 by tlucanti         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef LEXICAL_CAST_HPP
# define LEXICAL_CAST_HPP

# include <string>
# include <climits>

# define _NAMESPACE_START	namespace tlucanti {
# define _NAMESPACE_END		} /* tlucanti */

# if __cplusplus <= 199711L
#  define __WUR __attribute__((warn_unused_result))
#  define noexcept
# else
#  define __WUR [[nodiscard]]
# endif /* C++20 */

_NAMESPACE_START

	template<class T, class U>
	struct is_same : std::false_type {};

	template<class T>
	struct is_same<T, T> : std::true_type {};

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

	__WUR static inline
	long long int
	s2ll(const char *str, lexical_cast_errors *error_ptr, int base)
	{
		long long int	ret_val	= 0;
		unsigned int	sign	= 0;

		assert(base >= 1 and base <= 32);
		if (error_ptr)
			*error_ptr = ok;
		while (isspace(*str))
			++str;
		while (*str == '+' or *str == '-')
		{
			if (*str == '-')
				sign ^= 1;
			++str;
		}
		while (isalnum(*str))
		{
			long long int i;
			if (isdigit(*str))
				i = *str - 48; // '0' = 48
			else if (islower(*str))
				i = *str - 87; // 'a' = 87 + 0xA = 97
			else
				i = *str - 55; // 'A' = 55 + 0xA = 65
			if (i > base)
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
			return ret_val * (sign << 1) - 1;
		else if (error_ptr)
			*error_ptr = bad_symbol;
		return 0;
	}

	__WUR
	unsigned long long int
	s2ull(const char *str, lexical_cast_errors *error_ptr, int base)
	{
		unsigned long long int ret_val = 0;
		unsigned long long int old_value;

		assert(base >= 1 and base <= 32);
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
			if (i > base)
			{
				if (error_ptr)
					*error_ptr = bad_base_symbol;
				return 0;
			}
			old_value = ret_val;
			ret_val = ret_val * base;
			if (ret_val / base != old_value)
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


	template <typename target_T, int base>
	__WUR inline
	target_T
	lexical_cast(const std::string &source, bool no_overflow=false)
	{
		static_assert(base >= 1 and base <= 32, "base should be in range [1:32]");
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
		}
		if (error == bad_symbol)
			throw lexical_bad_symbol();
		else if (error == bad_base_symbol)
			throw lexical_bad_base_symbol();
		else if (error == overflow and not no_overflow)
			throw lexical_overflow();
		else if (error == underflow and not no_overflow)
			throw lexical_underflow();
		if (error == neg_unsigned)
			throw lexical_unsigned_negative();
		else // ok
			return static_cast<target_T>(ret);
	}

_NAMESPACE_END
#endif /* LEXICAL_CAST_HPP */
