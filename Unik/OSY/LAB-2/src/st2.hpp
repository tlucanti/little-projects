/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   st2.cpp                                            :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: kostya <kostya@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2021/12/20 20:41:10 by kostya            #+#    #+#             */
/*   Updated: 2021/12/21 12:59:29 by kostya           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

namespace tlucanti
{
	class string
	{
	public:
		string();
		explicit string(const char *st);
		explicit string(char c);
		explicit string(const string &st);
		explicit string(const std::string &st);
		explicit string(string &&st);

		~string();

		string	&operator =(const char *st);
		string	&operator =(char c);
		string	&operator =(const string &st);
		string	&operator =(const std::string &st);
		string	&operator =(string &&op);

		bool	operator ==(const char *st) const;
		bool	operator ==(char c) const;
		bool	operator ==(const string &st) const;
		bool	operator ==(const std::string &st) const;

		bool	operator !=(const char *st) const;
		bool	operator !=(char c) const;
		bool	operator !=(const string &st) const;
		bool	operator !=(const std::string &st) const;

		bool	operator <(const char *st) const;
		bool	operator <(char c) const;
		bool	operator <(const string &st) const;
		bool	operator <(const std::string &st) const;

		bool	operator >(const char *st) const;
		bool	operator >(char c) const;
		bool	operator >(const string &st) const;
		bool	operator >(const std::string &st) const;

		bool	operator <=(const char *st) const;
		bool	operator <=(char c) const;
		bool	operator <=(const string &st) const;
		bool	operator <=(const std::string &st) const;

		bool	operator >=(const char *st) const;
		bool	operator >=(char c) const;
		bool	operator >=(const string &st) const;
		bool	operator >=(const std::string &st) const;

		string	operator  +(const char *st) const;
		string	operator  +(char c) const;
		string	operator  +(const string &st) const;
		string	operator  +(const std::string &st) const;

		string	&operator +=(const char *st);
		string	&operator +=(char c);
		string	&operator +=(const string &st);
		string	&operator +=(const std::string &st);

		string	operator  *(size_t n) const;

		string	&operator *=(size_t n);

		size_t	size() const;
		char	*c_str() const;
		size_t	hash() const;
		bool	cmp(const string &st) const;

	private:
		size_t	_alloc_size;
		size_t	_size;
		mutable size_t	_hash
		mutable char	*_container = nullptr;
		static double	_phi = 1.6180339887'4989484820'4586834365'6381177203;

		string(const char *container, size_t size, size_t alloc_size);

		void	__init__(const char *st);
		void	__move__(string &&st);
		void	__del__();

		int		__cmp__(const char *st) const;
		string	&__add__(const char *st) const;
		void	__iadd__(const char *st);
		string	&__mul__(const char *st, size_t n) const;
		void	__imul__(const char *st, size_t n);

		static bool _check_u64mul_overflow(size_t x, double y, size_t *o) const;
		static bool _check_u64mul_overflow(size_t x, size_t y, size_t *o) const;
		static bool _check_u64add_overflow(size_t x, size_t y, size_t *o) const;
	}
}
