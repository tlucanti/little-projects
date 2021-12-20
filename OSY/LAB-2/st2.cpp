/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   st2.cpp                                            :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: kostya <kostya@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2021/12/20 20:41:10 by kostya            #+#    #+#             */
/*   Updated: 2021/12/21 01:39:20 by kostya           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

namespace tlucanti
{
	class string
	{
	public:
		string();
		string(const char *st);
		string(char c);
		string(const string &st);
		string(const std::string &st);
		string(string &&st);

		~string();

		string	&operator =(const char *st);
		string	&operator =(char c);
		string	&operator =(const string &st);
		string	&operator =(const std::string &st);
		string	&operator =(string &&op);

		bool	operator ==(const char *st);
		bool	operator ==(char c);
		bool	operator ==(const string &st);
		bool	operator ==(const std::string &st);

		bool	operator !=(const char *st);
		bool	operator !=(char c);
		bool	operator !=(const string &st);
		bool	operator !=(const std::string &st);

		bool	operator <(const char *st);
		bool	operator <(char c);
		bool	operator <(const string &st);
		bool	operator <(const std::string &st);

		bool	operator >(const char *st);
		bool	operator >(char c);
		bool	operator >(const string &st);
		bool	operator >(const std::string &st);

		bool	operator <=(const char *st);
		bool	operator <=(char c);
		bool	operator <=(const string &st);
		bool	operator <=(const std::string &st);

		bool	operator >=(const char *st);
		bool	operator >=(char c);
		bool	operator >=(const string &st);
		bool	operator >=(const std::string &st);

		string	operator  +(const char *st);
		string	operator  +(char c);
		string	operator  +(const string &st);
		string	operator  +(const std::string &st);

		string	&operator +=(const char *st);
		string	&operator +=(char c);
		string	&operator +=(const string &st);
		string	&operator +=(const std::string &st);

		size_t	size();
		char	*c_str();
		size_t	hash();
		bool	cmp(const string &st);

	private:
		size_t	_alloc_size;
		size_t	_size;
		size_t	_hash
		mutable char	*_container = nullptr;
		static double	_phi = 1.6180339887'4989484820'4586834365'6381177203;

		string(const char *container, size_t size, size_t alloc_size);

		void	__init__(const char *st);
		void	__move__(string &&st);
		void	__del__();

		int		__cmp__(const char *st);
		string	&__add__(const char *st);
		void	__iadd__(const char *st);
		string	&__mul__(const char *st, size_t n);
		void	__imul__(const char *st, size_t n);

	}
}
