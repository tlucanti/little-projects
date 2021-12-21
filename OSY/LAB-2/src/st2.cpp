/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   st2.hpp                                            :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: kostya <kostya@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2021/12/20 20:56:45 by kostya            #+#    #+#             */
/*   Updated: 2021/12/21 12:58:49 by kostya           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "st2.hpp"

namespace tlucanti
{
	string::string()
	{
		print("called default empty constructor");
		__init__("");
	}

	string::string(const char *st)
	{
		print("called default [char *] constructor");
		__init__(st);
	}

	string::string(char c)
	{
		print("called default [char] constructor");
		char	st[2] = {c, 0};
		__init__(st);
	}

	string::string(const string &st)
	{
		print("called [string &] copy constructor");
		__init__(st._container);
	}

	string::string(const std::string &st)
	{
		print("called [std::string &] copy constructor");
		__init__(st.c_str());
	}

	string::string(string &&st)
	{
		print("called [string &&] move constructor");
		__move__(st);
	}

	string::string(const char *container, size_t size, size_t alloc_size)
		: _alloc_size(alloc_size), _size(size), _hash(0), _container(container)
		{}

	string::~string()
	{
		print("called default destructor");
		delete[_container];
	}

	string &
	string::operator =(const char *st)
	{
		print("called [char *] assign operator");
		__init__(st);
		return *this;
	}

	string &
	string::operator =(char c)
	{
		print("called [char] assign operator");
		char	st[2] = {c, 0};
		__init__(st);
		return *this;
	}

	string &
	string::operator =(const string &st)
	{
		print("called [string &] assign operator");
		__init__(st._container);
		return *this;
	}

	string &
	string::operator =(const std::string &st)
	{
		print("called [std::string &] assign operator");
		__init__(st.c_str());
		return *this;
	}

	string &
	string::operator =(string &&st)
	{
		print("called [string &&] move assign operator");
		__move__(st);
		return *this;
	}

	bool
	operator ==(const char *st) const
	{
		return st == _container || std::memcmp(_container, st, _size + 1);
	}

	bool
	operator ==(char c) const
	{
		return _size == 1 && _container[0] == c;
	}

	bool
	operator ==(const string &st) const
	{
		if (_hash == 0)
			__hash__();
		return _hash == st.hash();
	}

	bool
	operator ==(const std::string &st) const
	{
		return strcmp(_container, st.c_str());
	}

	bool
	operator !=(const char *st) const
	{
		return !(*this == st);
	}

	bool
	operator !=(char c) const
	{
		return !(*this == c);
	}

	bool
	operator !=(const string &st) const
	{
		return !(*this == st);
	}

	bool
	operator !=(const std::string &st) const
	{
		return !(*this == st);
	}

	bool
	operator <(const char *st) const
	{
		return __cmp__(st) < 0;	
	}

	bool
	operator <(char c) const
	{
		const char st[2] = {c, 0};
		return __cmp__(st) < 0;	
	}

	bool
	operator <(const string &st) const
	{
		return __cmp__(st._container) < 0;	
	}

	bool
	operator <(const std::string &st) const
	{
		return __cmp__(st.c_str()) < 0;	
	}

	bool
	operator >(const char *st) const
	{
		return __cmp__(st) > 0;	
	}

	bool
	operator >(char c) const
	{
		const char st[2] = {c, 0};
		return __cmp__(st) > 0;	
	}

	bool
	operator >(const string &st) const
	{
		return __cmp__(st._container) > 0;	
	}

	bool
	operator >(const std::string &st) const
	{
		return __cmp__(st.c_str()) > 0;	
	}

	bool
	operator <=(const char *st) const
	{
		return __cmp__(st) <= 0;	
	}

	bool
	operator <=(char c) const
	{
		const char st[2] = {c, 0};
		return __cmp__(st) <= 0;	
	}

	bool
	operator <=(const string &st) const
	{
		return __cmp__(st._container) <= 0;	
	}

	bool
	operator <=(const std::string &st) const
	{
		return __cmp__(st.c_str()) <= 0;	
	}

	bool
	operator >=(const char *st) const
	{
		return __cmp__(st) >= 0;	
	}

	bool
	operator >=(char c) const
	{
		const char st[2] = {c, 0};
		return __cmp__(st) >= 0;	
	}

	bool
	operator >=(const string &st) const
	{
		return __cmp__(st._container) >= 0;	
	}

	bool
	operator >=(const std::string &st) const
	{
		return __cmp__(st.c_str()) >= 0;	
	}

	string
	operator  +(const char *st) const
	{
		return __add__(st, strlen(st));
	}

	string
	operator  +(char c) const
	{
		const char st[2] = {c, 0}
		return __add__(st, 1);
	}

	string
	operator  +(const string &st) const
	{
		return __add__(st._container, st._size);
	}

	string
	operator  +(const std::string &st) const
	{
		return __add__(st.c_str());
	}


	string &
	operator  +=(const char *st)
	{
		__iadd__(st, strlen(st));
		return *this;
	}
	
	string &
	operator  +=(char c)
	{
		const char st[2] = {c, 0}
		__iadd__(st, 1);
		return *this;
	}
	
	string &
	operator  +=(const string &st)
	{
		__iadd__(st._container, st._size);
		return *this;
	}
	
	string &
	operator  +=(const std::string &st)
	{
		__iadd__(st.c_str());
		return *this;
	}

	string	operator  *(size_t) const
	{
		return __mul__(n);
	}

	string	&operator *=(size_t n)
	{
		return __imul__(n);
	}

	void
	__init__(const char *st)
	{
		__del__();
		_size = std::strlen(st);
		_alloc_size = _size * _phi;
		_hash = 0;
		_container = new char[_alloc_size];
		std::memcpy(_container, st, _size);
	}

	void
	__move__(string &&st)
	{
		_size = st.size;
		_alloc_size = st._alloc_size;
		_hash = st._hash;
		_container = st._container;
		st._size = 0;
		st._alloc_size = 0;
		st._hash = 0;
		st._container = nullptr;
	}

	void
	__del__()
	{
		_size = 0;
		_alloc_size = 0;
		delete [_container];
	}

	int
	__cmp__(const char *st) const
	{
		return std::memcmp(_container, st._container, _size + 1);
	}

	string &
	__add__(const char *st, size_t z) const
	{
		char	*new_container;
		size_t	expected_alloc_size;
		size_t	new_alloc_size;
		size_t	alloc_size;

		if (_check_u64add_overflow(_size, z, &expected_alloc_size))
			throw runtime_error("strings are too big");
		if (_check_u64mul_overflow(expected_alloc_size, _phi, &new_alloc_size))
			alloc_size = expected_alloc_size;
		else
			alloc_size = new_alloc_size;

		new_container = new char[alloc_size];
		std::memcpy(new_container, _container, _size);
		std::memcpy(new_container + _size, st, z);
		return string(new_container, expected_alloc_size, alloc_size);
	}

	void
	__iadd__(const char *st, size_t z)
	{
		size_t	expected_alloc_size;

		if (_check_u64add_overflow(_size, z, &expected_alloc_size))
			throw runtime_error("strings are too big");
		if (expected_alloc_size > _alloc_size)
		{
			size_t	new_alloc_size;
			char	*new_container;

			if (_check_u64mul_overflow(_size + z, _phi, &new_alloc_size))
				_alloc_size = expected_alloc_size;
			else
				_alloc_size = new_alloc_size;

			new_container = new char[_alloc_size];
			std::memcpy(new_container, _container, _size);
			delete [_container];
			_container = new_container;
		}
		std::memcpy(_container + _size, st, z);
		_size = expected_alloc_size;
	}

	string &
	__mul__(size_t n) const
	{
		char	*new_container;
		size_t	expected_alloc_size;
		size_t	new_alloc_size;
		size_t	alloc_size;

		if (_check_u64add_overflow(_size, n, &new_alloc_size))
			throw runtime_error("strings are too big");
		if (_check_u64mul_overflow(expected_alloc_size, _phi, &new_alloc_size))
			alloc_size = expected_alloc_size;
		else
			alloc_size = new_alloc_size;
		new_container = new char[alloc_size]
		for (size_t i=0; i < n; ++i)
			std::memcpy(new_container + _size * i, _container, _size);
		return string(new_container, expected_alloc_size, alloc_size)
	}
	
	void
	__imul__(size_t n)
	{
		size_t	expected_alloc_size;

		if (_check_u64mul_overflow(_size, n, &expected_alloc_size))
			throw runtime_error("strings are too big");
		if (expected_alloc_size > _size)
		{
			size_t	new_alloc_size;
			char	*new_container;

			if (_check_u64mul_overflow(expected_alloc_size, phi, &new_alloc_size))
				_alloc_size = expected_alloc_size;
			else
				_alloc_size = new_alloc_size;
			new_container = new char[_alloc_size];
			std::memcpy(new_container, _container, _size);
			delete [_container];
			_container = new_container;
		}
		for (size_t i=1; i < n; ++i)
			std::memcpy(_container + _size * i, _container, _size);
		_size = expected_alloc_size;
	}

	size_t
	string::size() const
	{
		return _size;
	}

	char *
	string::c_str() const
	{
		return _container;
	}

	size_t
	string::hash() const
	{
		if (_hash == 0)
			__hash__();
		return _hash;
	}

	bool
	string::cmp(const string &st) const
	{
		return __cmp__(st._container);
	}
}
