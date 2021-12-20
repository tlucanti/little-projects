/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   string.cpp                                         :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: kostya <kostya@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2021/12/20 15:35:37 by kostya            #+#    #+#             */
/*   Updated: 2021/12/20 20:41:52 by kostya           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "string.hpp"

tlucanti::string::string()
{
	_internal_string = "";
}

tlucanti::string::string(const std::string &st)
{
	_internal_string = st;
}

tlucanti::string::string(const char st[])
{
	_internal_string = st;
}

std::vector<tlucanti::string>
tlucanti::string::split()
{
	std::stringstream			ss(_internal_string);
	std::vector<string>			out;
	std::string					next;

	while (ss.eof())
	{
		ss >> next;
		out.emplace_back(next);
	}
	return out;
}

template<typename STL_Tp>
tlucanti::string
tlucanti::string::join(const STL_Tp &iter)
{
	auto		it = iter.begin();
	string		st;

	st += static_cast<std::string>(*it);
	for (; iter != iter.end(); ++iter)
	{
		st += _internal_string;
		st += static_cast<std::string>(*it);
	}
	return st;
}

tlucanti::string &
tlucanti::string::operator =(const char *st)
{
	_internal_string = st;
	return *this;
}

tlucanti::string &
tlucanti::string::operator =(const std::string &st)
{
	_internal_string = st;
	return *this;
}

tlucanti::string &
tlucanti::string::operator =(char st)
{
	_internal_string = "";
	_internal_string += st;
	return *this;
}

tlucanti::string &
tlucanti::string::operator +=(const char *st)
{
	_internal_string += st;
	return *this;
}

tlucanti::string &
tlucanti::string::operator +=(const std::string &st)
{
	_internal_string += st;
	return *this;
}

tlucanti::string &
tlucanti::string::operator +=(char st)
{
	_internal_string += st;
	return *this;
}

std::string tlucanti::string::std_str() const
{
	return _internal_string;
}

std::ostream &
operator <<(std::ostream &out, const tlucanti::string &st)
{
	out << st.std_str();
	return out;
}
