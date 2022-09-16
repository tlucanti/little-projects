/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   lab2.cpp                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: kostya <kostya@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2021/12/21 13:06:09 by kostya            #+#    #+#             */
/*   Updated: 2021/12/21 13:08:41 by kostya           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "string.hpp"

using tlucanti::string;

# define info(st) do {						\
	std::cout << s.c_str() << std::endl;	\
	std::cout << s.size() << std::endl;		\
} while (0)

int main()
{
	string s;
	info(s);
}
