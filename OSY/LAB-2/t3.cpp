/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   t3.cpp                                             :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: kostya <kostya@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2021/12/20 15:35:09 by kostya            #+#    #+#             */
/*   Updated: 2021/12/20 20:35:11 by kostya           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include <vector>
#include "string.hpp"
#include "lab2.hpp"

//using tlucanti::input;
//using tlucanti::print;

[[noreturn]]
int		main()
{
	std::vector<tlucanti::string>	s;

	while (true)
	{
		s = tlucanti::input().split();
		tlucanti::print(s);
	}
}
