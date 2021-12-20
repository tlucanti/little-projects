/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   lab2.hpp                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: kostya <kostya@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2021/12/20 16:50:33 by kostya            #+#    #+#             */
/*   Updated: 2021/12/20 20:34:50 by kostya           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef LAB2_HPP
# define LAB2_HPP

# include <iostream>
# include "string.hpp"

namespace tlucanti
{
	template <typename STL_Tp>
	void	print(const STL_Tp &vec, const string &sep, const string &end)
	{
		string j = sep.join(vec);
		std::cout << j << end;
	}

 	template <typename STL_Tp>
	void	print(const STL_Tp &vec)
	{
		string sep(" ");
		string endl("\n");
		print(vec, sep, endl);
	}

	string	input(const std::string &message)
	{
		std::string	ret;

		std::cout << message;
		std::getline(std::cin, ret);
		return string(ret);
	}

	string	input()
	{
		return input("");
	}
}

#endif	// LAB2_HPP
