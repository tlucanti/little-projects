/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   utils.cpp                                          :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: kostya <kostya@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2021/12/21 13:03:16 by kostya            #+#    #+#             */
/*   Updated: 2021/12/21 13:07:01 by kostya           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

# include <iostream>

namespace tlucanti
{
	void print(const char *st)
	{
		std::cout << st << std::endl;
	}

	void print(const string &st)
	{
		std::cout << st.c_str();
	}
}
