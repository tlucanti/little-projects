/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   string.hpp                                         :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: kostya <kostya@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2021/12/20 15:36:05 by kostya            #+#    #+#             */
/*   Updated: 2021/12/20 20:35:12 by kostya           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef STRING_HPP
# define STRING_HPP

# include <string>
# include <vector>
# include <sstream>

namespace tlucanti
{
	class string {
	public:
		string();
		explicit string(const std::string &st);
		explicit string(const char st[]);

		[[nodiscard]] std::vector<string> split();
		[[nodiscard]] std::string std_str() const;

		template<typename STL_Tp>
		[[nodiscard]] string join(const STL_Tp &iter);

		string &operator =(const char *st);
		string &operator =(const std::string &st);
		string &operator =(char st);
		string &operator +=(const char *st);
		string &operator +=(const std::string &st);
		string &operator +=(char st);


		friend std::ostream &
		operator <<(std::ostream &out, const tlucanti::string &st);

	private:
		std::string _internal_string;
	};
}

#endif  // STRING_HPP
