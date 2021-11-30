/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   utils.c                                            :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: kostya <kostya@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2021/11/30 23:07:39 by kostya            #+#    #+#             */
/*   Updated: 2021/12/01 00:20:57 by kostya           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

extern long int strtol(const char *nptr, char **endptr, int base);

char	*str2long(char *str, long int *number)
{
	char	*endptr;

	*number = strtol(str, &endptr, 10);
	return endptr;
}

char	*str2int(char *str, int *number)
{
	long	l_num;
	char	*ret;

	ret = str2long(str, &l_num);
	if (l_num > 2147483647L)
		*number = 2147483647;
	else if (l_num < -2147483648)
		*number = -2147483648;
	return ret;
}
