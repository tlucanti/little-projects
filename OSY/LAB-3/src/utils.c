/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   utils.c                                            :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: kostya <kostya@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2021/11/29 16:24:08 by kostya            #+#    #+#             */
/*   Updated: 2021/11/29 19:52:41 by kostya           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "../inc/lab3.h"

static double  ft_atod_noexp(const char **str, double *sgn) __NOEXC __WUR;
static double  ft_powd_fast(double n, int exp) __NOEXC __WUR;
static double  ft_atod_frac(const char **str__) __NOEXC __WUR;

__NOEXC __WUR
char			*str2double(const char *str, double *number)
/*
** converts [str] to double, save result in [number]
** return pointer to next symbol after end of number
*/
{
	double ans;
	double sgn;

	if (str == NULL)
		return (NULL);
	while (isspace(*str))
		str++;
	ans = ft_atod_noexp(&str, &sgn);
	if (*str == '.')
		ans += ft_atod_frac(&str);
	if (*str == 'e' || *str == 'E')
		*number = sgn * ans * ft_powd_fast(10, atoi(++str));
	else
		*number = sgn * ans;
	if (str[-1] == 'e' || str[-1] == 'E')
	{
		if (*str == '-' || *str == '+')
			++str;
		while (isdigit(*str))
			++str;
	}
	return ((char *)str);
}

__NOEXC __WUR
static double	ft_atod_noexp(const char **str, double *sgn)
{
	double ans;

	ans = 0;
	*sgn = 1;
	if (**str == '-')
		*sgn = -1;
	if (**str == '-' || **str == '+')
		(*str)++;
	while (isdigit(**str))
		ans = ans * (double)10 + (double)(*(*str)++ - 48);
	return (ans);
}

__NOEXC __WUR
static double	ft_powd_fast(double n, int exp)
{
	double nn;

	if (exp == 0)
		return (1);
	else if (exp < 0)
		return ((double)1 / ft_powd_fast(n, -exp));
	else if (exp % 2)
		return (n * ft_powd_fast(n, exp - 1));
	else
	{
		nn = ft_powd_fast(n, exp / 2);
		return (nn * nn);
	}
}

__NOEXC __WUR
static double	ft_atod_frac(const char **str__)
{
	char        *ptr;
	char        *str;
	double ans;

	ans = 0;
	ptr = *((char **)str__);
	str = ptr;
	++ptr;
	while (isdigit(*ptr))
		ptr++;
	*str__ = ptr;
	ptr -= 1;
	while (str != ptr)
	{
		ans = ans * (double)0.1 + (double)(*ptr-- - 48);
	}
	return (ans * (double)0.1);
}
