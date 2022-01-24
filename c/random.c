/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   random.c                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: kostya <kostya@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2022/01/22 18:52:02 by kostya            #+#    #+#             */
/*   Updated: 2022/01/22 18:52:49 by kostya           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include <stdlib.h>

int random(int start, int stop)
{
	int		ret;
	do {
		ret = rand();
	} while (ret >= RAND_MAX - RAND_MAX % (stop - start + 1));
	return ret % (stop - start + 1) + start;
}
