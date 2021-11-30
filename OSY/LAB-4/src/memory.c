/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   memory.c                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: kostya <kostya@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2021/09/08 11:58:13 by kostya            #+#    #+#             */
/*   Updated: 2021/11/30 21:24:10 by kostya           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "../inc/memory.h"
#include "../inc/error.h"

__NOEXC __WUR __MALLOC
void	*xmalloc(size_t size)
/*
** function allocates memory or terminates program if allocation faled
*/
{
	register void	*ptr;

	ptr = malloc(size);
	if (!ptr)
	{
		ft_perror("malloc", errno, NULL);
		ft_exit(1);
	}
	return (ptr);
}

__NOEXC
void	free_s(void *__restrict ptr)
{
	free(*(void **)ptr);
	*(void **)ptr = NULL;
}
