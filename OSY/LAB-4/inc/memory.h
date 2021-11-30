/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   memory.h                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: kostya <kostya@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2021/09/05 22:35:26 by kostya            #+#    #+#             */
/*   Updated: 2021/11/30 21:24:24 by kostya           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef MEMORY_H
# define MEMORY_H

# include <malloc.h>
# include <errno.h>

# ifndef __MALLOC
#  define __MALLOC __attribute__((malloc))
# endif
# ifndef __NOEXC
#  define __NOEXC __attribute__((__nothrow__))
# endif
# ifndef __WUR
#  define __WUR __attribute__((warn_unused_result))
# endif

void	*xmalloc(size_t size) __NOEXC __WUR __MALLOC;
void	free_s(void *ptr) __NOEXC;

#endif // MEMORY_H
