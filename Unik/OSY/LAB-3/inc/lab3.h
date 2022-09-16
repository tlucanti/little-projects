/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   lab3.h                                             :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: kostya <kostya@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2021/11/29 15:03:58 by kostya            #+#    #+#             */
/*   Updated: 2021/11/29 19:27:30 by kostya           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef LAB3_H
# define LAB3_H

# include <string.h>
# include <unistd.h>
# include <ctype.h>
# include <stdlib.h>
# include <signal.h>
# include <errno.h>
# include <sys/types.h>
# include <sys/wait.h>
# include <stdio.h>


# define __UNUSED	__attribute__((unused))
# define __WUR		__attribute__((warn_unused_result))
# define __NOEXC	__attribute__((__nothrow__))
# define __NORET	__attribute__((noreturn))

char	*str2double(const char *str, double *number) __NOEXC __WUR;
void	swap_led(void) __NOEXC;
void	on_led(void) __NOEXC;
void	off_led(void) __NOEXC;

#endif
