/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   lab4.h                                             :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: kostya <kostya@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2021/11/30 19:47:25 by kostya            #+#    #+#             */
/*   Updated: 2021/12/15 16:56:22 by kostya           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef LAB4_H
# define LAB4_H

# include <sys/socket.h>
# include <arpa/inet.h>
# include <stdio.h>
# include <unistd.h>
# include <pthread.h>
# include <string.h>
# include <ctype.h>
# include <errno.h>
# include <signal.h>

typedef enum	s_button
{
	up		= 0x0,
	down	= 0x1,
	left	= 0x2,
	right	= 0x3,
	var_1	= 0x4,
	var_2	= 0x5,

	led_count = 0x6,

	led1	= 0x6,
	led2	= 0x7,
	led3	= 0x8,
	led4	= 0x9,
	led5	= 0xa,
	led6	= 0xb
}				e_button;

typedef struct	s_shared_led
{
	sig_atomic_t	new_data;
	e_button		button;
	unsigned int	leds;
	pthread_mutex_t	led_mutex;
}					t_shared_led;

char	*str2int(char *str, int *number);

# define __UNUSED	__attribute__((unused))
# define __WUR		__attribute__((warn_unused_result))
# define __NOEXC	__attribute__((__nothrow__))
# define __NORET	__attribute__((noreturn))
# define __INLINE	__attribute__((always_inline))

# define REQUEST_MESSAGE_SIZE	1024

# define or		||
# define and	&&
# define not	!

#endif
