/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   error.h                                            :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: kostya <kostya@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2021/09/09 20:40:07 by kostya            #+#    #+#             */
/*   Updated: 2021/12/16 15:35:05 by kostya           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef ERROR_H
# define ERROR_H

# include <stdio.h>
# include <stdlib.h>
# include <string.h>
# include "lab4.h"

void	ft_perror(const char *__restrict parent, int errorcode, const char *__restrict message) __NOEXC;
void	ft_warning(const char *__restrict parent, int errorcode, const char *__restrict message) __NOEXC;
void	ft_ok(const char *__restrict parent, int errorcode, const char *__restrict message) __NOEXC;
void	ft_info(const char *__restrict parent, int errorcode, const char *__restrict message) __NOEXC;
void	ft_exit(int status) __NORET __NOEXC;
void	print_help_message(void) __NORET __NOEXC;

extern	pthread_mutex_t		print_mutex;

// errrors
typedef enum	s_error
{
	E_CLIENT_WRITE		= -1,
	E_CLIENT_READ		= -2,
	E_RESPONSE_FORMAT	= -3,
	E_BUTTON_VALUE		= -4,
	E_CLIENT_BADARG		= -5,

	W_UNKNOWN_REQUEST	= -6,

	K_BUTTON_CHANGE		= -7,
	K_LED_CHANGE		= -8,

	I_CLIENT_INFO		= -9,
	I_SERVER_INFO		= -10

}				e_error;

// oks
# define K_EXIT			(-4)
# define K_PERIODCH		(-5)

// infos
# define I_HELP			(-6)

#endif
