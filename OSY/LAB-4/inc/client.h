/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   client.h                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: kostya <kostya@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2021/11/30 19:49:55 by kostya            #+#    #+#             */
/*   Updated: 2021/11/30 23:46:44 by kostya           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef CLIENT_H
# define CLIENT_H

# include "lab4.h"

typedef struct	s_shared_led
{
	sig_atomic_t	new_data;
	e_button		button;
	int				socket;
}					t_shared_led;

#endif
