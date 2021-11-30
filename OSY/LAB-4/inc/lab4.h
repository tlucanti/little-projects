/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   lab4.h                                             :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: kostya <kostya@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2021/11/30 19:47:25 by kostya            #+#    #+#             */
/*   Updated: 2021/11/30 21:29:38 by kostya           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef LAB4_H
# define LAB4_H

# include <sys/socket.h>
# include <arpa/inet.h>
# include <stdio.h>
# include <unistd.h>
# include <pthread.h>

# define __UNUSED	__attribute__((unused))
# define __WUR		__attribute__((warn_unused_result))
# define __NOEXC	__attribute__((__nothrow__))
# define __NORET	__attribute__((noreturn))

# define REQUEST_MESSAGE_SIZE	256
# define RESPONSE_MESSAGE_SIZE	256

#endif
