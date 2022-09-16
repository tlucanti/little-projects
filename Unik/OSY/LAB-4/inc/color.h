/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   color.h                                            :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: kostya <kostya@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2021/09/09 20:23:37 by kostya            #+#    #+#             */
/*   Updated: 2021/11/29 19:00:43 by kostya           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef COLOR_H
# define COLOR_H

# define RED			0x00ff0000
# define GREEN			0x0000ff00
# define BLUE			0x000000ff
# define YELLOW			0x00ffff00
# define PURPLE			0x00ff00ff
# define CYAN			0x0000ffff
# define WHITE			0x00ffffff
# define BLACK			0x00000000

# define TERM_BLACK		"\e[1;90m"
# define TERM_RED		"\e[1;91m"
# define TERM_GREEN		"\e[1;92m"
# define TERM_YELLOW	"\e[1;93m"
# define TERM_BLUE		"\e[1;94m"
# define TERM_PURPLE	"\e[1;95m"
# define TERM_CYAN		"\e[1;96m"
# define TERM_WHITE		"\e[1;97m"
# define TERM_RESET		"\e[0m"
# define TLUCANTI		"\033[19;1;96;21;6m"

# define OK				TERM_GREEN
# define WARNING		TERM_YELLOW
# define ERROR			TERM_RED
# define INFO			TERM_BLUE
# define RESET			TERM_RESET

# define OK_TOKEN		1
# define WARNING_TOKEN	2
# define ERROR_TOKEN	3
# define INFO_TOKEN		4

#endif
