/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   error.c                                            :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: kostya <kostya@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2021/09/09 20:34:33 by kostya            #+#    #+#             */
/*   Updated: 2021/11/29 19:38:08 by kostya           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "error.h"
#include "color.h"

static const char	*ft_strerror(int errorcode) __NOEXC __WUR;

__NOEXC
void	ft_message(unsigned int type, const char *__restrict parent,
	int errorcode, const char *__restrict message)
{
	const char	*type_chr;

	if (type == ERROR_TOKEN)
		type_chr = ERROR"[FAIL]";
	else if (type == WARNING_TOKEN)
		type_chr = WARNING"[WARN]";
	else if (type == OK_TOKEN)
		type_chr = OK"[ OK ]";
	else if (type == INFO_TOKEN)
		type_chr = INFO"[INFO]";
	else
		type_chr = "[INTERNAL_TOKEN_ERROR]";
	printf("%s%s ", type_chr, RESET);
	if (parent)
		printf("%s%s:%s ", TERM_WHITE, parent, RESET);
	printf("%s", ft_strerror(errorcode));
	if (message)
		printf(": %s%s%s\n", WARNING, message, RESET);
	else
		printf("\n");
}

__NOEXC __WUR
static const char	*ft_strerror(int errorcode)
{
	if (errorcode == 0)
		return ("");
	else if (errorcode == E_BADOPTION)
		return ("Invalid option");
	else if (errorcode == E_BADPERIOD)
		return ("Invalid period value");
	else if (errorcode == E_BADARG)
		return ("Invalid arguments");
	else if (errorcode == K_EXIT)
		return ("exit");
	else if (errorcode == K_PERIODCH)
		return ("period changed to");
	else if (errorcode == I_HELP)
		return ("Try " TERM_WHITE "'./lab3 " TERM_YELLOW "--help'" RESET
			" for more information");
	return (strerror(errorcode));
}

__NOEXC
void	ft_perror(const char *__restrict parent, int errorcode,
	const char *__restrict message)
{
	ft_message(ERROR_TOKEN, parent, errorcode, message);
}

__NOEXC
void	ft_warning(const char *__restrict parent, int errorcode,
	const char *__restrict message)
{
	ft_message(WARNING_TOKEN, parent, errorcode, message);
}

__NOEXC
void	ft_info(const char *__restrict parent, int errorcode,
	const char *__restrict message)
{
	ft_message(INFO_TOKEN, parent, errorcode, message);
}

__NOEXC
void	ft_ok(const char *__restrict parent, int errorcode,
	const char *__restrict message)
{
	ft_message(OK_TOKEN, parent, errorcode, message);
}

__NOEXC __NORET
void	print_help_message(void)
{
	printf(
		"Operation Systems LAB-3 implementation program\n"
		"Usage: lab3 [OPTION] [PERIOD]\n"
		"\n"
		"after run - program read from STDIN for parameter [PERIOD]\n"
		"  and changes it when Enter pressed\n"
		"\n"
		"Options: ([OPTION] part)\n"
		"\t-h, --help\tprint this help message\n"
		"\n"
		"Parameters (other arguments)\n"
		"\t[PERIOD]\tset period of led blink\n"
		"\n"
		"Exit status\n"
		"\t0\tif OK\n"
		"\tnon 0\tif error ocured\n"
		"\n"
		"Examples\n"
		"  ./lab3 0.5\n"
		"  ./lab3 2\n"
		"\n"
		"Sources avaliable in:"
		"  <https://github.com/antikostya/little-projects/tree/main/OSY/LAB-3\n"
		"\n"
		"(OSY-LAB-3 v0.1)\t(C) " TLUCANTI "tlucanti" RESET "\n"
	);
	ft_exit(0);
}

__NOEXC __NORET
void	ft_exit(int status)
{
	exit(status);
}
