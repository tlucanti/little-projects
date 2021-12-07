/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   main.c                                             :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: kostya <kostya@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2021/11/29 19:35:46 by kostya            #+#    #+#             */
/*   Updated: 2021/12/07 19:58:45 by kostya           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "../inc/error.h"
#include "../inc/lab3.h"

volatile sig_atomic_t	g_usr_sig = 0;

static void	run_led(long long int period) __NOEXC;
static void	handler_sigint(__UNUSED int signum) __NOEXC;
static void handler_sigusr1(__UNUSED int signum) __NOEXC;

__NOEXC
int		main(int argc, char **__restrict argv)
{
	double	period = 0;

	if (argc != 2)
	{
		ft_perror("lab3", E_BADARG, NULL);
		ft_info("lab3", I_HELP, NULL);
		ft_exit(1);
	}
	if (argv[1][0] == '-')
	{
		if (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help"))
			print_help_message();
		else
		{
			ft_perror("lab3", E_BADOPTION, argv[1]);
			ft_info("lab3", I_HELP, NULL);
			ft_exit(1);
		}
	}
	else
	{
		if (*str2double(argv[1], &period) != 0)
		{
			ft_perror("lab3", E_BADPERIOD, argv[1]);
			ft_info("lab3", I_HELP, NULL);
			ft_exit(1);
		}
	}
	run_led((long long int)(period * 500000));
	return 0;
}

__NOEXC
static void	run_led(long long int period)
{
	pid_t	p_id;
	int		pipes[2];

	if (pipe(pipes))
	{
		ft_perror("pipe", errno, NULL);
		ft_exit(1);
	}
	p_id = fork();
	if (p_id == -1)
	{
		ft_perror("fork", errno, NULL);
		ft_exit(1);
	}
	else if (p_id == 0)
	{
		close(pipes[1]);
		signal(SIGINT, handler_sigint);
		signal(SIGUSR1, handler_sigusr1);
		while (1)
		{
			if (g_usr_sig)
			{
				double	new_period;
				if (read(pipes[0], &new_period, sizeof(double))
						!= sizeof(double))
					ft_perror("read", errno, NULL);
				period = new_period * 500000;
				g_usr_sig = 0;
			}
			usleep(period);
			swap_led();
		}
	}
	else
	{
		char	buff[256] = {};
		double	new_period = 0;

		close(pipes[0]);
		while (1)
		{
			if (scanf("%255s", buff) != 1)
				ft_perror("lab3", E_BADPERIOD, NULL);
			else if (*str2double(buff, &new_period) != 0)
				ft_perror("lab3", E_BADPERIOD, buff);
			else
			{
				kill(p_id, SIGUSR1);
				if (write(pipes[1], &new_period, sizeof(double))
						!= sizeof(double))
					ft_perror("write", errno, NULL);
				ft_ok("lab3", K_PERIODCH, buff);
			}
		}
	}
}

__NOEXC
static void	handler_sigint(__UNUSED int signum)
{
	off_led();
	ft_ok("lab3", K_EXIT, "successfully");
	ft_exit(0);
}

__NOEXC
static void handler_sigusr1(__UNUSED int signum)
{
	g_usr_sig = 1;
}
