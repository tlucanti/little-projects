/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   client.c                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: kostya <kostya@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2021/11/29 22:35:59 by kostya            #+#    #+#             */
/*   Updated: 2021/12/08 15:19:42 by kostya           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "../inc/lab4.h"
#include "../inc/client.h"
#include "../inc/error.h"

static int		init_socket(char *ip) __NOEXC __WUR;
static void		*run_led(t_shared_led *shared_led) __NOEXC;
static int		parse_argv(int argc, char ** __restrict argv) __NOEXC;
static void		send_leds(int sock, unsigned int leds) __NOEXC;
static void		parse_data(char *response, t_shared_led *shared_led) __NOEXC;
int		isbutton(e_button button) __NOEXC __WUR __INLINE;

__NOEXC __NORET
int		main(int argc, char ** __restrict argv)
{
	int				sock;
	char			request[REQUEST_MESSAGE_SIZE];
	char			response[RESPONSE_MESSAGE_SIZE];
	t_shared_led	shared_led;
	pthread_t		pthread;

	sock = parse_argv(argc, argv);
	shared_led.socket = sock;
	shared_led.new_data = 0;
	pthread_create(&pthread, NULL, (void *(*)(void *))(void *)run_led,
		(void *)&shared_led);
	sprintf(request, "client [%d]: waiting for response", getpid());

	while (1)
	{
		ssize_t		bytes;

		printf("sending data to server: <%s>\n", request);
		bytes = printf("%s\n", request);
		if (bytes <= 0)
		{
			ft_perror("client", E_CLIENT_WRITE, NULL);
			continue ;
		}
		bytes = read(sock, response, RESPONSE_MESSAGE_SIZE);
		if (bytes <= 0)
		{
			ft_perror("client", E_CLIENT_READ, NULL);
			continue ;
		}
		printf("read data from server (%zdB):\n<%s>\n", bytes, response);
		parse_data(response, &shared_led);
	}
}

__NOEXC __WUR
static int		init_socket(char *ip)
{
	struct sockaddr_in	server;
	int					sock;

	sock = socket(AF_INET, SOCK_STREAM, 0);
	if (sock == -1)
	{
		ft_perror("socket", errno, "cannot create socket");
		ft_exit(1);
	}

	server.sin_family = AF_INET;
	server.sin_port = htons(8080);

	if (inet_pton(AF_INET, ip, &server.sin_addr) == -1)
	{
		ft_perror("ip", errno, ip);
		ft_exit(1);
	}
	if (connect(sock, (struct sockaddr *)&server, sizeof(server)))
	{
		ft_perror("connect", errno, "cannot connect to server");
		ft_exit(1);
	}
	return sock;
}

__NOEXC __NORET
static void		*run_led(t_shared_led *shared_led)
{
	unsigned int	leds;
	int				period;
	short			variant;
	e_button		direction;


	leds = 14 & 0b111111;
	send_leds(shared_led->socket, leds);
	variant = 0;
	period = 500000;
	while (1)
	{
		if (shared_led->new_data)
		{
			if (shared_led->button == up)
				period /= 2;
			else if (shared_led->button == down)
				period *= 2;
			else if (shared_led->button == left)
				direction = left;
			else if (shared_led->button == right)
				direction = right;
			else if (shared_led->button == var_1)
				variant = 1;
			else if (shared_led->button == var_2)
				variant = 2;
			else
				leds = leds ^ (0x1 << (shared_led->button - 6));
		}
		if (direction == left)
		{
			if (variant == 1)
				leds = (leds << 1 & 0b111111) | leds >> 5;
			else if (variant == 2)
				leds = (leds - 1) & 0b111111;
		}
		else if (direction == right)
		{
			if (variant == 0)
				leds = (leds >> 1) | ((leds & 0x1) << 5);
			else
				leds = (leds + 1) & 0b111111;
		}
		send_leds(shared_led->socket, leds);
		usleep(period);
	}
}

__NOEXC
static void		send_leds(int sock, unsigned int leds)
{
	static char		pattern_on[11]  = "LEDx: green";
	static char		pattern_off[11] = "LEDx: grey ";
	int				bytes;

	for (int i=0; i < 7; ++i)
	{
		if ((leds >> i) & 0x1)
		{
			pattern_on[3] = (char)(i + 48);
			bytes = write(sock, pattern_on, 11);
		}
		else
		{
			pattern_off[3] = (char)(i + 48);
			bytes = write(sock, pattern_off, 11);
		}
		printf("sent %d bytes\n", bytes);
	}
}

__NOEXC __WUR
static int		parse_data(char *response)
{
	e_button	button;
	char		*ptr;

	while (isspace(*response))
		++response;
	if (memcmp(response, "BUTTON", 6) != 0)
	{
		ft_perror("client", E_RESPONSE_FORMAT, response);
		return ;
	}
	response += 6;
	while (isspace(*response))
		++response;
	ptr = str2int(response, (int *)&button);
	if ((*ptr != ':' && !isspace(*ptr)) || !isbutton(button))
	{
		ft_perror("client", E_BUTTON_VALUE, response);
		return ;
	}
	response = ptr;
	if (*response == ':')
		++response;
	while (isspace(*response))
		++response;
	if (strcmp(response, "clicked") != 0)
	{
		ft_perror("client", E_RESPONSE_FORMAT, response);
		return ;
	}
	return button;
}

__NOEXC __WUR __INLINE
int		isbutton(e_button button)
{
	return (unsigned int) button <= 0xb;
}

__NOEXC __WUR
static int		parse_argv(int argc, char ** __restrict argv)
{
	if (argc != 2)
	{
		ft_perror("client", E_CLIENT_BADARG, NULL);
		ft_info("client", I_CLIENT_INFO, NULL);
		ft_exit(1);
	}
	if (!strcmp(argv[1], "-h") or !strcmp(argv[1], "--help"))
		print_help_message();
	if (!strcmp(argv[1], "stdin"))
		return 1;
	else
		return init_socket(argv[1]);
}
