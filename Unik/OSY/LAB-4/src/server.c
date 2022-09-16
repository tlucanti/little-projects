/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   server.c                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: kostya <kostya@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2021/11/29 22:45:00 by kostya            #+#    #+#             */
/*   Updated: 2021/12/16 15:59:13 by kostya           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "../inc/lab4.h"
#include "../inc/memory.h"
#include "../inc/error.h"

extern ssize_t	read(int fd, void *buf, size_t count);

static int		init_socket() __NOEXC __WUR;
static void		*backend(int sock) __NOEXC __NORET;
static char		*read_file(char *fname, size_t *size) __NOEXC __WUR;
static int		parse_data(char *response, int verbose) __NOEXC __WUR;
static size_t	create_http(char *dest, char *data) __NOEXC;
static void		*run_led(t_shared_led *shared_led) __NOEXC __NORET;
static void		get_led_data(unsigned char *dest, const t_shared_led *shared_led) __NOEXC;
static int		min2(int a, int b) __NOEXC __WUR __INLINE;
static void		*stdin_read(t_shared_led *shared_led) __NOEXC __NORET;
static int		parse_led(char *response, int *power) __NOEXC __WUR;

pthread_mutex_t print_mutex;

__NOEXC
int main()
{
	int		sock;

	sock = init_socket();
	if (sock == -1)
		ft_exit(1);
	backend(sock);
}

__NOEXC __NORET
static void		*backend(int sock)
{
	pthread_t		thread;
	pthread_t		stdin_thread;
	char			request[REQUEST_MESSAGE_SIZE];
	char			response[RESPONSE_MESSAGE_SIZE];
	t_shared_led	shared_led;
	ssize_t			bytes;
	int				client = -1;

	shared_led.new_data = 0;
	pthread_mutex_init(&shared_led.led_mutex, 0);
	pthread_create(&stdin_thread, NULL, (void *(*)(void *))(void *)stdin_read, (void *)&shared_led);
	pthread_create(&thread, NULL, (void *(*)(void *))(void *)run_led, (void *)&shared_led);
	
	while (1)
	{
		close(client);
		client = accept(sock, NULL, NULL);
		if (client < 0)
		{
			ft_perror("server", errno, "cannot accept socket");
			continue ;
		}

		while (1)
		{
			close(client);
			client = accept(sock, NULL, NULL);

			bytes = recv(client, request, REQUEST_MESSAGE_SIZE, 0);
			
			if (bytes < 0)
			{
				ft_perror("server", errno, "cannot read data from client");
				break ;
			}
			if (strncasecmp(request, "GET / HTTP/1.1", 14) == 0)
			{
				size_t		fsize;
				char		*index;
				char		*html;

				index = read_file(INDEX_HTML, &fsize);
				if (index == NULL)
					break ;
				html = xmalloc(fsize + 100);
				fsize = create_http(html, index);
				if (send(client, html, fsize, 0) < 0)
					ft_perror("server", errno, "cannot send data to client");
				free(index);
				free(html);
			}
			else if (strncasecmp(request, "POST /writedata HTTP/1.1", 22) == 0)
			{
				char	*needle;
				int		button_num;

				needle = strcasestr(request, "BUTTON");
				if (needle == NULL)
					continue ;
				button_num = parse_data(needle, 1);

				pthread_mutex_lock(&shared_led.led_mutex);
				shared_led.button = button_num;
				shared_led.new_data = 1;
				pthread_mutex_unlock(&shared_led.led_mutex);
			}
			else if (strncasecmp(request, "GET /readdata HTTP/1.1", 22) == 0)
			{
				unsigned char	led_response[20] = "LEDCHAIN: ";
				char			response[200];

				get_led_data(led_response + 10, &shared_led);
				create_http(response, (char *)led_response);

				if (send(client, response, strlen(response), 0) < 0)
					ft_perror("server", errno, "cannot send data to client");
			}
			else
			{
				size_t	size = create_http(response, "");

				request[min2(strlen(request), 22)] = 0;
				ft_warning("server", W_UNKNOWN_REQUEST, request);
				if (send(client, response, size, 0) < 0)
					ft_perror("server", errno, "cannot send data to client");
			}
		}
	}
}

__NOEXC __NORET
static void		*run_led(t_shared_led *shared_led)
{
	unsigned int	leds;
	float			period;
	short			variant;
	e_button		direction;


	leds = 14;
	variant = 1;
	period = 1000000;
	while (1)
	{
		if (shared_led->new_data)
		{
			if (shared_led->new_data == 1)
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
			}
			else if (shared_led->new_data == 2)
			{
				if (shared_led->power)
					leds = (leds | (0x1 << (shared_led->button))) & 0b111111;
				else
					leds = (leds & ~(0x1 << (shared_led->button))) & 0b111111;
			}
			pthread_mutex_lock(&shared_led->led_mutex);
			shared_led->new_data = 0;
			pthread_mutex_unlock(&shared_led->led_mutex);
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
			if (variant == 1)
				leds = (leds >> 1) | ((leds & 0x1) << 5);
			else if (variant == 2)
				leds = (leds + 1) & 0b111111;
		}
		shared_led->leds = leds;
		usleep((int)period);
	}
}

__NOEXC __NORET
static void		*stdin_read(t_shared_led *shared_led)
{
	char	buff[200];
	int		n;
	int		power;

	while (1)
	{
		fgets(buff, 200, stdin);
		buff[strlen(buff) - 1] = 0;
		n = parse_data(buff, 0);
		if (n != -1)
		{
			pthread_mutex_lock(&shared_led->led_mutex);
			shared_led->button = n;
			shared_led->new_data = 1;
			pthread_mutex_unlock(&shared_led->led_mutex);
			ft_ok("server", K_BUTTON_CHANGE, buff);
			continue ;
		}
		n = parse_led(buff, &power);
		if (n != -1)
		{
			pthread_mutex_lock(&shared_led->led_mutex);
			shared_led->button = n;
			shared_led->power = power;
			shared_led->new_data = 2;
			pthread_mutex_unlock(&shared_led->led_mutex);
			ft_ok("server", K_LED_CHANGE, buff);
			continue ;
		}
		ft_perror("server", E_RESPONSE_FORMAT, buff);
	}
}

__NOEXC
static void		get_led_data(unsigned char *dest, const t_shared_led *shared_led)
{
	unsigned int				leds;

	leds = shared_led->leds;
	for (int i=0; i < led_count; ++i)
	{
		dest[led_count - i - 1] = leds % 2 + '0';
		leds /= 2;
	}
}

__NOEXC __WUR
static int		init_socket()
{
	int		sock;
	struct sockaddr_in	server = {};

	sock = socket(AF_INET, SOCK_STREAM, 0);
	if (sock == -1)
	{
		ft_perror("server", errno, "cannot create socket");
		return -1;
	}

	server.sin_family = AF_INET;
	server.sin_addr.s_addr = inet_addr("0.0.0.0");
	server.sin_port = htons(8080);

	if (bind(sock, (struct sockaddr *)&server, sizeof(server)))
	{
		ft_perror("server", errno, "cannot bind address");
		return -1;
	}
	if (listen(sock, 10))
	{
		ft_perror("server", errno, "cannot listen port");
		return -1;
	}
	return sock;
}

__NOEXC __WUR
static char		*read_file(char *fname, size_t *size)
{
	struct stat	st;
	char		*ret;
	int			fd;

	fd = open(fname, O_RDONLY);
	if (fd == -1 || stat(fname, &st))
	{
		ft_perror("server", errno, fname);
		return NULL;
	}
	*size = st.st_size;
	ret = xmalloc(st.st_size + 1);
	read(fd, ret, st.st_size);
	ret[st.st_size] = 0;
	return ret;
}

__NOEXC __WUR
static int		parse_data(char *response, int verbose)
{
	e_button	button;
	char		*ptr;

	while (isspace(*response))
		++response;
	if (strncasecmp(response, "BUTTON", 6) != 0)
	{
		if (verbose)
			ft_perror("server", E_RESPONSE_FORMAT, response);
		return -1;
	}
	response += 6;
	while (isspace(*response))
		++response;
	ptr = str2int(response, (int *)&button);
	response = ptr;
	while (isspace(*response))
		++response;
	if (*response == ':')
		++response;
	while (isspace(*response))
		++response;
	if (strncasecmp(response, "clicked", 6) != 0)
	{
		if (verbose)
			ft_perror("server", E_RESPONSE_FORMAT, response);
		return -1;
	}
	return button;
}

__NOEXC __WUR
static int		parse_led(char *response, int *power)
{
	int		led;
	char	*ptr;

	while (isspace(*response))
		++response;
	if (strncasecmp(response, "LED", 3) != 0)
		return -1;
	response += 3;
	while (isspace(*response))
		++response;
	ptr = str2int(response, (int *)&led);
	response = ptr;
	while (isspace(*response))
		++response;
	if (*response == ':')
		++response;
	while (isspace(*response))
		++response;
	if (strncasecmp(response, "on", 2) == 0)
		*power = 1;
	else if (strncasecmp(response, "off", 3) == 0)
		*power = 0;
	else
		return -1;
	return led;
}

__NOEXC
static size_t	create_http(char *dest, char *data)
{
	size_t		size = strlen(data);

	if (size == 0)
		data ="\r\n\r\n";
	return sprintf(dest,
		"HTTP/1.1 200 OK\r\n"
		"Status: 200 OK\r\n"
		"Content-Length: %zu\r\n\r\n"
		"%s",
		size, data);
}

__NOEXC __WUR __INLINE
static int		min2(int a, int b)
{
	if (a <= b)
		return a;
	return b;
}
