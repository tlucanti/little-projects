/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   server.c                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: kostya <kostya@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2021/11/29 22:45:00 by kostya            #+#    #+#             */
/*   Updated: 2021/12/13 15:20:06 by kostya           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "../inc/lab4.h"
#include "../inc/server.h"
#include "../inc/memory.h"
#include "../inc/error.h"

static int		init_socket() __NOEXC __WUR;
static void		*backend(size_t cli_t) __NOEXC;
static char		*read_file(char *fname, ssize_t *size) __NOEXC __WUR;
static int		parse_data(char *response) __NOEXC __WUR;
// static int		isbutton(e_button button) __NOEXC __WUR __INLINE;
static size_t	create_http(char *dest, char *data) __NOEXC;

pthread_mutex_t print_mutex;

void http_200OK (char *response, char *data) {
	strcpy(response, "HTTP/1.1 200 OK\r\n");
	strcat(response, "Status: 200 OK\r\n");
	strcat(response, "Content-Length: ");
	if (strlen(data) == 0) {
		strcat(response, "0\r\n\r\n");
	} else {
		sprintf(response+strlen(response), "%lu\r\n\r\n%s", strlen(data), data);
	}
}


__NOEXC __NORET
int main()
{
	int		sock;

	sock = init_socket();
	if (sock == -1)
		ft_exit(1);

	while (1)
	{
		pthread_t	thread;
		int			client;

		client = accept(sock, NULL, NULL);
		if (client < 0)
		{
			ft_perror("server", errno, "cannot accept socket");
			continue ;
		}
		printf("got socket from %d\n", client);
		if (pthread_create(&thread, NULL,
			(void *(*)(void *))(void *)backend, (void *)(size_t)client))
		{
			ft_perror("pthread", errno, "cannot create thread");
			continue ;
		}
		pthread_join(thread, NULL);
		// pthread_detach(thread);
	}
}

__NOEXC
static void		*backend(size_t cli_t)
{
	// volatile sig_atomic_t	button;
	// pthread_t				thread;
	char		request[REQUEST_MESSAGE_SIZE];
	// char		response[RESPONSE_MESSAGE_SIZE];
	size_t		bytes;
	const int	client = (int)cli_t;
	// pid_t		pid;

	// pthread_create(&thread, NULL, (void *(*)(void *))(void *)button_checker, (void *)&button);
	while (1)
	{
		bytes = recv(client, request, REQUEST_MESSAGE_SIZE, 0);
		printf("got request %.22s\n", request);
		// printf("------------------------------------------------------\n"
			// "read data from client (%zuB)\n%s\n"
			// "------------------------------------------------------\n",
			// bytes, request);
		if (bytes < 0)
		{
			ft_perror("server", errno, "cannot read data from client");
			break ;
		}
		if (memcmp(request, "GET / HTTP/1.1", 14) == 0)
		{
			// printf("------------------------------------------------------\n"
			// 	"read data from client (%zuB)\n%s\n"
			// 	"------------------------------------------------------\n",
			// 	bytes, request);
			ssize_t		fsize;
			char		*index;
			char		*response;

			index = read_file(INDEX_HTML, &fsize);
			if (index == NULL)
				break ;
			response = xmalloc(fsize + 100);
			fsize = (size_t)create_http(response, index);
			printf("++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
				"write data to client %lu(B) \n%s\n"
				"++++++++++++++++++++++++++++++++++++++++++++++++++++++\n",
				fsize, response);
			if (send(client, response, fsize, 0) < 0)
				ft_perror("server", errno, "cannot send data to client");
			else
				printf("OK\n");
			free(index);
			free(response);
		}
		else if (strncmp(request, "POST /writedata HTTP/1.1", 22) == 0)
		{
			// printf("------------------------------------------------------\n"
			// 	"read data from client (%zuB)\n%s\n"
			// 	"------------------------------------------------------\n",
			// 	bytes, request);

			char response[200];

			for (size_t i=0; i < bytes; ++i)
			{
				if (memcmp(request + i, "BUTTON", 6) == 0)
				{
					char	led[20];
					int		button_num = parse_data(request + i);

					sprintf(led, "LED%d: green\n", button_num);
					http_200OK(response, led);
					size_t size = strlen(response);
					printf("led\n");
					printf("++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
						"write data to client %lu(B) \n%s\n"
						"++++++++++++++++++++++++++++++++++++++++++++++++++++++\n",
						size, response);
					if (write(client, response, size) < 0)
					// if (send(client, response, size, 0) < 0)
						ft_perror("server", errno, "cannot send data to client");
					break ;

					// // for (int KK=0; KK<5; ++KK) {
					// 	int		button_num = parse_data(request + i);
					// 	char	led[20];
					// 	char	response[200];
					// 	size_t	r_size;

					// 	// sprintf(led, "LED%d: green", button_num);
					// 	http_200OK(response, "LED0: green\n");
					// 	r_size = strlen(response);
					// 	// r_size = create_http(response, led);
					// 	printf("++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
					// 		"send data to clinet\n%.*s\n"
					// 		"++++++++++++++++++++++++++++++++++++++++++++++++++++++\n",
					// 		(int)r_size, response);
					// 	if (send(client, response, r_size, 0) < 0)
					// 		ft_perror("server", errno, "cannot send data to client");
					// // }
				}
			}
		}
		else if (strncmp(request, "GET /readdata HTTP/1.1", 22) == 0)
		{
			char	response[200];
			printf("keep-alive\n");
			http_200OK(response, "");
			printf("send\n");
			if (write(client, response, strlen(response)) < 0)
			// if (send(client, response, size, 0) < 0)
				ft_perror("server", errno, "cannot send data to client");
		}
		else
		{
			char	response[200];
			size_t	size = create_http(response, "");

			// http_200OK(response, "LED0: green\n");
			// size_t size = strlen(response);

			printf("unknown-request\n");
			if (write(client, response, size) < 0)
			// if (send(client, response, size, 0) < 0)
				ft_perror("server", errno, "cannot send data to client");
		}
		// str2int(request, &pid);
		// int button = 0;
		// (void)_;
		// sprintf(response, "button%d: clicked", button);
	}
	close(client);
	return (NULL);
}

// __NOEXC
// static void		*button_checker(t_button_checker *data)
// {
// 	int		button_num;

// 	while (1)
// 	{
// 		pthread_mutex_lock(&data->mutex);
// 		if (data->buton == 2)
// 		{
// 			pthread_mutex_unlock(&data->mutex);
// 			pthread_mutex_destroy(&data->mutex);
// 			return NULL;
// 		}
// 		pthread_mutex_unlock(&data->mutex);
// 		int _ = scanf("%d", &data);
// 		if (!isbutton(data))
// 			ft_perror("server", E_BUTTON_VALUE, NULL);
// 		pthread_mutex_lock(&data->mutex);
// 		dprintf(data->socket, "button%d: clicked", button_num);
// 		pthread_mutex_unlock(&data->mutex);
// 	}
// }

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
	server.sin_addr.s_addr = htonl(INADDR_ANY);
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
static char		*read_file(char *fname, ssize_t *size)
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
static int		parse_data(char *response)
{
	e_button	button;
	char		*ptr;

	while (isspace(*response))
		++response;
	if (memcmp(response, "BUTTON", 6) != 0)
	{
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
	if (memcmp(response, "clicked", 6) != 0)
	{
		ft_perror("server", E_RESPONSE_FORMAT, response);
		return -1;
	}
	return button;
}

// __NOEXC __WUR __INLINE
// static int		isbutton(e_button button)
// {
// 	return (unsigned int) button <= 0xb;
// }

__NOEXC
static size_t	create_http(char *dest, char *data)
{
	size_t		size = strlen(data);

	if (size == 0)
		data ="\r\n\r\n";
	return sprintf(dest,
		"HTTP/1.1 200 OK\r\n"
		"Status: 200 OK\r\n"
		// "Content-Type: \"text/plain\"\r\n"
		"Content-Length: %zu\r\n\r\n"
		"%s",
		size, data);
}
