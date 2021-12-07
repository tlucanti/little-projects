/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   server.c                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: kostya <kostya@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2021/11/29 22:45:00 by kostya            #+#    #+#             */
/*   Updated: 2021/12/07 20:07:52 by kostya           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "../inc/lab4.h"
#include "../inc/server.h"
#include "../inc/memory.h"
#include "../inc/error.h"

static int		init_socket() __NOEXC __WUR;
static void		*backend(size_t cli_t) __NOEXC;
static char		*read_file(char *fname, ssize_t *size) __NOEXC __WUR;

__NOEXC
int main()
{
	int		sock;

	sock = init_socket();
	if (sock == -1)
		return 1;

	while (1)
	{
		pthread_t	thread;
		int			client;

		client = accept(sock, NULL, NULL);
		printf("got socket from %d\n", client);
		pthread_create(&thread, NULL, (void *(*)(void *))(void *)backend, (void *)(size_t)client);
		pthread_detach(thread);
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
		if (bytes <= 0)
		{
			perror("cannot read data from client");
			break ;
		}
		printf("read data from client (%zuB):\n<%s>\n", bytes, request);
		printf("reqv: %d %d %d %d\n", request[13], request[14], request[15], request[16]);
		if (memcmp(request, "GET / HTTP/1.1", 14) == 0)
		{
			ssize_t		fsize;
			char		*index;
			char		*response;

			index = read_file(INDEX_HTML, &fsize);
			if (index == NULL)
				break ;
			response = xmalloc(fsize + 100);
			sprintf(response,
				"HTTP/1.1 200 OK\r\nStatus: 200 OK\r\nContent-Length: %ld\r\n\r\n%s",
				fsize, index);
			printf("write data to client %lu(B) <%s>\n", strlen(response), response);
			write(client, response, strlen(response));
			printf("OK\n");
		}
		else
			printf("unknown request format\n");
		// str2int(request, &pid);
		// int button = 0;
		// (void)_;
		// sprintf(response, "button%d: clicked", button);
		// dprintf(client, response, RESPONSE_MESSAGE_SIZE);
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
