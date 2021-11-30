/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   server.c                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: kostya <kostya@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2021/11/29 22:45:00 by kostya            #+#    #+#             */
/*   Updated: 2021/11/30 21:53:04 by kostya           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "../inc/lab4.h"
#include "../inc/server.h"
#include "../inc/memory.h"

static int		init_socket() __NOEXC __WUR;
static void		*backend(size_t cli_t) __NOEXC;

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
	char		request[REQUEST_MESSAGE_SIZE];
	char		response[RESPONSE_MESSAGE_SIZE];
	size_t		bytes;
	const int	client = (int)cli_t;

	while (1)
	{
		bytes = read(client, request, REQUEST_MESSAGE_SIZE);
		if (bytes <= 0)
		{
			perror("cannot read data from client");
			continue ;
		}

		printf("read data from client (%zuB):\n<%s>\n", bytes, request);
		printf("responsing to client with button >> ");
		int button = 0;
		int _ = scanf("%d", &button);
		(void)_;
		sprintf(response, "button%d: clicked", button);
		dprintf(client, response, RESPONSE_MESSAGE_SIZE);
	}
	return (NULL);
}

__NOEXC __WUR
static int		init_socket()
{
	int		sock;
	struct sockaddr_in	server = {};

	sock = socket(AF_INET, SOCK_STREAM, 0);
	if (sock == -1)
	{
		perror("cannot create socket");
		return -1;
	}

	server.sin_family = AF_INET;
	server.sin_addr.s_addr = htonl(INADDR_ANY);
	server.sin_port = htons(8080);

	if (bind(sock, (struct sockaddr *)&server, sizeof(server)))
	{
		perror("cannot bind address");
		return -1;
	}
	if (listen(sock, 10))
	{
		perror("cannot listen port");
		return -1;
	}
	return sock;
}
