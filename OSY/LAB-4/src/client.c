/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   client.c                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: kostya <kostya@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2021/11/29 22:35:59 by kostya            #+#    #+#             */
/*   Updated: 2021/11/30 21:53:34 by kostya           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "../inc/lab4.h"
#include "../inc/client.h"

static int		init_socket() __NOEXC __WUR;

__NOEXC
int		main()
{
	int			sock;
	char		request[REQUEST_MESSAGE_SIZE];
	char		response[RESPONSE_MESSAGE_SIZE];

	sock = init_socket();
	if (sock == -1)
		return 1;
	sprintf(request, "client [%d]: waiting for response", getpid());

	while (1)
	{
		int		bytes;

		printf("sending data to server: <%s>\n", request);
		bytes = write(sock, request, REQUEST_MESSAGE_SIZE);
		if (bytes <= 0)
		{
			perror("cannot write data to server");
			continue ;
		}
		bytes = read(sock, response, RESPONSE_MESSAGE_SIZE);
		if (bytes <= 0)
		{
			perror("cannot read data from server");
			continue ;
		}
		printf("read data from server (%dB):\n<%s>\n", bytes, response);
	}
}

__NOEXC __WUR
static int		init_socket()
{
	struct sockaddr_in	server;
	const char			ip[] = "0.0.0.0";
	int					sock;

	sock = socket(AF_INET, SOCK_STREAM, 0);
	if (sock == -1)
	{
		perror("cannot create socket");
		return -1;
	}

	server.sin_family = AF_INET;
	server.sin_port = htons(8080);

	if (inet_pton(AF_INET, ip, &server.sin_addr) == -1)
	{
		perror("cannot parse address");
		return -1;
	}
	if (connect(sock, (struct sockaddr *)&server, sizeof(server)))
	{
		perror("cannot connect to server");
		return -1;
	}
	return sock;
}
