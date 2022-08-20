
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <netdb.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <stdlib.h>
#include <stdio.h>
#include <fcntl.h>

int MSG_SIZE = 262143;

void pt(char *s) { write(2,s,strlen(s)); }

void ft_memcpy(char *dst, char *src, long size) {
	for (long i=0; i < size; ++i) dst[i] = src[i];
}

int fatal() {
	pt("fatal\n");
	exit(1);
}

typedef struct s_list {
	int fd;
	int id;
	struct s_list *next;
} t_list;

void *xmalloc(long size) {
	void *ret = memset(malloc(size), 0, size);
	if (ret == NULL) fatal();
	return ret;
}

void client_add(t_list **the_list, int fd, int id) {
	t_list *node = xmalloc(sizeof(t_list));
	node->fd = fd;
	node->id = id;
	t_list *root = *the_list;
	if (root == NULL) {
		*the_list = node;
		return ;
	} else while (root->next)
		root = root->next;
	root->next = node;
}

void do_send(int fd, char *msg, long size) {
	long c, sent = 0;
	while (sent < size) {
		c = send(fd, msg + sent, size - sent, 0);
		if (c < 0)
			fatal();
		sent += c;
	}
}

long do_recv(int fd, char *dst) {
	long c, rc = 0;
	while (1) {
		c = recv(fd, dst, 65535, 0);
		if (c <= 0)
			return rc;
		rc += c;
	}
}

void remove_client(t_list **the_root, int fd) {
	t_list *next, *root = *the_root;
	if (root->fd == fd) {
		*the_root = (*the_root)->next;
		free(root);
		close(fd);
		return ;
	} else while (root->next->fd != fd)
		root = root->next;
	next = root->next->next;
	free(root->next);
	close(fd);
	root->next = next;
}

void send_all_except(t_list **the_root, char *msg, int fd) {
	t_list *root = *the_root;
	while (root) {
		if (root->fd != fd)
			do_send(root->fd, msg, strlen(msg));
		root = root->next;
	}
}

void parse_msg(char msg[], int id) {
	static char *buf = NULL;
	if (buf == NULL)
		buf = xmalloc(MSG_SIZE);
	bzero(buf, MSG_SIZE);
	int i=0;
	int pos = sprintf(buf, "client %d: ", id);
	while (msg[i]) {
		if (msg[i] == '\n' && msg[i + 1] != '\0')
			pos += sprintf(buf + pos, "\nclient %d: ", id);
		else if (msg[i] != '\n')
			buf[pos++] = msg[i];
		++i;
	}
	buf[pos] = '\n';
	ft_memcpy(msg, buf, pos + 1);
}

int main(int argc, char **argv) {
	int sock, client;
	struct sockaddr_in sock_addr, cli_addr;
	if (argc != 2) {
		pt("wrong argumenst\n");
		exit(1);
	}
	sock = socket(AF_INET, SOCK_STREAM, 0); 
	fcntl(sock, F_SETFL, O_NONBLOCK);
	if (sock < 0)
		fatal();
	bzero(&sock_addr, sizeof(sock_addr)); 
	bzero(&cli_addr, sizeof(cli_addr));
	sock_addr.sin_family = AF_INET; 
	sock_addr.sin_addr.s_addr = htonl(2130706433); //127.0.0.1
	sock_addr.sin_port = htons(atoi(argv[1]));
	if (bind(sock, (const struct sockaddr *)&sock_addr, sizeof(sock_addr)))
		fatal();
	if (listen(sock, 10))
		fatal();
	fd_set in, out;
	char *msg = xmalloc(MSG_SIZE);
	t_list *clients = NULL;
	int last_client = 0;
	int max_fd = sock;
	FD_ZERO(&out);
	while (1) {
		FD_ZERO(&in);
		FD_SET(sock, &in);
		t_list *root = clients;
		while (root) {
			FD_SET(root->fd, &in);
			root = root->next;
		}
		int sel = select(max_fd + 100, &in, &out, NULL, NULL);
		if (sel <= 0)
			continue ;
		if (FD_ISSET(sock, &in)) {
			unsigned len = sizeof(cli_addr);
			client = accept(sock, (struct sockaddr *)&cli_addr, &len);
			if (client <= 0)
				continue ;
			max_fd = max_fd > client ? max_fd : client;
			client_add(&clients, client, last_client);
			char buf[127];
			sprintf(buf, "new client %d\n", last_client);
			send_all_except(&clients, buf, client);
			++last_client;
			continue ;
		}
		root = clients;
		while (root) {
			t_list *next = root->next;
			if (FD_ISSET(root->fd, &in)) {
				bzero(msg, MSG_SIZE);
				int r = do_recv(root->fd, msg);
				if (r <= 0)
				{
					int id = root->id;
					remove_client(&clients, root->fd);
					char buf[127];
					sprintf(buf, "server: client %d died\n", id);
					send_all_except(&clients, buf, -1);
					root = next;
					continue ;
				}
				parse_msg(msg, root->id);
				send_all_except(&clients, msg, root->fd);
			}
			root = next;
		}
	}
}

