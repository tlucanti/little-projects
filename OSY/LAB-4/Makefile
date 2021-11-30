# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    Makefile                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: kostya <kostya@student.42.fr>              +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2021/09/08 23:21:10 by kostya            #+#    #+#              #
#    Updated: 2021/11/30 21:48:36 by kostya           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

# ------------------------------ compiler options ------------------------------
NAME_SERVER	=	server
NAME_CLIENT	=	client
CC			=	clang
CFLAGS		=	-Wall -Wextra -Werror
COPTIONS	=	-O2
# ------------------------------- linker options -------------------------------
LIBRARY		=	-lpthread
OBJS_DIR	=	obj
INCLUDE_DIR	=	inc
SCRS_DIR	=	src
# ------------------------------- project sorces -------------------------------
SRCS_SERVER	=	\
				server	\
				memory	\
				error
# ------------------------------------------------------------------------------
SRCS_CLIENT	=	\
				client	\
				error
# ------------------------------------------------------------------------------
HDRS_SERVER	=	\
				server	\
				lab4	\
				memory	\
				color	\
				error
# ------------------------------------------------------------------------------
HDRS_CLIENT	=	\
				client	\
				lab4	\
				color	\
				error

# ======================= UNCHANGEABLE PART OF MAKEFILE ========================
# ------------------------------------------------------------------------------
RM			=	rm -f
OBJS_SERVER	=	$(addprefix ${OBJS_DIR}/,${SRCS_SERVER:=.o})
OBJS_CLIENT	=	$(addprefix ${OBJS_DIR}/,${SRCS_CLIENT:=.o})
DEPS_CLIENT	=	$(addprefix ${INCLUDE_DIR}/,${HDRS_CLIENT:=.h})
DEPS_SERVER	=	$(addprefix ${INCLUDE_DIR}/,${HDRS_SERVER:=.h})
INCLUDE		=	-I ${INCLUDE_DIR}

# ------------------------------------------------------------------------------
all:			${OBJS_DIR}
	@$(MAKE)	$(NAME_SERVER) -j
	@$(MAKE)	$(NAME_CLIENT) -j

# ------------------------------------------------------------------------------
${OBJS_DIR}/%.o: ${SCRS_DIR}/%.c Makefile
	${CC}		${CFLAGS} ${COPTIONS} -c -o $@ $< ${INCLUDE}

# ------------------------------------------------------------------------------
$(NAME_SERVER):	${OBJS_SERVER} ${DEPS_SERVER}
	${CC}		-o ${NAME_SERVER} ${CFLAGS} ${COPTIONS} ${OBJS_SERVER} ${LIBRARY}

$(NAME_CLIENT):	${OBJS_CLIENT} ${DEPS_CLIENT}
	${CC}		-o ${NAME_CLIENT} ${CFLAGS} ${COPTIONS} ${OBJS_CLIENT} ${LIBRARY}

# ------------------------------------------------------------------------------
clean:
	${RM}		${OBJS_SERVER}
	${RM}		${OBJS_CLIENT}

# ------------------------------------------------------------------------------
fclean:			clean
	${RM}		${NAME_SERVER}
	${RM}		${NAME_CLIENT}

# ------------------------------------------------------------------------------
${OBJS_DIR}:
	mkdir		-p ${OBJS_DIR}

# ------------------------------------------------------------------------------
re:				fclean all

# ------------------------------------------------------------------------------
.PHONY:			all clean fclean re
