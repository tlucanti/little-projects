cmake_minimum_required(VERSION 3.20)
project(LAB_4 C)

set(CMAKE_C_STANDARD 11)

add_compile_options("-DINDEX_HTML=\"../html/index.html\"")

add_executable(server
	src/server.c
	src/memory.c
	src/error.c
	src/utils.c
	inc/lab4.h
	inc/memory.h
	inc/color.h
	inc/error.h
)

target_link_libraries(server
	pthread
)