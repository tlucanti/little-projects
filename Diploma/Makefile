
CXX_FLAGS	=
CXX_OPTIONS	=	-O3 -std=c++2a

INCLUDE		=	-I../../RayTracer/cllib -I../../RayTracer/cllib/include -I../../stl/test/ -I../../stl/include
CLLIB_SO	=	../../RayTracer/cllib/cllib.so
OPENCL_SO	=	/usr/lib/x86_64-linux-gnu/libOpenCL.so
LIBRARY		=	${CLLIB_SO} ${OPENCL_SO}

SRCS		=	\
				main.cpp

all:
	clang++ ${INCLUDE} ${CXX_FLAGS} ${CXX_OPTIONS} ${SRCS} ${LIBRARY}
