
all:
	g++ \
		-Wall -Wextra \
		-std=c++2a -fdiagnostics-color=always \
		-O3 \
		-I ../../GUIlib/include/ \
		main.cpp \
		\
		`$$GUIROOT/guiflags.sh $$GUIROOT)` \
		-lstdgui \
		-o nbody

