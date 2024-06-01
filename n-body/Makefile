
GUIROOT = ../../GUIlib
GUILIB = opengl
CC = g++

CXXFLAGS = -Wall -Wextra
CXXFLAGS += -std=c++17 -fdiagnostics-color
CXXFLAGS += -O3
CXXFLAGS += -D __linux__

all: fast slow example draw

draw:
	$(CC) $(CXXFLAGS) \
		-I ../../GUIlib/include/ \
		slow.cpp \
		`$(GUIROOT)/guiflags.sh $(GUIROOT)` \
		-lstdgui \
		-o nbody.elf
.PHONY: draw

fast:
	$(CC) $(CXXFLAGS) \
		-D DRAW=0 \
		fast.cpp \
		-o fast.elf
.PHONY: fast

slow:
	$(CC) $(CXXFLAGS) \
		-D DRAW=0 \
		slow.cpp \
		-o slow.elf
.PHONY: slow

example:
	$(CC) $(CXXFLAGS) \
		-I example \
		example/*.cpp \
		-o example.elf
.PHONY: example

