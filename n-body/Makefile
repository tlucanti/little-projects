
GUIROOT = ../../GUIlib
GUILIB = opengl

CXXFLAGS = -Wall -Wextra
CXXFLAGS += -std=c++17 -fdiagnostics-color
CXXFLAGS += -O3 -g3
CXXFLAGS += -D __linux__

all: fast slow nbody

nbody:
	g++ $(CXXFLAGS) \
		-I ../../GUIlib/include/ \
		main.cpp \
		`$(GUIROOT)/guiflags.sh $(GUIROOT)` \
		-lstdgui \
		-o nbody.elf
.PHONY: nbody

fast:
	g++ $(CXXFLAGS) \
		-D DRAW=0 \
		main.cpp \
		-o fast.elf
.PHONY: fast

slow:
	g++ $(CXXFLAGS) \
		-I slow \
		slow/*.cpp \
		-o slow.elf
.PHONY: slow

