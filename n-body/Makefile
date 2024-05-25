
GUIROOT = ../../GUIlib
GUILIB = opengl

CXXFLAGS = -Wall -Wextra
CXXFLAGS += -std=c++17 -fdiagnostics-color
CXXFLAGS += -O3

all: fast slow

fast:
	g++ $(CXXFLAGS) \
		-I ../../GUIlib/include/ main.cpp \
		`$(GUIROOT)/guiflags.sh $(GUIROOT)` \
		-lstdgui \
		-o nbody.elf
.PHONY: fast

slow:
	g++ $(CXXFLAGS) \
		-I slow \
		slow/*.cpp \
		-o slow.elf
.PHONY: slow

