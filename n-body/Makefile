
GUIROOT = ../../GUIlib
GUILIB = opengl

CXXFLAGS = -Wall -Wextra
CXXFLAGS += -std=c++17 -fdiagnostics-color
CXXFLAGS += -O3 -ffast-math -fopenmp
CXXFLAGS += -D __linux__
CXXFLAGS_SLOW += -O3 -g3

ALGO = RK4

all: fast slow nbody

nbody:
	g++ $(CXXFLAGS) \
		-I ../../GUIlib/include/ \
		main.cpp \
		-D ${ALGO} \
		`$(GUIROOT)/guiflags.sh $(GUIROOT)` \
		-lstdgui \
		-o nbody.elf
.PHONY: nbody

fast:
	g++ $(CXXFLAGS) \
		-D DRAW=0 \
		-D ${ALGO} \
		main.cpp \
		-o fast.elf
.PHONY: fast

slow:
	g++ $(CXXFLAGS_SLOW) \
		-D DRAW=0 \
		-D ${ALGO} \
		main.cpp \
		-o slow.elf
.PHONY: slow

