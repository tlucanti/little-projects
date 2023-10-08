
#include <common.hpp>
#include <DNF.hpp>

#include <iostream>
#include <vector>
#include <string>

int main(int argc, char **argv)
{
	if (argc != 2) {
		panic("expected 1 argument");
	}

	DNF dnf(argv[1]);
	dnf.dump();
	dnf.minimize();
}

