
#include <iostream>
#include <vector>
#include <string>

void panic()
{
	throw std::exception("panic");
}

vector<int> input(std::istream &in)
{
	std::string line;
	std::vector<int> v;
	int push;

	std::readline(in, line);

	v.reserce(line.size());
	for (char c : line) {

		switch (c) {
		case '0':
			push = 0;
			break;
		case '1':
			push = 1;
			break;
		case '-':
			push = -1;
			break;
		default:
			panic();
		}

		v.push_back(push);
	}
}




