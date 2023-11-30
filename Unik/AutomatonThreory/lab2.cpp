
#include <Table.hpp>
#include <common.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>

void parse_table(std::istream &in, std::vector<std::vector<int> > &transition)
{
	int rows = 0;
	std::string line;

	while (true) {
		std::getline(in, line);
		if (line.empty()) {
			break;
		}

		++rows;
		std::stringstream ss(line);
		transition.resize(rows);

		while (ss) {
			std::string edge;
			int state;
			ss >> edge;

			if (edge.empty()) {
				break;
			}

			std::cout << edge << ' ';
			if (edge == "-") {
				state = 0;
			} else {
				state = std::stoi(edge);
			}

			transition.back().push_back(state);
		}
		std::cout << std::endl;
	}

	size_t cols = transition.front().size();
	for (const auto &v : transition) {
		if (v.size() != cols) {
			panic("bad shape");
		}
	}
}

int main(int argc, char **argv)
{
	if (argc != 2) {
		panic("argv err");
	}

	std::ifstream tr(argv[1]);
	if (!tr.is_open()) {
		panic("open error");
	}

	std::vector<std::vector<int>> trans;
	parse_table(tr, trans);

	Table table(trans);
	std::cout << '\n';
	table.dump();
}

