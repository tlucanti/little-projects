
#include <common.hpp>
#include <Automat.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>

static void parse_table(std::istream &in, std::vector<std::vector<int>> &transition)
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
			unsigned state;
			ss >> edge;

			if (edge.empty()) {
				break;
			}

			if (edge == "-") {
				state = -1;
			} else {
				state = std::stoi(edge) - 1;
			}

			transition.back().push_back(state);
		}
	}

	size_t cols = transition.front().size();
	for (const auto &v : transition) {
		if (v.size() != cols) {
			panic("bad shape");
		}
	}
}

static void parse_output(std::ifstream &in, std::vector<int> &output)
{
	std::string line;
	std::getline(in, line);

	std::stringstream ss(line);
	while (ss) {
		std::string sout;
		ss >> sout;

		if (sout.empty()) {
			break;
		}
		output.push_back(std::stoi(sout, nullptr, 2));
	}
}

int main(int argc, char **argv)
{
	std::vector<std::vector<int>> trans;
	std::vector<int> states;

	if (argc != 3) {
		panic("argv err");
	}

	std::ifstream tr(argv[1]);
	if (!tr.is_open()) {
		panic("open transition file error");
	}

	std::ifstream st(argv[2]);
	if (!st.is_open()) {
		panic("open state file error");
	}

	parse_table(tr, trans);
	parse_output(st, states);

	Table table(trans);

	Automat at(trans, states);
}

