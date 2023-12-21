
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

static void parse_output(std::istream &in, std::vector<int> &output)
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

static void parse_output(const std::string &s, std::vector<int> &output)
{
	std::stringstream ss(s);
	parse_output(ss, output);
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

	at.encode();
	//at.minimize();
	at.dump();
	at.print();

	std::vector<int> w_states, w_inputs, w_outputs;
	w_inputs = { 3, 4, 2, 1, 1, 3, 3 };
	w_states = { 1, 7, 2, 9, 1, 2, 8, /*10*/ };
	parse_output("10111 11010 10001 00011 10111 10001 01011", w_outputs);

	std::cerr << "check word using table\n";
	at.check_word_table(w_inputs, w_states, w_outputs);
	std::cerr << "check word using encoded triggers\n";
	at.check_encoded_table(w_inputs, w_states, w_outputs);
}

