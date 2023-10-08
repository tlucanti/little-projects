
#include <common.hpp>
#include <DNF.hpp>

DNF::DNF(const std::string &input)
	: input(input)
{
	if (not powerof2(input.size())) {
		panic("input size not power of two");
	}

	for (unsigned int i = 0; i < input.size(); ++i) {
		switch (input.at(i)) {
		case '1':
			data.push_back(Impl(i, false));
			break;
		case '0':
			break;
		case '-':
			data.push_back(Impl(i, true));
			break;
		default:
			break;
		}
	}
}

DNF::~DNF()
{
}

int DNF::step()
{
	std::vector<Impl> next;
	int patched = 0;

	for (unsigned int i = 0; i < data.size(); ++i) {
		for (unsigned int j = 0; j < data.size(); ++j) {
			if (Impl::can_patch(data.at(i), data.at(j))) {
				std::cout << "patching " << i << ' ' << j << std::endl;
				Impl impl = Impl::patch(data.at(i), data.at(j));
				if (std::find(next.begin(), next.end(), impl) == next.end()) {
					next.push_back(impl);
				}
				++patched;
			}
		}
	}
	for (unsigned int i = 0; i < data.size(); ++i) {
		if (data.at(i).pw == false) {
			next.push_back(data.at(i));
		}
	}

	data.swap(next);

	std::cout << "patched " << patched << " implicants" << std::endl;
	return patched;
}

void DNF::minimize(std::ostream &out)
{
	while (true) {
		if (step() == 0) {
			break;
		}
		std::cout << "new implicants number: " << data.size() << std::endl;
		dump();
		std::cout << std::endl;
	}
	std::cout << "patching done" << std::endl;

	std::cout << std::endl;
	std::cout << "implicant table" << std::endl;
	table();

}

void DNF::dump(void)
{
	const int w = 3;

	std::cout << "num |";
	for (auto &i : data) {
		std::cout << std::setw(w);
		std::cout << i.num << " |";
	}
	std::cout << std::endl;

	std::cout << "ind |";
	for (auto &i : data) {
		std::cout << std::setw(w);
		std::cout << i.ind << " |";
	}
	std::cout << std::endl;

	std::cout << "p   |";
	for (auto &i : data) {
		std::cout << std::setw(w);
		std::cout << i.p << " |";
	}
	std::cout << std::endl;

	std::cout << "pw  |";
	for (auto &i : data) {
		std::cout << std::setw(w);
		std::cout << i.pw << " |";
	}
	std::cout << std::endl;

	std::cout << "inf |";
	for (auto &i : data) {
		std::cout << std::setw(w);
		std::cout << i.inf << " |";
	}
	std::cout << std::endl;
}

void DNF::table(void)
{
	const int w = 3;

	std::cout << "N   ";
	for (int i = 0; i < input.size(); ++i) {
		if (input.at(i) == '1') {
			std::cout << std::setw(w);
			std::cout << i + 1 << " |";
		}
	}

	std::cout << '|';
	for (int i = 0; i < input.size(); ++i) {
		if (input.at(i) == '-') {
			std::cout << std::setw(w);
			std::cout << i + 1;
		}
	}

	for (auto &i : data) {
		for (int i = 0; i < input.size(); ++i) {
			if (input.at(i) == '1') {
				std::cout << setw(w);
				std::cout << i + 1;
			}
		}

		std::cout << '|';
		for (int i = 0; i < input.size(); ++i) {
			if (input.at(i) == '-') {
				std::cout << std:setw(w);
}
