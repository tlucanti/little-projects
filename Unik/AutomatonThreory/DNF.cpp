
#include <common.hpp>
#include <DNF.hpp>

DNF::DNF(const std::string &input)
{
	if (not powerof2(input.size())) {
		panic("input size not power of two");
	}

	for (unsigned int i = 0; i < input.size(); ++i) {
		switch (input.at(i)) {
		case '1':
			data.push_back(Impl(i));
			break;
		case '0':
			break;
		case '-':
			data.push_back(Impl(i));
			break;
		default:
			break;
		}
	}
}

DNF::~DNF()
{
}

void DNF::step()
{
	vector<Impl> next;

	for (unsigned int i = 0; i < data.size(); ++i) {
		for (unsigned int j = 0; j < data.size(); ++j) {
			if (Impl::can_patch(data.at(i), data.at(j))) {
				next.push_back(
					Impl::patch(data.at(i), data.at(j)));
			}
		}
	}
	for (unsigned int i = 0; i < data.size(); ++i) {
		if (data.at(i).pw == false) {
			next.push_back(data.at(i));
		}
	}

	data.swap(next);
}

void DNF::minimize(std::ostream &out)
{
	step();

	for (auto &i : data) {
		std::cout << i << std::endl;
	}
}
