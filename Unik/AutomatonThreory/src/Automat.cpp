
#include <Automat.hpp>

Automat::Automat(const Table &transition, const std::vector<int> &output)
{
	std::vector<std::vector<int>> mt(transition.get_rows());

	if (transition.get_cols() != output.size()) {
		panic("transition & output shape mismatch");
	}

	for (size_t r = 0; r < transition.get_rows(); ++r) {
		mt.at(r).resize(transition.get_cols());
		for (size_t c = 0; c < transition.get_cols(); ++c) {
			mt.at(r).at(c) = output.at(c);
		}
	}

	init_automat(transition, Table(mt));
}

Automat::Automat(const Table &transition, const Table &output)
{
	init_automat(transition, output);
}

void Automat::init_automat(const Table &trans, const Table &out)
{
	transition = trans;
	output = out;

	if (transition.get_rows() != output.get_rows() || transition.get_cols() != output.get_cols()) {
		panic("transition and output shape mismatch");
	}

	std::cout << "Transition\n";
	transition.dump();
	std::cout << "Output\n";
	output.dump();
}

