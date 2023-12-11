
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
	encoded = false;

	if (transition.get_rows() != output.get_rows() ||
	    transition.get_cols() != output.get_cols()) {
		panic("transition and output shape mismatch");
	}

	std::cerr << "Transition\n";
	transition.dump();
	std::cerr << "Output\n";
	output.dump();
}

void Automat::encode(void)
{
	std::vector<std::string> Dstr;
	std::vector<std::string> ystr;

	std::cerr << "Encoding D:\n";
	encode_table(transition, Dstr);
	std::cerr << "\nEncoding y:\n";
	encode_table(output, ystr);

	for (const auto &s : Dstr) {
		D.emplace_back(s);
	}

	for (const auto &s : ystr) {
		y.emplace_back(s);
	}

	if (D.size() != y.size()) {
		// panic("encoded triggers size mismatch");
	}
	encoded = true;
}

void Automat::minimize(void)
{
	for (auto &d : D) {
		d.minimize();
	}
	for (auto &d : y) {
		d.minimize();
	}
}

void Automat::dump(void)
{
	transition.dump();
	output.dump();
}

void Automat::print(void)
{
	for (size_t i = 0; i < D.size(); ++i) {
		std::cout << 'D' << i << ":\n";
		D.at(i).print();
		std::cout << '\n';
	}

	for (size_t i = 0; i < y.size(); ++i) {
		std::cout << 'y' << i << ":\n";
		y.at(i).print();
		std::cout << '\n';
	}
}

std::pair<int, int> Automat::run_encoded(int state, int input)
{
	if (!encoded) {
		panic("cannot run before encoding");
	}

	int state_bits = ilog2(transition.get_cols() - 1) + 1;

	unsigned int bitv = (input << state_bits) | state;
	unsigned int next_state = 0;
	unsigned int next_output = 0;

	for (const auto &dt : D) {
		next_state <<= 1;
		next_state |= dt(bitv);
	}

	for (const auto &yt : y) {
		next_output <<= 1;
		next_output |= yt(bitv);
	}

	return { next_state, next_output };
}

std::pair<int, int> Automat::run_table(int state, int input)
{
	int next_state = transition.at(state - 1, input - 1).get() + 1;
	int next_outp = output.at(state - 1, input - 1).get() + 1;

	return { next_state, next_outp };
}

void Automat::_check_word(const std::vector<int> &states,
			  const std::vector<int> &inputs,
			  const std::vector<int> &outputs,
			  bool table)
{
	std::pair<int, int> next;

	if (states.size() != inputs.size() || inputs.size() != outputs.size()) {
		panic("check word length mismatch");
	}

	for (size_t i = 0; i < states.size(); ++i) {
		if (table) {
			next = run_table(states.at(i), inputs.at(i));
		} else {
			next = run_encoded(states.at(i), inputs.at(i));
		}

		std::cout << "state: " << states.at(i)
			  << " (" << next.first << "), ";
		std::cout << "oputput: " << outputs.at(i)
			  << " (" <<  next.second << ")\n";
	}

}

void Automat::check_word_table(const std::vector<int> &states,
			       const std::vector<int> &inputs,
			       const std::vector<int> &outputs)
{
	_check_word(states, inputs, outputs, true);
}

void Automat::check_encoded_table(const std::vector<int> &states,
				  const std::vector<int> &inputs,
				  const std::vector<int> &outputs)
{
	_check_word(states, inputs, outputs, false);
}

void Automat::encode_table(const Table &tbl, std::vector<std::string> &trig)
{
	int nr_triggers = tbl.get_w();

	trig.resize(nr_triggers);
	for (int tr = 0; tr < nr_triggers; ++tr) {
		encode_trigger(tbl, trig.at(tr), tr);
	}
}

void Automat::encode_trigger(const Table &tbl, std::string &s, int bit)
{
	int row_bits = ilog2(tbl.get_rows() - 1) + 1;
	int col_bits = ilog2(tbl.get_cols() - 1) + 1;

	s.resize(1 << (col_bits + row_bits), '-');

	for (size_t r = 0; r < tbl.get_rows(); ++r) {
		for (size_t c = 0; c < tbl.get_cols(); ++c) {
			size_t pos = tbl_to_idx(r, c, row_bits, col_bits);
			TableItem val = tbl.at(r, c);

			if (val.inf()) {
				s.at(pos) = '-';
			} else if (val.get_bit(bit)) {
				s.at(pos) = '1';
			} else {
				s.at(pos) = '0';
			}
		}
	}
	std::cerr << "encoded: " << s << '\n';
}

size_t Automat::tbl_to_idx(size_t r, size_t c, size_t rb, size_t cb) const
{
	(void) rb;
	return (r << cb) + c;
}

