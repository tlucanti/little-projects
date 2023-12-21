
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

	std::cout << "TABLE DUMP\n";
	Table(mt).dump();
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

std::pair<int, int> Automat::run_encoded(int input, int state)
{
	if (!encoded) {
		panic("cannot run before encoding");
	}

	int state_bits = ilog2(transition.get_cols() - 1) + 1;

	unsigned int bitv = ((input - 1) << state_bits) | (state - 1);
	unsigned int next_state = 0;
	unsigned int next_output = 0;

	for (const auto &dt : D) {
		std::cout << "state " << next_state;
		next_state <<= 1;
		next_state |= dt(bitv);
		std::cout << " -> " << dt(bitv) << ' ' << next_state << '\n';
	}

	for (const auto &yt : y) {
		//std::cout << "out " << next_output;
		next_output <<= 1;
		next_output |= yt(bitv);
		//std::cout << " -> " << yt(bitv) << ' ' << next_output << '\n';
	}

	return { next_state, next_output };
}

std::pair<int, int> Automat::run_table(int input, int state)
{
	int next_state = transition.at(input - 1, state - 1).get() + 1;
	int next_outp = output.at(input - 1, state - 1).get() + 1;

	return { next_state, next_outp };
}

void Automat::_check_word(const std::vector<int> &inputs,
			  const std::vector<int> &states,
			  const std::vector<int> &outputs,
			  bool table)
{
	std::pair<int, int> next;

	if (states.size() != inputs.size() || inputs.size() != outputs.size()) {
		panic("check word length mismatch");
	}

	for (size_t i = 0; i < states.size(); ++i) {
		if (table) {
			next = run_table(inputs.at(i), states.at(i));
		} else {
			next = run_encoded(inputs.at(i), states.at(i));
		}

		std::cout << "state: " << states.at(i)
			  << " (" << next.first << "), ";
		std::cout << "oputput: " << outputs.at(i)
			  << " (" <<  next.second - 1 << ")\n";
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

	std::cout << "NR TRIGGERS:" << nr_triggers << '\n';
	tbl.dump(true);
	trig.resize(nr_triggers);
	for (int tr = 0; tr < nr_triggers; ++tr) {
		encode_trigger(tbl, trig.at(tr), tr);
	}
}

void Automat::encode_trigger(const Table &tbl, std::string &s, int bit)
{
	int row_bits = ilog2(tbl.get_rows() - 1) + 1;
	int col_bits = ilog2(tbl.get_cols() - 1) + 1;
	size_t bsize = 1u << (col_bits + row_bits);

	std::cout << "encoding trigger size " << row_bits << ' ' << col_bits
		  << ' ' << bsize << '\n';
	s.resize(bsize, '-');

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

