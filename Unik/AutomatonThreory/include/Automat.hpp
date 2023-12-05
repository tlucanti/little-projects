
#ifndef AUTOMAT_HPP
#define AUTOMAT_HPP

#include <Table.hpp>
#include <DNF.hpp>

class Automat {
public:
	Automat(const Table &transition, const std::vector<int> &output);
	Automat(const Table &transition, const Table &output);

	void encode(void);
	void minimize(void);
	void dump(void);
	void print(void);
	std::pair<int, int> run_encoded(int state, int input);
	std::pair<int, int> run_table(int state, int input);

	void check_word_table(const std::vector<int> &states,
			      const std::vector<int> &inputs);

    private:
	void init_automat(const Table &trnsition, const Table &output);
	void encode_table(const Table &tbl, std::vector<std::string> &trig);
	void encode_trigger(const Table &tbl, std::string &s, int bit);
	size_t tbl_to_idx(size_t r, size_t c, size_t rb, size_t cb) const;

	Table transition;
	Table output;

	std::vector<DNF> D;
	std::vector<DNF> y;

	bool encoded;
};

#endif /* AUTOMAT_HPP */

