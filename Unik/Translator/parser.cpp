
#include "util.hpp"
#include "lexer.hpp"

int bracket_counter;

class Symbol {
public:
	enum sym_type {
		sym_char		= 0b1000000000,
		sym_digit		= 0x0100000000,
		sym_open_bracket	= 0b0010000000,
		sym_close_bracket	= 0b0001000000,
		sym_semicolon		= 0b0000100000,
		sym_operator		= 0b0000010000,
		sym_assign		= 0b0000001000,
		sym_space		= 0b0000000100,
		sym_eof			= 0b0000000010,
		sym_unknown		= 0b0000000001,
	};

	static constexpr char eof = 0;

	Symbol(char c) : _c(c) {
		switch (c) {
		case '(':
			_type = sym_open_bracket;
			break;
		case ')':
			_type = sym_close_bracket;
			break;
		case ';':
			_type = sym_semicolon;
			break;
		case '=':
			_type = sym_assign;
			break;
		case '+':
		case '-':
		case '*':
		case '/':
			_type = sym_operator;
			break;
		case eof:
			_type = sym_eof;
			break;
		default:
			if (isdigit(c)) {
				_type = sym_digit;
			} else if (isalpha(c)) {
				_type = sym_char;
			} else if (isspace(c)) {
				_type = sym_space;
			} else {
				_type = sym_unknown;
			}
		}
	}

	enum sym_type type() const {
		return _type;
	}

	bool is_type(long t) const {
		return type() & t;
	}

	char get() const {
		return _c;
	}

	long to_digit() const {
		panic_on(not is_type(sym_digit), "cannot convert to digit");
		return static_cast<long>(get() - '0');
	}

	void assert_type(long expected) const
	{
		if (not is_type(expected)) {
			std::cerr << "invalid character " << get();
			panic("translation error");
		}
	}

private:
	char _c;
	enum sym_type _type;

};

Symbol get_sym(std::istream &in, bool skip_spaces=false)
{
	char c;

again:
	if (in >> c) {
		Symbol s(c);

		if (skip_spaces and s.is_type(Symbol::sym_space)) {
			goto again;
		}
		return s;
	} else {
		return Symbol(Symbol::eof);
	}
}

static Expression parse_expression(std::istream &in, Symbol &s)
{
	Expression expression;

	while (true) {
		/**
		 * read next Lexema
		 */
		s.assert_type(Symbol::sym_digit |
			      Symbol::sym_char |
			      Symbol::sym_open_bracket);

		switch (s.type()) {
		case Symbol::sym_open_bracket: {
			bracket_counter++;

			s = get_sym(in, true);
			expression.add_lexema(parse_expression(in, s));

			s = get_sym(in, true);
			break;

		} case Symbol::sym_digit: {
			/**
			 * read integer literal
			 */
			long literal = s.to_digit();

			while (true) {
				s = get_sym(in);
				s.assert_type(Symbol::sym_digit |
					      Symbol::sym_operator |
					      Symbol::sym_semicolon |
					      Symbol::sym_close_bracket |
					      Symbol::sym_space);

				if (not s.is_type(Symbol::sym_digit)) {
					break;
				}

				/* update literal value */
				literal = literal * 10 + s.to_digit();
			}

			/* save read literal to expression */
			Literal li(literal);
			expression.add_lexema(li);

			if (s.is_type(Symbol::sym_space)) {
				s = get_sym(in, true);
			}
			break;
		} case Symbol::sym_char: {
			/**
			 * read identifier
			 */
			std::string name;
			name.push_back(s.get());

			/* get next chars or numbers for variable name */
			while (true) {
				s = get_sym(in);
				s.assert_type(Symbol::sym_char |
					      Symbol::sym_digit |
					      Symbol::sym_operator |
					      Symbol::sym_semicolon |
					      Symbol::sym_close_bracket |
					      Symbol::sym_space);

				if (not s.is_type(Symbol::sym_char | Symbol::sym_digit)) {
					break;
				} else {
					name.push_back(s.get());
				}
			}

			Identifier id(name);
			expression.add_lexema(id);
			name.clear();

			if (s.is_type(Symbol::sym_space)) {
				s = get_sym(in, true);
			}
			break;
		} default:
			panic("BUG");
		}

		switch (s.type()) {
		case Symbol::sym_close_bracket:
			goto close_bracket;
		case Symbol::sym_semicolon:
			goto done;
		case Symbol::sym_operator:
			break;
		default:
			panic("BUG");
		}

		s.assert_type(Symbol::sym_operator);
		Operator op(s.get());
		expression.add_lexema(op);

		s = get_sym(in, true);
	}

close_bracket:
	if (bracket_counter == 0) {
		std::cerr << "unmatched ')'";
		panic("translation error");
	}

done:
	return expression;
}

void parse(std::istream &in)
{
	Symbol s(Symbol::eof);
	std::string name;

	/**
	 * get lvalue before '=' operator
	 */

	/* get first char for identifier name */
	s = get_sym(in, true);
	s.assert_type(Symbol::sym_char);
	name.push_back(s.get());

	/* get next chars or numbers for variable name */
	while (true) {
		s = get_sym(in);
		s.assert_type(Symbol::sym_char |
			      Symbol::sym_digit |
			      Symbol::sym_assign |
			      Symbol::sym_space);

		if (s.is_type(Symbol::sym_char | Symbol::sym_digit)) {
			name.push_back(s.get());
		} else {
			break;
		}
	}

	/* confirm that next character is '=' operator */
	if (s.is_type(Symbol::sym_space)) {
		s = get_sym(in, true);
	}
	s.assert_type(Symbol::sym_assign);

	s = get_sym(in, true);

	Expression expression = parse_expression(in, s);
	expression.print(false, name);
}

