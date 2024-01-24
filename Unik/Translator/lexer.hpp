
#ifndef _LEXER_HPP_
#define _LEXER_HPP_

#include "util.hpp"

#include <string>
#include <vector>
#include <map>

class Lexema {
public:
	enum lex_type {
		lex_literal,
		lex_identifier,
		lex_expression,
		lex_operator,
	};

	virtual enum lex_type type() const = 0;
	virtual long compute() const = 0;
	virtual Lexema *clone() const = 0;
	virtual void print(bool subs) const = 0;
	virtual ~Lexema() {}
};

class POLIZ {
	std::vector<Lexema *> pz;

public:
	POLIZ(const std::vector<Lexema *> &pz);
	const std::vector<Lexema *> &poliz() const;
	long compute() const;
	void print(bool subs) const;
};

class Literal : public Lexema {
	long value;

public:
	Literal(long value);
	enum lex_type type() const override;
	long compute() const override;
	Lexema *clone() const override;
	void print(bool subs) const override;
};

class Identifier : public Lexema {
	static std::map<std::string, long> identifiers;
	const std::string name;

public:
	Identifier(const std::string &name);
	enum lex_type type() const override;
	long compute() const override;
	Lexema *clone() const override;
	void print(bool subs) const override;

	const std::string &get_name() const;
	static void add_identifier(const std::string &name, long value);
	static long get_identifier(const std::string &name);
};

class Operator : public Lexema {
	char _type;

public:
	Operator(char type);
	enum lex_type type() const override;
	long compute() const override;
	Lexema *clone() const override;
	void print(bool subs) const override;

	int priority() const;
};

class Expression : public Lexema {
	std::vector<Lexema *> expression;

public:
	Expression();
	enum lex_type type() const override;
	long compute() const override;
	Lexema *clone() const override;
	void print(bool subs) const override;
	void print(bool subs, const std::string &lvalue) const;

	void add_lexema(const Lexema &l);
	POLIZ poliz() const;
};

#endif /* _LEXER_HPP_ */

