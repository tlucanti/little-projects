
#ifndef _LEXER_HPP_
#define _LEXER_HPP_

#include <string>
#include <vector>
#include <map>

class Lexema {
	virtual long compute() const = 0;
	virtual Lexema *clone() const = 0;
	virtual void print(bool subs) const = 0;
};

class Literal : public Lexema {
	long value;

public:
	Literal(long value);
	long compute() const override;
	Lexema *clone() const override;
	void print(bool subs) const override;
};

class Identifier : public Lexema {
	static std::map<std::string, long> identifiers;
	const std::string name;

public:
	Identifier(const std::string &name);
	long compute() const override;
	Lexema *clone() const override;
	void print(bool subs) const override;

	const std::string &get_name() const;
	static void add_identifier(const std::string &name, long value);
	static long get_identifier(const std::string &name);
};

class Operator : public Lexema {
	char type;

public:
	Operator(char type);
	long compute() const override;
	Lexema *clone() const override;
	void print(bool subs) const override;
};

class Expression : public Lexema {
	std::vector<Lexema *> &expression;

public:
	Expression();
	long compute() const override;
	Lexema *clone() const override;
	void print(bool subs) const override;
	void print(bool subs, const std::string &lvalue) const;

	void add_lexema(const Lexema &l);
};

#endif /* _LEXER_HPP_ */

