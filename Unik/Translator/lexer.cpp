
#include "lexer.hpp"
#include "util.hpp"

#include <iostream>

std::map<std::string, long> Identifier::identifiers;

Literal::Literal(long value) : value(value) {}

long
Literal::compute() const
{
	return value;
}

Lexema *
Literal::clone() const
{
	return new Literal(value);
}

void
Literal::print(bool) const
{
	std::cout << value;
}

Identifier::Identifier(const std::string &name) : name(name) {}

long
Identifier::compute() const
{
	return Identifier::get_identifier(name);
}

Lexema *
Identifier::clone() const
{
	return new Identifier(name);
}

void
Identifier::print(bool subs) const
{
	if (subs) {
		std::cout << compute();
	} else {
		std::cout << name;
	}
}

const std::string &
Identifier::get_name() const
{
	return name;
}

void
Identifier::add_identifier(const std::string &name, long value)
{
	Identifier::identifiers[name] = value;
}

long
Identifier::get_identifier(const std::string &name)
{
	auto it = Identifier::identifiers.find(name);

	if (it == Identifier::identifiers.end()) {
		std::cerr << "identifier `" << name << "` is not defined";
		panic("Translation error");
	}
	return it->second;
}

Operator::Operator(char type) : type(type) {}

long
Operator::compute() const
{
	panic("cannot compute operator");
}

Lexema *
Operator::clone() const {
	return new Operator(type);
}

void
Operator::print(bool) const
{
	std::cout << type;
}

Expression::Expression() {}

long
Expression::compute() const
{
	panic("ENOSYS");
}

Lexema *
Expression::clone() const
{
	Expression *e = new Expression;

	e->expression = expression;
	return e;
}

void
Expression::print(bool subs) const
{
	std::cout << '(';
	for (const auto &i : expression) {
		i->print(subs);
		std::cout << ' ';
	}
	std::cout << ") ";
}

void
Expression::print(bool subs, const std::string &lvalue) const
{
	std::cout << lvalue << " = ";
	for (const auto &i : expression) {
		i->print(subs);
		std::cout << ' ';
	}
	std::cout << ";\n";
}

void
Expression::add_lexema(const Lexema &l)
{
	expression.push_back(l.clone());
}

