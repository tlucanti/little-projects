
#include "lexer.hpp"

Literal::Literal(long value) : value(value) {}

long
Literal::compute() const override
{
	return value;
}

Lexema *
Literal::clone() const override
{
	return new Literal(value);
}

void
Literal::print(bool) const override
{
	std::cout << value;
}

Identifier::Identifier(const std::string &name) : name(name) {}

long
Identifier::compute() const override
{
	return Identifier::get_identifier(name);
}

Lexema *
Identifier::clone() const override
{
	return new Identifier(name);
}

void
Identifier::print(bool subs) const override
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
static Identifier::add_identifier(const std::string &name, long value)
{
	Identifier::identifiers[name] = value;
}

long
static Identifier::get_identifier(const std::string &name)
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
Operator::compute() const override
{
	panic("cannot compute operator");
}

Lexema *
Operator::clone() const override {
	return new Operator(type);
}

void
Operator::print(bool) const override
{
	std::cout << type;
}

Expression::Expression() {}

long
Expression::compute() const override
{
	panic("ENOSYS");
}

Lexema *
Expression::clone() const override
{
	Expression *e = new Expression;

	e->expression = expression;
	return e;
}

void
Expression::print(bool subs) const override
{
	std::cout << '(';
	for (const auto &i : expression) {
		std::cout << i.print(subs) << ' ';
	}
	std::cout << ") "
}

void
Expression::print(bool subs, const std::string &lvalue) const
{
	std::cout << lvalue << " = ";
	for (const auto &i : expression) {
		std::cout << i.print(subs) << ' ';
	}
	std::cout << ";\n";
}

void
Expression::add_lexema(const Lexema &l)
{
	expression.push_back(l.clone());
}

