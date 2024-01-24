
#include "lexer.hpp"
#include "util.hpp"

#include <iostream>
#include <stack>

std::map<std::string, long> Identifier::identifiers;

POLIZ::POLIZ(const std::vector<Lexema *> &pz) : pz(pz) {}

const std::vector<Lexema *> &
POLIZ::poliz() const
{
	return pz;
}

long
POLIZ::compute() const
{
	panic("ENOSYS");
}

void
POLIZ::print(bool subs) const
{
	for (const Lexema *i : pz) {
		i->print(subs);
		std::cout << ' ';
	}
}

Literal::Literal(long value) : value(value) {}

Lexema::lex_type
Literal::type() const
{
	return Lexema::lex_literal;
}

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

Lexema::lex_type
Identifier::type() const
{
	return Lexema::lex_identifier;
}

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

Operator::Operator(char type) : _type(type) {}

Lexema::lex_type
Operator::type() const
{
	return Lexema::lex_operator;
}

long
Operator::compute() const
{
	panic("cannot compute operator");
}

Lexema *
Operator::clone() const
{
	return new Operator(_type);
}

void
Operator::print(bool) const
{
	std::cout << _type;
}

int
Operator::priority() const {
	switch (_type) {
	case '+':
	case '-':
		return 1;
	case '*':
	case '/':
		return 2;
	default:
		panic("BUG");
	}
}

Expression::Expression() {}

Lexema::lex_type
Expression::type() const
{
	return Lexema::lex_expression;
}

long
Expression::compute() const
{
	return poliz().compute();
}

POLIZ
Expression::poliz() const
{
	std::stack<Operator *> operator_stack;
	std::vector<Lexema *> out;

	for (size_t i = 0; i < expression.size(); ) {
		Lexema *lx = expression.at(i);

		switch (lx->type()) {
		case Lexema::lex_expression: {
			auto ex = dynamic_cast<Expression *>(lx);
			POLIZ pz = ex->poliz();

			out.insert(out.end(), pz.poliz().begin(), pz.poliz().end());
			i++;
			break;
		} case Lexema::lex_literal:
		  case Lexema::lex_identifier: {
			out.push_back(static_cast<Lexema *>(lx));
			i++;
			break;
		} case Lexema::lex_operator: {
			Operator *op = dynamic_cast<Operator *>(lx);

			if (operator_stack.empty() or
			    op->priority() > operator_stack.top()->priority()) {
				operator_stack.push(op);
				i++;
			} else {
				out.push_back(operator_stack.top());
				operator_stack.pop();
			}
			break;
		} default:
			panic("BUG");
		}
	}

	while (not operator_stack.empty()) {
		out.push_back(operator_stack.top());
		operator_stack.pop();
	}

	return POLIZ(out);
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

