
#include <memory>
#include <stdexcept>
#include <iostream>

template<typename T>
class Stack {

	struct StackNode {
		T value;
		std::unique_ptr<StackNode> next;

		StackNode(T value, std::unique_ptr<StackNode> next)
			: value(value), next(std::move(next)) {}
	};

public:
	std::unique_ptr<StackNode> top;

	void push(T value) {
		top = std::make_unique<StackNode>(value, std::move(top));
	}

	T pop() {
		if (top == nullptr) {
			throw std::out_of_range("Stack is empty, can't pop");
		}

		T value = top->value;
		top = std::move(top->next);
		return value;
	}
};

int main()
{
	Stack<int> s;

	try {
		s.pop();
	} catch (std::out_of_range) {}

	s.push(1);
	s.push(2);
	s.push(3);

	std::cout << s.pop() << ' ' << s.pop() << ' ' << s.pop() << '\n';

}
