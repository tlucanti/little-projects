
#include <memory>
#include <stdexcept>
#include <iostream>

template<typename T>
class StackNode {
public:
    T value;
    std::unique_ptr<StackNode<T>> next;

    StackNode(T value, std::unique_ptr<StackNode<T>> next)
        : value(value), next(std::move(next)) {}
};

template<typename T>
class Stack {
public:
    std::unique_ptr<StackNode<T>> top;

    void push(T value) {
        top = std::make_unique<StackNode<T>>(value, std::move(top));
    }

    T pop() {
        if (top != nullptr) {
            T value = top->value;
            top = std::move(top->next);
            return value;
        } else {
            throw std::out_of_range("Stack is empty, can't pop");
        }
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
