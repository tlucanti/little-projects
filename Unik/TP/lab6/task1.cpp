
#include <memory>
#include <stdexcept>
#include <iostream>

template <class T>
class Stack {
	struct StackNode {
		StackNode *next;
		T data;

		StackNode(const T &data, StackNode *next)
			: next(next), data(data)
		{}
	};

	StackNode *top;
public:
	Stack()
		: top(nullptr)
	{}

	void push(const T &data)
	{
		auto next = new StackNode(data, top);
		top = next;
	}

	T pop(void)
	{
		if (top == nullptr) {
			throw std::out_of_range("poping from empty stack");
		}
		StackNode *prev = top;
		T ret = top->data;
		top = top->next;
		delete prev;
		return ret;
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
