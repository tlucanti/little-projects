
#include "parser.hpp"
#include "util.hpp"

#include <iostream>

int main()
{
	try {
		parse(std::cin);
	} catch (const TranslationError &e) {
		std::cerr << "\nTranslation error: " << e.what() << '\n';
	} catch (const TranslationEOF &e) {
		std::cout << "\nTranslation EOF\n";
		return 0;
	}
}

