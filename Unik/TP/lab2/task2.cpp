
#include <string>
#include <iterator>
#include <iostream>
#include <sstream>

std::string replace(std::string str, const std::string& oldSub, const std::string& newSub){
    size_t pos = 0;

    while ((pos = str.find(oldSub, pos)) != std::string::npos){
         str.replace(pos, oldSub.length(), newSub);
         pos += newSub.length();
    }
    return str;
}

int words(const std::string& str) {
    std::stringstream s(str);
    std::istream_iterator<std::string> begin(s), end;
    return std::distance(begin, end);
}

void task(const std::string &s, char c)
{
	std::string from = { c };
	std::string to = from + ',';
	std::cout << "original: " << s << '\n';
	std::cout << "replaced: " << replace(s, from, to) << '\n';
	std::cout << "words: " << words(s) << '\n';
}

int main()
{
	task("1111", '1');
	task("q w e r", ' ');
}

