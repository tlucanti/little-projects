
#include <string>
#include <cstdio>
#include <ctime>
#include <stdexcept>
#include <iostream>
#include <sstream>

#include <iostream>
#include <sstream>
#include <iomanip>

bool checkLeapYear(int year) {
	if (year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)) {
		return true;
	}
	return false;
}

void check_date(std::string date) {
	int daysInMonth[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
	std::istringstream iss(date);
	std::string day, month, year;
	int dayNum, monthNum, yearNum;

	if (date.size() != 10) {
		goto err;
	}

	getline(iss, day, '-');
	getline(iss, month, '-');
	getline(iss, year, '\0');

	dayNum = std::stoi(day);
	monthNum = std::stoi(month);
	yearNum = std::stoi(year);

	if (yearNum < 1000 || yearNum > 9999 || monthNum < 1 || monthNum > 12) {
		goto err;
	}

	if (checkLeapYear(yearNum)) {
		daysInMonth[1] = 29;
	}
	if (dayNum == 0 || dayNum > daysInMonth[monthNum - 1]) {
		goto err;
	}
	return;

err:
	throw std::invalid_argument(date);
}

void OK(std::string e)
{
	try {
		check_date(e);
		std::cout << "OK\n";
	} catch (std::invalid_argument &) {
		std::cout << "ERR\n";
	}
}

void ERR(std::string e)
{
	try {
		check_date(e);
		std::cout << "ERR\n";
	} catch (std::invalid_argument &) {
		std::cout << "OK\n";
	}
}

int main()
{
	OK("17-07-2001");
	ERR("17.07.2001");
	ERR("17-07-2001qq");
	ERR("99-99-2001qq");
	ERR("---");
	ERR("");
}
