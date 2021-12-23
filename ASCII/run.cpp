/**
 *	Author:		antikostya
 *	Created:	2021-12-23 23:04:18
 *	Modified:	2021-12-23 23:58:47
 **/

#include <iostream>
#include <string>
#include <fstream>
#include <unistd.h>

using namespace::std;

int main()
{
	ifstream	file("compiled.txt");
	string		line;
	string		block;

	if (not file.is_open())
	{
		cout << "cannot open file\n";
		return 1;
	}
	while (not file.eof())
	{
		getline(file, line, '\n');
		if (line.empty())
		{
			cout << block;
			block.clear();
			cout << "\033c";
			usleep(10000); // 100 FPS
			continue ;
		}
		block += line + "\n";
	}
	file.close();
	return 0;
}
