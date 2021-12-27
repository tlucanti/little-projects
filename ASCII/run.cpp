/**
 *	Author:		antikostya
 *	Created:	2021-12-23 23:04:18
 *	Modified:	2021-12-24 00:29:52
 **/

#include <iostream>
#include <string>
#include <fstream>
#include <unistd.h>

using namespace::std;

int main(int argc, char **argv)
{
	char *fname = argv[1];

	if (argc == 1)
		fname = (char *)"compiled.txt";
	ifstream	file(fname);
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
			write(1, block.c_str(), block.size());
			block.clear();
			// cout << "\033c";
			usleep(20000); // 80 FPS
			continue ;
		}
		block += line + "\n";
	}
	file.close();
	return 0;
}
