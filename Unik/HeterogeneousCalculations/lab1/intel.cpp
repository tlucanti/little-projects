//=======================================================================
//
// SAMPLE SOURCE CODE - SUBJECT TO THE TERMS OF END-USER LICENSE AGREEMENT FOR
// INTEL(R) ADVISOR XE 2013.
//
// Copyright (C) 2009-2011 Intel Corporation. All rights reserved
//
// THIS FILE IS PROVIDED "AS IS" WITH NO WARRANTIES, EXPRESS OR IMPLIED, INCLUDING BUT
// NOT LIMITED TO ANY IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
// PURPOSE, NON-INFRINGEMENT OF INTELLECTUAL PROPERTY RIGHTS.
//
// ========================================================================

// [DESCRIPTION]
// Solve the nqueens problem  - how many positions of queens can fit on a chess board
// of a given board_size without attacking each other.
// This is the serial version used to find a candidate hotspot function to parallelize.
//
// [BUILD]
// Use a Release configuration to ensure you find a hotspot representative of a final production build.
//
// [RUN]
// To set the board board_size in Visual Studio, right click on the project,
// select Properies > Configuration Properties > General > Debugging.  Set
// Command Arguments to the desired value.  14 has been set as the default.
//
// [EXPECTED OUTPUT]
//
// Board board_size   Number of Solutions
//     4                2
//     5               10
//     6                4
//     7               40
//     8               92
//     9              352
//    10              724
//    11             2680
//    12            14200
//    13            73712
//    14           365596
//    15          2279184

#include <iostream>
#include <time.h>


//ADVISOR COMMENT: Uncomment the #include <advisor-annotate.h> line after you've added the annotation header to the project to allow you to use Advisor annotations

//#include <advisor-annotate.h>

using namespace std;

int nrOfSolutions = 0;  //keeps track of the number of solutions
int board_size = 0;  // the board-board_size read from command-line
int correctSolution[16]; //array of the number of correct solutions for each board board_size

/*
Recursive function to find all solutions on a board
represented by the argument "queens", placing the next queen
at location (row, col)

On Return: nrOfSolutions has been increased to add solutions for this board

*/
void setQueen(int queens[], int row, int col) {
	//check all previously placed rows for attacks
	for (int i = 0; i < row; i++) {
		// vertical attacks
		if (queens[i] == col) {
			return;
		}
		// diagonal attacks
		if (abs(queens[i] - col) == (row - i)) {
			return;
		}
	}
	// column is ok, set the queen
	queens[row] = col;

	if (row == board_size - 1) {
		nrOfSolutions++;  //Placed final queen, found a solution
	}
	else {
		// try to fill next row
		for (int i = 0; i < board_size; i++) {
			setQueen(queens, row + 1, i);
		}
	}
}

/*
Function to find all solutions for nQueens problem on board_size x board_size chessboard.

On Return: nrOfSoultions = number of solutions for board_size x board_size chessboard.
*/
void solve() {
	int* queens = new int[board_size]; //array representing queens placed on a chess board.  Index is row position, value is column.

	for (int i = 0; i < board_size; i++) {
		// try all positions in first row
		setQueen(queens, 0, i);
	}
}


int main(int argc, char* argv[]) {
	clock_t start, stop;
	if (argc != 2) {
		cerr << "Usage: 1_nqueens_serial.exe boardboard_size [default is 14].\n";
		board_size = 14;
	}
	else {
		board_size = atoi(argv[1]);
		if (board_size < 4 || board_size > 15) {
			cerr << "Boardboard_size should be between 4 and 15; setting it to 14. \n" << endl;
			board_size = 14;
		}
	}
	// Set the expected number of solutions for each board-board_size for later checking.
	correctSolution[0] = 0;
	correctSolution[1] = 0;
	correctSolution[2] = 0;
	correctSolution[3] = 0;
	correctSolution[4] = 2;
	correctSolution[5] = 10;
	correctSolution[6] = 4;
	correctSolution[7] = 40;
	correctSolution[8] = 92;
	correctSolution[9] = 352;
	correctSolution[10] = 724;
	correctSolution[11] = 2680;
	correctSolution[12] = 14200;
	correctSolution[13] = 73712;
	correctSolution[14] = 365596;
	correctSolution[15] = 2279184;

	cout << "Starting nqueens solver for board_size " << board_size << "...\n";
	start = clock();
	solve();
	stop = clock();
	cout << "Number of solutions: " << nrOfSolutions << endl;
	if (nrOfSolutions != correctSolution[board_size])
		cout << "!!Incorrect result!! Number of solutions should be " << correctSolution[board_size] << endl << endl;
	else
		cout << "Correct result!" << endl;

	cout << endl << "Calculations took " << ((double)(stop - start)) / CLOCKS_PER_SEC << " seconds.\n";

	return 0;
}

