
#include <vector>
#include <thread>
#include <stdio.h>
#include <exception>
#include <locale.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include <omp.h>

using namespace std;

// перечисление, определяющее как будет происходить вычисление
// средних значений матрицы: по строкам или по столбцам
enum class eprocess_type
{
	by_rows = 0,
	by_cols
};

void InitMatrix(double **matrix, const size_t numb_rows, const size_t numb_cols)
{
	ifstream myfile;
	myfile.open("matrix.txt");
	if (myfile.is_open())
	{
		for (size_t i = 0; i < numb_rows; ++i)
		{
			for (size_t j = 0; j < numb_cols; ++j)
			{
				myfile >> matrix[i][j];
			}
		}
	}
	myfile.close();
}

// Функция PrintMatrix() печатает элементы матрицы <i>matrix</i> на консоль;
// numb_rows - количество строк в исходной матрице <i>matrix</i>
// numb_cols - количество столбцов в исходной матрице <i>matrix</i>
void PrintMatrix(double **matrix, const size_t numb_rows, const size_t numb_cols)
{
	printf("Generated matrix:\n");
	for (size_t i = 0; i < numb_rows; ++i)
	{
		for (size_t j = 0; j < numb_cols; ++j)
		{
			printf("%lf ", matrix[i][j]);
		}
		printf("\n");
	}
}

// Функция FindAverageValues() находит средние значения в матрице <i>matrix</i>
// по строкам, либо по столбцам в зависимости от значения параметра <i>proc_type</i>;
// proc_type - признак, в зависимости от которого средние значения вычисляются
// либо по строкам, либо по стобцам исходной матрицы <i>matrix</i>
// matrix - исходная матрица
// numb_rows - количество строк в исходной матрице <i>matrix</i>
// numb_cols - количество столбцов в исходной матрице <i>matrix</i>
// average_vals - массив, куда сохраняются вычисленные средние значения
void FindAverageValues(eprocess_type proc_type, double **matrix, const size_t numb_rows, const size_t numb_cols, double *average_vals, double *true_vals)
{
	ifstream result1, result2;

	switch (proc_type)
	{
	case eprocess_type::by_rows:
	{
		result1.open("result_rows.txt");
		if (not result1.is_open()) {
			throw std::runtime_error("cannot open 'result_rows.txt'");
		}
		double *true_vals_in_rows = new double[numb_rows];
		// #pragma omp parallel for collapse(2) reduction(+ : sum) num_threads(6)
		// #pragma omp parallel for collapse(2) num_threads(6)
		// #pragma omp simd collapse(2)
		for (size_t i = 0; i < numb_rows; ++i)
		{
			double sum(0.0);
			// #pragma omp parallel for reduction(+ : sum) num_threads(6)
			// #pragma omp parallel for reduction(+ : sum)
			for (size_t j = 0; j < numb_cols; ++j)
			{
				sum += matrix[i][j];
			}
			average_vals[i] = sum / numb_cols;
			result1 >> true_vals_in_rows[i];
			true_vals[i] = true_vals_in_rows[i];
		}
		result1.close();
		break;
	}
	case eprocess_type::by_cols:
	{
		result2.open("result_cols.txt");
		if (not result2.is_open()) {
			throw std::runtime_error("cannot open 'result_cols.txt'");
		}
		double *true_vals_in_cols = new double[numb_cols];
		// #pragma omp parallel for
		// #pragma omp parallel for collapse(2) num_threads(6)
		// #pragma omp simd collapse(2)
		for (size_t j = 0; j < numb_cols; ++j)
		{
			double sum(0.0);
			// #pragma omp parallel for reduction(+ : sum) num_threads(6)
			// #pragma omp parallel for reduction(+ : sum)
			for (size_t i = 0; i < numb_rows; ++i)
			{
				sum += matrix[i][j];
			}
			average_vals[j] = sum / numb_rows;
			result2 >> true_vals_in_cols[j];
			true_vals[j] = true_vals_in_cols[j];
		}
		result2.close();
		break;
	}
	default:
	{
		throw("Incorrect value for parameter 'proc_type' in function FindAverageValues() call!");
	}
	}
}

void CheckValues(double *average_vals, double *true_vals, const size_t counter)
{
	for (size_t i = 0; i < counter; ++i)
	{
		if (std::abs(average_vals[i] - true_vals[i]) > 1e-9)
		{
			printf("average_vals[%zu]: %lf \n", i, average_vals[i]);
			printf("true_vals[%zu]: %lf \n", i, true_vals[i]);
			printf("Error! CheckValues\n");
			std::abort();
		}
	}
	printf("Complite! CheckValues\n");
	return;
}

int main()
{
	const unsigned ERROR_STATUS = -1;
	const unsigned OK_STATUS = 0;
	clock_t start, stop;
	unsigned status = OK_STATUS;

	try
	{
		srand((unsigned)time(0));

		const size_t numb_rows = 1000;
		const size_t numb_cols = 1000;
		start = clock();
		double **matrix = new double *[numb_rows];
		for (size_t i = 0; i < numb_rows; ++i)
		{
			matrix[i] = new double[numb_cols];
		}

		double *average_vals_in_rows = new double[numb_rows];
		double *average_vals_in_cols = new double[numb_cols];

		double *true_vals_in_rows = new double[numb_rows];
		double *true_vals_in_cols = new double[numb_cols];

		InitMatrix(matrix, numb_rows, numb_cols);

		// PrintMatrix(matrix,numb_rows, numb_cols);

		// #pragma omp simd
		FindAverageValues(eprocess_type::by_rows, matrix, numb_rows, numb_cols, average_vals_in_rows, true_vals_in_rows);
		FindAverageValues(eprocess_type::by_cols, matrix, numb_rows, numb_cols, average_vals_in_cols, true_vals_in_cols);

		CheckValues(average_vals_in_rows, true_vals_in_rows, numb_rows);
		CheckValues(average_vals_in_cols, true_vals_in_cols, numb_cols);
		stop = clock();
		cout << endl
			 << "Calculations took " << ((double)(stop - start)) / CLOCKS_PER_SEC << " seconds.\n";

		delete[] matrix;
		delete[] average_vals_in_rows;
		delete[] average_vals_in_cols;

		delete[] true_vals_in_rows;
		delete[] true_vals_in_cols;
	}
	catch (std::exception &except)
	{
		printf("Error occured!\n");
		std::cout << except.what();
		status = ERROR_STATUS;
	}

	return status;
}
