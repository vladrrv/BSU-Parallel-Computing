#include <iostream>
#include <fstream>
#include <ctime>
#include <chrono>
#include <conio.h>
#include <omp.h>
using namespace std;

typedef unsigned long long ull;

void save_matrix(double *matrix, ull m, ull n) {
	char name[256];
	cout << "\nEnter file name: ";
	cin >> name;
	cout << "Saving... ";
	ofstream output_file(name);
	output_file << m << " " << n;
	for (ull i = 0; i < m; ++i) {
		output_file << endl;
		for (ull j = 0; j < n; ++j) {
			output_file << matrix[i*n+j] << " ";
		}
	}
	cout << "Done!";
	output_file.close();
}
void load_matrix(double *&matrix, ull &m, ull &n) {
	char path[256];
	ifstream input_file;
	while (true) {
		cout << "\nSpecify input file path: ";
		cin >> path;
		input_file = ifstream(path);
		if (input_file.is_open()) break;
		else cout << "\nError opening file!";
	}

	input_file >> m >> n;
	matrix = new double[m * n];
	cout << "Loading... ";
	for (ull i = 0; i < m; ++i) {
		for (ull j = 0; j < n; ++j) {
			input_file >> matrix[i * n + j];
		}
	}
	cout << "Done!";
}
void generate_matrix(double *&matrix, ull m, ull n) {
	cout << "\nGenerating... ";
	matrix = new double[m * n];
	for (ull i = 0; i < m; ++i) {
		for (ull j = 0; j < n; ++j) {
			matrix[i * n + j] = rand() % 201 - 100;
		}
	}
	cout << "Done!";
}
void print_matrix(const char* title, double* matrix, ull m, ull n) {
	cout << "\n " << title << ":";
	cout << "\n------------------------------------------------";
	for (ull i = 0; i < m; ++i) {
		cout << endl;
		for (ull j = 0; j < n; ++j) {
			cout << matrix[i * n + j] << " ";
		}
	}
	cout << "\n------------------------------------------------";
}


#define MIN(a,b) (((a) < (b))? (a): (b))

void dot_product(double* A, double* B, double* C, ull n1, ull n2, ull n3, ull r, ull ib, ull jb, ull kb) {
	ull start_A = ib * r, end_A = MIN(start_A + r, n1);
	ull start_B = jb * r, end_B = MIN(start_B + r, n3);
	ull start_d = kb * r, end_d = MIN(start_d + r, n2);

	for (ull i = start_A; i < end_A; ++i) {
		for (ull j = start_B; j < end_B; ++j) {
			for (ull k = start_d; k < end_d; ++k) {
				C[i * n3 + j] += A[i * n2 + k] * B[k * n3 + j];
			}
		}
	}
}

void block_product(double* A, double* B, double* C, ull n1, ull n2, ull n3, ull r, ull threads) {
	ull num_blocks1 = n1 / r + ((n1 % r == 0) ? 0 : 1);
	ull num_blocks2 = n2 / r + ((n2 % r == 0) ? 0 : 1);
	ull num_blocks3 = n3 / r + ((n3 % r == 0) ? 0 : 1);

	omp_set_num_threads(threads);
#pragma omp parallel for collapse(2)
	for (ull ib = 0; ib < num_blocks1; ++ib) {
		for (ull jb = 0; jb < num_blocks3; ++jb) {
			//printf("i = %d, j= %d, threadId = %d \n", ib, jb, omp_get_thread_num());
			for (ull kb = 0; kb < num_blocks2; ++kb) {
				dot_product(A, B, C, n1, n2, n3, r, ib, jb, kb);
			}
		}
	}
}


int main(int argc, char* argv[]) {
	ull n1, n2, n3, r, threads;
	double* A;
	double* B;
	double* C;
	if (argc >= 6) {
		n1 = atoi(argv[1]);
		n2 = atoi(argv[2]);
		n3 = atoi(argv[3]);
		r = atoi(argv[4]);
		threads = atoi(argv[5]);
		cout << "n1 = " << n1 << "; n2 = " << n2 << "; n3 = " << n3 << "\nr = " << r << "; threads = " << threads;
		cout << "\nGenerate A";
		generate_matrix(A, n1, n2);
		cout << "\nGenerate B";
		generate_matrix(B, n2, n3);
	}
	else {
		r = 2;
		threads = 4;
		cout << "\nPress Spacebar to generate A and B, press any other key to load A and B from files: ";
		if (_getch() == ' ') {
			cout << "\nSpecify n1, n2, n3: ";
			cin >> n1 >> n2 >> n3;
			cout << "\nGenerate A";
			generate_matrix(A, n1, n2);
			cout << "\nGenerate B";
			generate_matrix(B, n2, n3);
		}
		else {
			cout << "\nLoad A";
			load_matrix(A, n1, n2);
			cout << "\nLoad B";
			load_matrix(B, n2, n3);
		}
		cout << "\nPrint A and B? y/[n] ";
		if (_getch() == 'y') {
			print_matrix("A", A, n1, n2);
			print_matrix("B", B, n2, n3);
		}
	}
	C = new double[n1 * n3];
	fill_n(C, n1 * n3, 0);
	
	auto start = chrono::high_resolution_clock::now();

	block_product(A, B, C, n1, n2, n3, r, threads);

	auto finish = chrono::high_resolution_clock::now();
	chrono::duration<double> elapsed = finish - start;
	int total = elapsed.count() * 1000;
	cout << "\nElapsed time: " << total << " ms\n";


	if (argc != 6) {
		cout << "\nPrint result? y/[n] ";
		if (_getch() == 'y') {
			print_matrix("C", C, n1, n3);
		}

		cout << "\nSave A, B, C? y/[n] ";
		if (_getch() == 'y') {
			cout << "\nSave A";
			save_matrix(A, n1, n2);
			cout << "\nSave B";
			save_matrix(B, n2, n3);
			cout << "\nSave C";
			save_matrix(C, n1, n3);
		}
	}
	return total;
}