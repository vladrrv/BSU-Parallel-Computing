#include <mpi.h>
#include <iostream>
#include <fstream>
#include <ctime>
#include <chrono>
#include <conio.h>
using namespace std;

#define RESULT_TAG 100
#define N_TAG 102
#define MATRIX_TAG 103


void generate_matrix() {
	int n;
	char name[256];
	cout << "\nEnter file name: ";
	cin >> name;
	cout << "\nEnter n: ";
	cin >> n;
	ofstream output_file(name);
	cout << "\nGenerating... ";
	output_file << n << endl;
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			double a = (i==j)? 100 : rand()%100;
			output_file << a << " ";
		}
		output_file << 1 << endl;
	}
	cout << "Done!";
	output_file.close();
}

void read_matrix(double *&matrix, int &n) {

	char path[256];
	ifstream input_file;
	while (true) {
		cout << "\nSpecify input file path: ";
		cin >> path;
		input_file = ifstream(path);
		if (input_file.is_open()) break;
		else cout << "\nError opening file!";
	}

	input_file >> n;
	matrix = new double[n * (n + 1)];
	cout << "\nReading file... ";
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n + 1; ++j) {
			input_file >> matrix[i * (n+1) + j];
		}
	}
	cout << "Done!";
}

void print_matrix(const char* title, double* matrix, int n) {
	cout << "\n " << title << ":\n";
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < n+1; ++j) {
			cout << matrix[i * (n+1) + j] << " ";
		}
		cout << endl;
	}
}


int n, r1, r2, r3, q1, q2, q3;


double* split_A(double* A, double* Aq, int process_id, int num_processes) {
	
	if (process_id == 0) {
		// Init A0
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < r3; ++j) {
				Aq[i*r3 + j] = A[i*(n + 1) + j];
			}
		}
		cout << "\nP0: Sending Aq to each Pq";
		for (int p = 1; p < num_processes; ++p) {
			int num = ((p+1)*r3 > n+1) ? (n+1)%r3 : r3;
			for (int i = 0; i < n; ++i) {
				MPI_Send(&(A[i*(n+1) + p*r3]), num, MPI_DOUBLE, p, MATRIX_TAG, MPI_COMM_WORLD);
			}
		}
	}
	else {
		for (int i = 0; i < n; ++i) {
			MPI_Status status;
			MPI_Recv(&(Aq[i*r3]), r3, MPI_DOUBLE, 0, MATRIX_TAG, MPI_COMM_WORLD, &status);
		}
	}
	return Aq;
}

void merge_A(double* A, double* Aq, int process_id, int num_processes) {

	if (process_id == 0) {
		// Merge A0
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < r3; ++j) {
				A[i*(n+1) + j] = Aq[i*r3 + j];
			}
		}
		cout << "\nP0: Merging Aq from each Pq";
		for (int p = 1; p < num_processes; ++p) {
			int num = ((p+1)*r3 > n+1) ? (n+1)%r3 : r3;
			for (int i = 0; i < n; ++i) {
				MPI_Status status;
				MPI_Recv(&(A[i*(n+1) + p*r3]), num, MPI_DOUBLE, p, MATRIX_TAG, MPI_COMM_WORLD, &status);
			}
		}
	}
	else {
		for (int i = 0; i < n; ++i) {
			MPI_Send(&(Aq[i*r3]), r3, MPI_DOUBLE, 0, MATRIX_TAG, MPI_COMM_WORLD);
		}
	}
}


void tile(double* Ap, int i_gl, int process_id, int num_processes) {
	int i = i_gl + 2;
	int j_gl = process_id;

	int ip = i - 1;

	double* l = new double[ip];
	if (process_id > 0) {
		MPI_Status status;
		MPI_Recv(l, ip, MPI_DOUBLE, process_id-1, MATRIX_TAG, MPI_COMM_WORLD, &status);
	}

	for (int j = 1 + j_gl*r3; (j <= n + 1) && (j <= (1+j_gl)*r3); ++j) {
		int jp = j - 1;
		for (int k = 1; (k <= i - 1) && (k <= j); ++k) {
			int kp = k - 1;
			if (k == j) {
				l[kp] = Ap[ip*r3 + kp] / Ap[kp*r3 + kp];
			}
			Ap[ip*r3 + jp] -= l[kp] * Ap[kp*r3 + jp];
		}
	}
	
	if (process_id < num_processes - 1) {
		MPI_Send(l, ip, MPI_DOUBLE, process_id+1, MATRIX_TAG, MPI_COMM_WORLD);
	}
}

void gauss(double* Ap, int process_id, int num_processes) {
	if (process_id == 0) {
		cout << "\nP0: Computing ...";
	}
	for (int i_gl = 0; i_gl <= q2-1; ++i_gl) {
		tile(Ap, i_gl, process_id, num_processes);
	}
}



void main(int argc, char** argv)
{
	MPI_Init(&argc, &argv);

	int process_id, num_processes;
	MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
	MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

	double* A = nullptr;


	if (process_id == 0) {
		cout << "\nGenerate input file? y/[n]";
		if (_getch() == 'y') {
			generate_matrix();
		}
		read_matrix(A, n);

		cout << "\nP0: Sending n="<< n << " to " << num_processes << " processes";
		for (int p = 1; p < num_processes; p++) {
			MPI_Send(&n, 1, MPI_LONG, p, N_TAG, MPI_COMM_WORLD);
		}
	}
	else {
		MPI_Status status;
		MPI_Recv(&n, 1, MPI_LONG, 0, N_TAG, MPI_COMM_WORLD, &status);
	}

	MPI_Barrier(MPI_COMM_WORLD);

	q1 = 1;
	q2 = n - 1;
	q3 = num_processes;
	r1 = n - 1;
	r2 = 1;
	r3 = ceil((double)(n+1) / q3);
	r3 = ((process_id+1) * r3 > n+1) ? (n+1)%r3 : r3;

	clock_t begin;
	begin = clock();
	double* Aq = new double[n * r3];
	split_A(A, Aq, process_id, num_processes);
	MPI_Barrier(MPI_COMM_WORLD);
	gauss(Aq, process_id, num_processes);
	MPI_Barrier(MPI_COMM_WORLD);
	merge_A(A, Aq, process_id, num_processes);

	if (process_id == 0) {
		
		clock_t end = clock();
		double elapsed_ms = double(end - begin) * 1000.0 / CLOCKS_PER_SEC;
		cout << "\nElapsed time: " << elapsed_ms << " ms";


		cout << "\nPrint matrix? y/[n]";
		if (_getch() == 'y') {
			print_matrix("Result", A, n);
		}

		system("pause");
	}

	MPI_Finalize();
}