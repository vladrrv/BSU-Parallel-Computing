#include <mpi.h>
#include <iostream>
#include <fstream>
#include <ctime>
#include <chrono>
#include <conio.h>
using namespace std;


#define N_TAG 102
#define MATRIX_TAG 103


void generate_matrix(double*& matrix, int n) {
	cout << "\nGenerating... ";
	matrix = new double[n * (n + 1)];
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			double a = (i==j)? 100 : rand()%100;
			matrix[i*(n+1)+j] = a;
		}
		matrix[i*(n+1)+n] = 1;
	}
	cout << "Done!";
}

void read_matrix(double *&matrix, int &n, char* name=NULL) {
	ifstream input_file;
	while (true) {
		if (name == NULL) {
			name = new char[256];
			cout << "\nSpecify input file name: ";
			cin >> name;
		}
		input_file = ifstream(name);
		if (input_file.is_open()) break;
		else {
			cout << "\nError opening file!";
			name = NULL;
		}
	}

	cout << "\nReading file... ";
	input_file >> n;
	matrix = new double[n * (n + 1)];
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n + 1; ++j) {
			input_file >> matrix[i * (n+1) + j];
		}
	}
	cout << "Done!";
}

void print_matrix(const char* title, double* matrix, int n, int m) {
	cout << "\n " << title << ":\n";
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < m; ++j) {
			cout << " " << matrix[i * m + j];
		}
		cout << endl;
	}
}

void save_matrix(double* matrix, int n, int m, char* name=NULL) {
	if (name == NULL) {
		name = new char[256];
		cout << "\nEnter file name: ";
		cin >> name;
	}
	ofstream output_file(name);
	cout << "\nSaving... ";
	output_file << n << " " << m;
	for (int i = 0; i < n; ++i) {
		output_file << endl;
		for (int j = 0; j < m; ++j) {
			output_file << matrix[i*m+j] << " ";
		}
	}
	output_file.close();
	cout << "Done!";
}

void save_time(int total, const char* name="time.json") {
	ofstream output_file(name);
	output_file << "{ \"total\" : " << total << " }";

	output_file.close();
}

int n, r1, r2, r3, r3_, q1, q2, q3;


double* split_A(double* A, double* Aq, int process_id, int num_processes) {
	
	if (process_id == 0) {
		// Init A0
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < r3_; ++j) {
				Aq[i*r3_ + j] = A[i*(n + 1) + j];
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
			MPI_Recv(&(Aq[i*r3_]), r3_, MPI_DOUBLE, 0, MATRIX_TAG, MPI_COMM_WORLD, &status);
		}
	}
	return Aq;
}

void merge_A(double* A, double* Aq, int process_id, int num_processes) {

	if (process_id == 0) {
		// Merge A0
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < r3_; ++j) {
				A[i*(n+1) + j] = Aq[i*r3_ + j];
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
			MPI_Send(&(Aq[i*r3_]), r3_, MPI_DOUBLE, 0, MATRIX_TAG, MPI_COMM_WORLD);
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

	for (int j = 1 + j_gl*r3, jp = 0; (j <= n + 1) && (j <= r3_+j_gl*r3); ++j, ++jp) {
		for (int k = 1, kp = 0; (k <= i - 1) && (k <= j); ++k, ++kp) {
			if (k == j) {
				int s = kp - j_gl*r3;
				l[kp] = Ap[ip*r3_ + s] / Ap[kp*r3_ + s];
			}
			Ap[ip*r3_ + jp] -= l[kp] * Ap[kp*r3_ + jp];
		}
	}
	
	if (process_id < num_processes - 1) {
		MPI_Send(l, ip, MPI_DOUBLE, process_id+1, MATRIX_TAG, MPI_COMM_WORLD);
	}
}

void gauss_forward(double* Ap, int process_id, int num_processes) {
	if (process_id == 0) {
		cout << "\nP0: Gauss forward pass ...";
	}
	for (int i_gl = 0; i_gl <= q2-1; ++i_gl) {
		tile(Ap, i_gl, process_id, num_processes);
	}
}


void gauss_backward(double* A, double* x) {

	cout << "\nP0: Gauss backward pass ...";

	x[n-1] = A[(n-1)*(n+1) + n] / A[(n-1)*(n+1) + n-1];
	for (int i = n-2; i >= 0; --i) {
		x[i] = A[i*(n+1) + n];
		for (int j = n - 1; j >= i + 1; --j) {
			x[i] -= A[i*(n+1) + j] * x[j];
		}
		x[i] /= A[i*(n+1) + i];
	}

}


char DEFAULT_MATRIX_FILE[] = "A.txt";
char DEFAULT_RESULT_FILE[] = "x.txt";


int main(int argc, char* argv[])
{
	MPI_Init(&argc, &argv);

	int process_id, num_processes;
	MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
	MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

	double* A = nullptr;
	bool dump_results = false;
	bool external = argc >= 2;

	if (process_id == 0) {
		if (external) {
			n = atoi(argv[1]);
			dump_results = atoi(argv[2]);
			cout << "\n________________________________";
			cout << "\nn = " << n << "; num_processes = " << num_processes;
			generate_matrix(A, n);
			if (dump_results) {
				save_matrix(A, n, n + 1, DEFAULT_MATRIX_FILE);
			}
		}
		else {
			cout << "\nPress 'y' to generate input or press any other key to read file";
			if (_getch() == 'y') {
				cout << "\nEnter n: ";
				cin >> n;
				generate_matrix(A, n);
				cout << "\nSave generated input? (y/[n])";
				if (_getch() == 'y') {
					save_matrix(A, n, n + 1, DEFAULT_MATRIX_FILE);
				}
			}
			else {
				read_matrix(A, n, DEFAULT_MATRIX_FILE);
			}
		}
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
	r3_ = ((process_id+1) * r3 > n+1) ? (n+1)%r3 : r3;

	clock_t begin;
	begin = clock();
	double* Aq = new double[n * r3_];
	split_A(A, Aq, process_id, num_processes);
	MPI_Barrier(MPI_COMM_WORLD);
	gauss_forward(Aq, process_id, num_processes);
	MPI_Barrier(MPI_COMM_WORLD);
	merge_A(A, Aq, process_id, num_processes);

	if (process_id == 0) {
		
		clock_t end = clock();
		int elapsed_ms = int(double(end - begin) * 1000.0 / CLOCKS_PER_SEC);

		cout << "\nElapsed time: " << elapsed_ms << " ms";

		double* x = new double[n];
		gauss_backward(A, x);

		if (external) {
			if (dump_results) {
				save_matrix(x, n, 1, DEFAULT_RESULT_FILE);
			}
			save_time(elapsed_ms);
		}
		else {
			cout << "\nSave result? (y/[n])";
			if (_getch() == 'y') {
				save_matrix(x, n, 1, DEFAULT_RESULT_FILE);
			}
			cout << "\nPrint result? y/[n]";
			if (_getch() == 'y') {
				print_matrix("Result", x, n, 1);
			}
			system("pause");
		}
	}

	MPI_Finalize();
	return 0;
}