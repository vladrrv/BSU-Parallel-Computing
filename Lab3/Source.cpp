#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <ctime>
#include <chrono>
#include <conio.h>
using namespace std;


#define N_TAG 102
#define MATRIX_TAG 103


struct timer {

	clock_t begin;
	int accumulated = 0;
	bool started = false;

	void start() {
		begin = clock();
		if (started) {
			accumulated = 0;
		}
		else {
			started = true;
		}
	}

	void pause() {
		if (started) {
			accumulated = elapsed();
			started = false;
		}
	}

	int elapsed() {
		if (started) {
			clock_t end = clock();
			int elapsed_ms = int(double(end - begin) * 1000.0 / CLOCKS_PER_SEC);
			return accumulated + elapsed_ms;
		}
		return accumulated;
	}

};


void generate_matrix(double*& matrix, int n) {
	cout << "\nGenerating... ";
	matrix = new double[n * (n + 1)];
	for (int i = 0; i < n; ++i) {
		double sum = 0;
		for (int j = 0; j < n; ++j) {
			double a = (i==j)? 100 : (double)(2*i+j)/100000;
			matrix[i*(n+1)+j] = a;
			sum += a;
		}
		matrix[i*(n+1)+n] = sum;
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
			output_file << setprecision(12) << fixed << matrix[i*m+j] << " ";
		}
	}
	output_file.close();
	cout << "Done!";
}

void save_time(int total, int* computation, int* communication, int backward, int num_processes, const char* name="time.json") {
	ofstream output_file(name);
	output_file << "{ \"total\" : " << total;

	for (int i = 0; i < num_processes; ++i) {
		output_file <<
			", \"computation_" << i << "\" : " << computation[i] <<
			", \"communication_" << i << "\" : " << communication[i];
	}
	output_file << ", \"backward\" : " << backward << " }";

	output_file.close();
}


timer total, computation, communication, backward;
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
		communication.start();
		for (int p = 1; p < num_processes; ++p) {
			int num = ((p+1)*r3 > n+1) ? (n+1)%r3 : r3;
			for (int i = 0; i < n; ++i) {
				MPI_Send(&(A[i*(n+1) + p*r3]), num, MPI_DOUBLE, p, MATRIX_TAG, MPI_COMM_WORLD);
			}
		}
		communication.pause();
	}
	else {
		communication.start();
		for (int i = 0; i < n; ++i) {
			MPI_Status status;
			MPI_Recv(&(Aq[i*r3_]), r3_, MPI_DOUBLE, 0, MATRIX_TAG, MPI_COMM_WORLD, &status);
		}
		communication.pause();
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
		communication.start();
		for (int p = 1; p < num_processes; ++p) {
			int num = ((p+1)*r3 > n+1) ? (n+1)%r3 : r3;
			for (int i = 0; i < n; ++i) {
				MPI_Status status;
				MPI_Recv(&(A[i*(n+1) + p*r3]), num, MPI_DOUBLE, p, MATRIX_TAG, MPI_COMM_WORLD, &status);
			}
		}
		communication.pause();
	}
	else {
		communication.start();
		for (int i = 0; i < n; ++i) {
			MPI_Send(&(Aq[i*r3_]), r3_, MPI_DOUBLE, 0, MATRIX_TAG, MPI_COMM_WORLD);
		}
		communication.pause();
	}
}


void tile(double* Ap, int i_gl, int process_id, int num_processes) {
	int i = i_gl + 2;
	int j_gl = process_id;

	int ip = i - 1;

	double* l = new double[ip];
	if (process_id > 0) {
		communication.start();
		MPI_Status status;
		MPI_Recv(l, ip, MPI_DOUBLE, process_id-1, MATRIX_TAG, MPI_COMM_WORLD, &status);
		communication.pause();
	}

	computation.start();
	for (int j = 1 + j_gl*r3, jp = 0; (j <= n + 1) && (j <= r3_+j_gl*r3); ++j, ++jp) {
		for (int k = 1, kp = 0; (k <= i - 1) && (k <= j); ++k, ++kp) {
			if (k == j) {
				int s = kp - j_gl*r3;
				l[kp] = Ap[ip*r3_ + s] / Ap[kp*r3_ + s];
			}
			Ap[ip*r3_ + jp] -= l[kp] * Ap[kp*r3_ + jp];
		}
	}
	computation.pause();
	
	if (process_id < num_processes - 1) {
		communication.start();
		MPI_Send(l, ip, MPI_DOUBLE, process_id+1, MATRIX_TAG, MPI_COMM_WORLD);
		communication.pause();
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

	backward.start();
	x[n-1] = A[(n-1)*(n+1) + n] / A[(n-1)*(n+1) + n-1];
	for (int i = n-2; i >= 0; --i) {
		x[i] = A[i*(n+1) + n];
		for (int j = n - 1; j >= i + 1; --j) {
			x[i] -= A[i*(n+1) + j] * x[j];
		}
		x[i] /= A[i*(n+1) + i];
	}
	backward.pause();

}


char DEFAULT_MATRIX_FILE[] = "A.txt";
char DEFAULT_RESULT_FILE[] = "x.txt";


int main(int argc, char* argv[])
{
	MPI_Init(&argc, &argv);
	

	int process_id, num_processes;
	MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
	MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

	int* computation_times = new int[num_processes];
	int* communication_times = new int[num_processes];

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
		total.start();
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


	double* Aq = new double[n * r3_];
	split_A(A, Aq, process_id, num_processes);
	MPI_Barrier(MPI_COMM_WORLD);
	gauss_forward(Aq, process_id, num_processes);
	MPI_Barrier(MPI_COMM_WORLD);
	merge_A(A, Aq, process_id, num_processes);

	int time_computation = computation.elapsed();
	int time_communication = communication.elapsed();
	MPI_Gather(&time_computation, 1, MPI_LONG, computation_times, num_processes, MPI_LONG, 0, MPI_COMM_WORLD);
	MPI_Gather(&time_communication, 1, MPI_LONG, communication_times, num_processes, MPI_LONG, 0, MPI_COMM_WORLD);

	if (process_id == 0) {
		
		double* x = new double[n];

		gauss_backward(A, x);

		int time_backward = backward.elapsed();
		int time_total = total.elapsed();

		cout << "\nP0: Total time: " << time_total << " ms";
		cout << "\nP0: Computation time: " << time_computation << " ms";
		cout << "\nP0: Communication time: " << time_communication << " ms";
		cout << "\nP0: Backward pass time: " << time_backward << " ms";

		if (external) {
			if (dump_results) {
				save_matrix(x, n, 1, DEFAULT_RESULT_FILE);
			}
			save_time(time_total, computation_times, communication_times, time_backward, num_processes);
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