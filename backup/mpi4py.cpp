//
// Created by Swapnil Das on 2/3/2018.
//

//fix imports
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <set>
#include <iterator>
#include <algorithm>
#include <vector>
#include <math.h>
#include <limits.h>
#include <string>
#include <numeric>
#include <algorithm>
#include <list>
#include <iostream>
#include <utility>
#include <cstdlib>
#include <fstream>
#include <ctime>
#include <chrono>
#include <time.h>
#include <sstream>
#include <limits>

// http://arma.sourceforge.net
// Used to solve linear systems.
#include "/usr/local/include/armadillo"
#include "/usr/lib/openmpi/include/mpi.h"

typedef std::vector<double> stdvec;
typedef std::vector<arma::mat> stdmat;
using namespace std::chrono;

// Given two vectors, a and b, it returns a vector that has elements
// in a that were not present in b.
stdvec diff(stdvec a, stdvec b) {
	/* no need to make a set for b: it does not change the overall result */
	stdvec result; int count = 0;
	for (stdvec::size_type i = 0; i != a.size(); i++) {
		/* running a find operation to see if elements of a are NOT in b */
		if (std::find(b.begin(), b.end(), a[i]) == b.end()) {
			result.push_back(a[i]); count += 1;
		}
	}
	if (count == 0) {
		result = {};
	}
	return result;
}
 
// Given a matrix, a vector and an index, it sets the index-th column
// of the matrix to that vector, filling in the 'gaps' with INT_MAX.
arma::mat assign(arma::mat currMat, stdvec lst, int index) {
	arma::mat newMat = arma::mat(currMat);
	for (int elem = 0; elem < newMat.n_rows; elem++) {
		if (elem < lst.size()) {
			newMat.at(elem, index) = lst[elem];
		} else {
			newMat.at(elem, index) = INT_MAX;
		}
	}
	return newMat;
}

// residual vs. number of columns, running time //// running time vs num columns /// error vs cola

// Solves a linear system using armadillo.
arma::mat solve_linear(arma::mat L, arma::mat b) {
	arma::mat y = arma::solve(L, b);
	arma::mat lTrans = L.t();
	return arma::solve(lTrans, y);
}

// Solves a quadratic system.
arma::mat solve_quadratic(arma::mat rhs) {
	int n = rhs.n_rows; // number of rows of matrix
	arma::mat omega = arma::mat(n, n);
	omega.zeros(); // square matrix of zeroes

	if (n <= 1) {
		omega.at(0, 0) = sqrt(rhs.at(0, 0));
	}
	else {
		for (int i = 0; i < n; i++) {
			for (int j = i; j < n; j++) {
				if (i == j) {
					if (i == 0) {
						omega.at(j, i) = sqrt(rhs.at(j, i));
					}
					else {
						arma::mat submatrix = omega.submat(j, 0, j, i);
						double sum = accu(submatrix % submatrix);
						omega.at(j, i) = sqrt(rhs.at(j, i) - sum);
					}
				}
				else {
					double temp = 0;
					if (i != 0) {
						for (int k = 0; k < i; k++) {
							temp += omega.at(j, k) * omega.at(i, k);
						}
					}
					omega.at(j, i) = (rhs.at(j, i) - temp) / omega.at(i, i);
				}
			}
		}
	}
	return omega;
}
	
// Given a vector, performs a slice operation on it, returning the
// new, sliced vector as a result. (i.e vector[m:n])
stdvec slice(stdvec const &v, int m, int n) {
	stdvec vec(v.begin() + m, v.begin() + n);
	return vec;
}

// Given a vector and a vector of indices, returns a sliced version
// of the first vector, remaining indices corresponding to the 
// values present in the second vector.
stdvec specificSlice(stdvec lst, stdvec indices) {
	if (indices.size() == 0) {
		return lst;
	}
	stdvec toReturn;
	for (stdvec::size_type i = 0; i != indices.size(); i++) {
		toReturn.push_back(lst[indices[i]]);
	}
	return toReturn;
}

// Assigns the result of an unusual operation.
arma::vec assign(arma::vec ta, stdvec lst, arma::vec data) {
	arma::vec tr = arma::vec(ta);
	for (stdvec::size_type i = 0; i < lst.size(); i++) {
		tr[lst[i]] = data[i];
	}
	return tr;
}

// Performs argsort on a vector x.
stdvec argsort(stdvec x) {
	std::vector<std::pair<double, int> > y;
	for (int k = 0; k < x.size(); k++) {
		y.push_back(std::make_pair(x[k], k));
	}
	std::sort(y.begin(), y.end());
	stdvec toRet;
	for (int j = 0; j < x.size(); j++) {
		toRet.push_back(y[j].second);
	}
	return toRet;
}

// Replicates the range(x) function in Python.
stdvec range(int n) {
	stdvec x;
	for (int k = 0; k < n; k++) {
		x.push_back(k);
	}
	return x;
}

// Performs a matrix slice operation given the requirements of
// X[:, lst] as needed below.
arma::mat selectCols(arma::mat X, stdvec lst) {
	arma::mat toRet = arma::mat(X.n_rows, lst.size());
	for (stdvec::size_type i = 0; i != lst.size(); i++) {
		toRet.col(i) = X.col(lst[i]);
	}
	return toRet;
}

void printVec(stdvec vec) {
	for (int x = 0; x < vec.size(); x++) {
		std::cout << vec[x] << " ";
	}
	std::cout << "\n";
}

std::string vec2str(stdvec vec) {
	std::string to_ret;
	for (int x = 0; x < vec.size(); x++) {
		to_ret = to_ret + std::to_string(vec[x]) + " ";
	}
	return to_ret;
}

std::vector<stdvec> aRet(std::string index) {
	std::string demo = "a" + index + ".txt";
	std::ifstream myfile (demo);
	std::vector<stdvec> toret;
	
	if (myfile.is_open()) {
		std::string line;
		std::getline(myfile, line);
		int iterations = std::stoi(line); // aData
		for (int k = 0; k < iterations; k++) {
			stdvec input;
			while (std::getline(myfile, line))	{
				/* std::cout << line << "\n"; */
				if (line == "---") {
					break;
				}
				input.push_back(std::stod(line));
			}	
			toret.push_back(input);
		}
		myfile.close();
	}
	return toret;
}

stdvec bRet(std::string index) {
	std::string demo = "b" + index + ".txt";
	std::ifstream myfile (demo);
	stdvec toret;
	
	if (myfile.is_open()) {
		std::string line;
		std::getline(myfile, line);
		int iterations = std::stoi(line); // aData
		while (std::getline(myfile, line))	{
			 // std::cout << line << "\n"; 
			toret.push_back(std::stod(line));
		}	
		myfile.close();
	}
	return toret;
}

std::vector<int> mnRet(void) {
	std::ifstream myfile ("demomn.txt");
	std::vector<int> toret;
	if (myfile.is_open()) {
		std::string line;
		while (std::getline(myfile, line))	{
			toret.push_back(std::stoi(line));
		}	
		myfile.close();
	}
	return toret;
}

stdvec mat2vec(arma::mat matrix) {
	stdvec toret;
	arma::vec currRow;
	for (int i = 0; i < matrix.n_cols; i++) {
		currRow = matrix.col(i); 
		stdvec vecform = arma::conv_to<stdvec>::from(currRow);
		toret.insert(toret.end(), vecform.begin(), vecform.end());
	}
	return toret;
}

int size; int p; 

// Runs parallel BLARS. void parallel_bLARS_svm_data(stdvec LARS_Ahallow, double block, std::string ff)
stdvec parallel_bLARS_svm_data(double block, stdvec LARS_Ahallow) {
	if (block == 1) {
		MPI_Init(NULL, NULL);
		MPI_Comm_size(MPI_COMM_WORLD, &size);
		MPI_Comm_rank(MPI_COMM_WORLD, &p);
		std::cout << size <<"\n\n\n\n";
		std::cout << p;
	}
	std::string pStr = std::to_string(p);
	std::vector<stdvec> aDataP = aRet("d" + pStr);
	std::vector<stdvec> aRowP = aRet("r" + pStr);
	stdvec bP = bRet(pStr); std::vector<int> mn = mnRet();
	int m = mn[0]; int n = mn[1]; int min_temp = std::min(m, n); 
	int iterations = std::ceil(min_temp/block);
	int og_block = (int) block; double final_time = 0;
	int max_block = std::floor(std::min(m, n) / 2);


	int n_local = aDataP.size(); int ct; int counting = 0;
	arma::mat A_local = arma::mat(n_local, m);
	for (int row = 0; row < n_local; row++) {
		ct = 0;
		for (int col = 0; col < aRowP[row].size(); col++) {
			A_local.at(row, aRowP[row][ct]) = aDataP[row][ct];
			ct = ct + 1;
		}
	}

	auto time_start = steady_clock::now(); auto time_end = steady_clock::now();
	// Time start

	arma::vec b_sol_local = arma::vec(n_local); double c[m]; 
	for (int i = 0; i < m; i++) {
		c[i] = 0;
	}
	arma::mat tempMatr = A_local.t();
	arma::vec c_local = tempMatr * arma::vec(bP);
	double c_local_arr[m]; for (int i = 0; i < m; i++) {c_local_arr[i] = c_local[i];}
	MPI_Allreduce(c_local_arr, c, m, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	stdvec c_vec(c, c + sizeof(c)/(sizeof(c[0])));
	arma::vec c_a_vec = arma::vec(c_vec);

	arma::vec temp_vec = -1 * arma::abs(c_a_vec);
	stdvec B = slice(argsort(arma::conv_to<stdvec>::from(temp_vec)), 0, block);
	double c_const = (c_a_vec[B[block - 1]]);
	if (c_const < 0) {
		c_const *= -1;
	}
	stdvec A_hallow; A_hallow = B;
	stdvec A_hallow_old; A_hallow_old = A_hallow;
	int l_A_hallow = A_hallow.size();
	int l_A_hallow_old = A_hallow_old.size();
	stdvec A_hallow_c = diff(range(m), A_hallow);

	stdvec error_Ahallow;
	stdvec linfty_result;
	stdvec l2_result;

	if (p == 0) {
		time_end = steady_clock::now();
		double time_taken = duration_cast<milliseconds>(time_end - time_start).count();
		final_time += time_taken;

		linfty_result.push_back(c_const);
		double x = arma::norm(c_a_vec, 2);
		l2_result.push_back(x);

		if (LARS_Ahallow.size() != 0) {
			if (block == 1) {
				stdvec temp;
				for (int k = 0; k < block; k++) {
					temp.push_back(LARS_Ahallow[k]);
				}
				int difference = diff(temp, A_hallow).size();
				error_Ahallow.push_back(difference);
			}
			else {
				int diffSize = diff(slice(LARS_Ahallow, 0, block), A_hallow).size();
				error_Ahallow.push_back(diffSize);
			}
		}
		time_start = steady_clock::now();
	}

	arma::mat tempA = selectCols(A_local, A_hallow);
	arma::mat G_local = tempA.t() * tempA;
	stdvec G_local_vec = mat2vec(G_local);
	double G_local_arr[l_A_hallow * l_A_hallow]; 
	for (int i = 0; i < (l_A_hallow * l_A_hallow); i++) {
		G_local_arr[i] = G_local_vec[i];
	} 
	arma::mat G = arma::mat(l_A_hallow, l_A_hallow); 
	double G_vec[l_A_hallow * l_A_hallow];
	for (int i = 0; i < (l_A_hallow * l_A_hallow); i++) {
		G_vec[i] = 0;
	} 

	MPI_Reduce(G_local_arr, G_vec, (l_A_hallow * l_A_hallow), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	stdvec g_vec_summed(G_vec, G_vec + sizeof(G_vec)/(sizeof(G_vec[0])));

	if (p == 0) {
		for (int i = 0; i < l_A_hallow; i++) {
			G.col(i) = arma::vec(slice(g_vec_summed, i * l_A_hallow, (i+1) * l_A_hallow));
		}
	}


	arma::vec w = arma::vec(l_A_hallow); w.zeros();	double h = 0;
	arma::vec compact_w_h_vec = arma::vec(l_A_hallow + 1); compact_w_h_vec.zeros();
	double compact_w_h[l_A_hallow + 1]; arma::mat L;

	if (p == 0) {
		L = chol(G, "lower");
		stdvec s = specificSlice(arma::conv_to<stdvec>::from(c_a_vec), A_hallow);
		arma::mat q = solve_linear(L, s);
		h = arma::as_scalar(1 / sqrt(arma::vec(s).t() * q));
		w = arma::as_scalar(h) * q;
		for (int i = 0; i < l_A_hallow + 1; i++) {
			if (i == l_A_hallow) {
				compact_w_h[i] = h;
			} else {
				compact_w_h[i] = w[i];
			}
		}
	}
	
	MPI_Bcast(compact_w_h, l_A_hallow + 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	stdvec compact_w_h_stdvec(compact_w_h, compact_w_h + sizeof(compact_w_h)/(sizeof(compact_w_h[0]))); 
	compact_w_h_vec = arma::vec(compact_w_h_stdvec);
	w = arma::vec(slice(compact_w_h_stdvec, 0, l_A_hallow)); 
	h = compact_w_h_vec[l_A_hallow]; 

	arma::vec u_local = (selectCols(A_local, A_hallow) * w);
	double a_arr[m]; for (int i = 0; i < m; i++) { a_arr[i] = 0; }
	arma::vec a_local = A_local.t() * u_local;
	double a_local_arr[m]; for (int i = 0; i < m; i++) { a_local_arr[i] = a_local[i];}

	MPI_Allreduce(a_local_arr, a_arr, m, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); 
	stdvec a_stdvec(a_arr, a_arr + sizeof(a_arr)/(sizeof(a_arr[0]))); 

	double gamma = 0; // works until here!

	for (int counter = 0; counter < (iterations - 1); counter++) {
		
		double l_A_hallow_c = A_hallow_c.size();

		if (l_A_hallow_c < block) { 
			block = l_A_hallow_c; 
		}

		arma::vec compact_B_gamma = arma::vec(block + 1);
		compact_B_gamma.zeros(); stdvec B; int B_size = 0;
		int B_size_container[1]; B_size_container[0] = B_size;

		if (p == 0) {
			arma::vec min_pos = arma::vec(l_A_hallow_c); min_pos.zeros();
			/* change variable name here */
			for (int j = 0; j < l_A_hallow_c; j++) {
				double idx = A_hallow_c[j];
				double val1 = (c_const - c_a_vec[idx]) / (h * c_const - a_stdvec[idx]);
				double val2 = (c_const + c_a_vec[idx]) / (h * c_const + a_stdvec[idx]);
				if (val1 < 0) {
					val1 = std::numeric_limits<double>::infinity();
				} else if (val2 < 0) {
					val2 = std::numeric_limits<double>::infinity();
				}
				double min_val = std::min(val1, val2);
				
				if (min_val > 0) {
					min_pos[j] = min_val;
				} 
			}
			int min_choice = std::min(block, l_A_hallow_c);
			stdvec idx_vec = slice(argsort(arma::conv_to<stdvec>::from(min_pos)), 0, min_choice);
			B = specificSlice(A_hallow_c, idx_vec);
			gamma = 0.9 * min_pos[idx_vec[min_choice - 1]]; 
			B_size = B.size(); B_size_container[0] = B_size;
		}

		MPI_Bcast(B_size_container, 1, MPI_INT, 0, MPI_COMM_WORLD);
		double compact_B_gamma_arr[B_size_container[0] + 1];

		if (p == 0) {
			for (int i = 0; i < B_size_container[0] + 1; i++) {
				if (i == (B_size_container[0])) {
					compact_B_gamma_arr[i] = gamma;
				} else {
					compact_B_gamma_arr[i] = B[i];
				}
			}
		}

		MPI_Bcast(compact_B_gamma_arr, B_size_container[0] + 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		stdvec bg_stdvec(compact_B_gamma_arr, compact_B_gamma_arr + sizeof(compact_B_gamma_arr)/(sizeof(compact_B_gamma_arr[0]))); 
		gamma = bg_stdvec[B_size_container[0]]; B = slice(bg_stdvec, 0, B_size_container[0]); 

		b_sol_local = b_sol_local + gamma * u_local;

		arma::vec cAhD = arma::vec(specificSlice(arma::conv_to<stdvec>::from(c_a_vec), A_hallow)) * (1 - gamma * h);
		arma::vec cAhcD = arma::vec(specificSlice(arma::conv_to<stdvec>::from(c_a_vec), A_hallow_c)) - gamma * arma::vec(specificSlice(a_stdvec, A_hallow_c));
		c_const = c_const * (1 - gamma * h); c_a_vec = assign(c_a_vec, A_hallow, cAhD); 
		c_a_vec = assign(c_a_vec, A_hallow_c, cAhcD);
			
		A_hallow.insert(A_hallow.end(), B.begin(), B.end());
		l_A_hallow = A_hallow.size(); int l_B = B.size();
		A_hallow_c = diff(range(m), A_hallow);

		if (p == 0) {
			time_end = steady_clock::now();
			double time_taken = duration_cast<milliseconds>(time_end - time_start).count();
			final_time += time_taken;

			linfty_result.push_back(c_const);
			double x = arma::norm(c_a_vec, 2);
			l2_result.push_back(x);

			if (LARS_Ahallow.size() != 0) {
				int diffSizeCurr = diff(slice(LARS_Ahallow, 0, l_B * (counter + 2)), A_hallow).size();
				error_Ahallow.push_back(diffSizeCurr);
			}

			time_start = steady_clock::now();
		}

		arma::mat mul1 = selectCols(A_local, A_hallow_old);
		arma::mat mul2 = selectCols(A_local, B);
		arma::mat A_hallow_t_old_A_B_local = mul1.t() * mul2;

		arma::mat aHallowToldAB = arma::mat(l_A_hallow_old, l_B);
		stdvec ahtab_vec = mat2vec(A_hallow_t_old_A_B_local);
		double ahtab_local_arr[l_A_hallow_old * l_B]; 
		for (int i = 0; i < (l_A_hallow_old * l_B); i++) {
			ahtab_local_arr[i] = ahtab_vec[i];
		} 
		double ahtab_arr[l_A_hallow_old * l_B]; 
		for (int i = 0; i < (l_A_hallow_old * l_B); i++) {
			ahtab_arr[i] = 0;
		} 

		MPI_Reduce(ahtab_local_arr, ahtab_arr, (l_A_hallow_old * l_B), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		stdvec ahtab_vec_summed(ahtab_arr, ahtab_arr + sizeof(ahtab_arr)/(sizeof(ahtab_arr[0])));


		if (p == 0) {
			for (int i = 0; i < l_B; i++) {
				aHallowToldAB.col(i) = arma::vec(slice(ahtab_vec_summed, i * l_A_hallow_old, (i+1) * l_A_hallow_old));
			}
		}

		arma::mat A_B_t_A_B_local = mul2.t() * mul2;
		arma::mat A_B_t_A_B = arma::mat(l_B, l_B);
		stdvec abtab_vec = mat2vec(A_B_t_A_B_local);
		double abtab_local_arr[l_B * l_B]; 
		for (int i = 0; i < (l_B * l_B); i++) {
			abtab_local_arr[i] = abtab_vec[i];
		} 
		double abtab_arr[l_B * l_B]; 
		for (int i = 0; i < (l_B * l_B); i++) {
			abtab_arr[i] = 0;
		} 

		MPI_Reduce(abtab_local_arr, abtab_arr, (l_B * l_B), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		stdvec abtab_vec_summed(abtab_arr, abtab_arr + sizeof(abtab_arr)/(sizeof(abtab_arr[0])));


		if (p == 0) {
			for (int i = 0; i < l_B; i++) {
				A_B_t_A_B.col(i) = arma::vec(slice(abtab_vec_summed, i * l_B, (i+1) * l_B));
			}
		}

		double compact_w_h_temp[l_A_hallow + 1];

		if (p == 0) {
			arma::mat H = arma::mat(l_A_hallow_old, l_B); H.zeros();
			/* variable needs changing here */
			for (int j = 0; j < l_B; j++) {
				arma::vec to_ret = arma::solve(L, aHallowToldAB.col(j));
				H.col(j) = to_ret;
			}
			arma::mat rhs_Omega = A_B_t_A_B - H.t() * H;
			arma::mat Omega = solve_quadratic(rhs_Omega);
			/// ask about SOLVE QUADRATIC (nan for high block values?)

			arma::mat temp4 = arma::join_rows(H.t(), Omega);

			arma::mat tempX = arma::mat(l_A_hallow_old, l_B); tempX.zeros();
			arma::mat L2 = arma::join_rows(L, tempX);

			L = arma::join_cols(L2, temp4);

			stdvec s = specificSlice(arma::conv_to<stdvec>::from(c_a_vec), A_hallow);
			arma::mat q = solve_linear(L, s);
			h = arma::as_scalar(1 / sqrt(arma::vec(s).t() * q));
			w = arma::as_scalar(h) * q;

			for (int i = 0; i < l_A_hallow + 1; i++) {
				if (i == l_A_hallow) {
					compact_w_h_temp[i] = h;
				} else {
					compact_w_h_temp[i] = w[i];
				}
			}
		}

		MPI_Bcast(compact_w_h_temp, l_A_hallow + 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		stdvec compact_w_h_stdvec_2(compact_w_h_temp, compact_w_h_temp + sizeof(compact_w_h_temp)/(sizeof(compact_w_h_temp[0]))); 
		compact_w_h_vec = arma::vec(compact_w_h_stdvec_2);

		w = slice(arma::conv_to<stdvec>::from(compact_w_h_vec), 0, l_A_hallow);
		h = compact_w_h_vec[l_A_hallow];

		arma::vec u_local = (selectCols(A_local, A_hallow) * w);
		double a_arr[m]; for (int i = 0; i < m; i++) { a_arr[i] = 0; }
		arma::vec a_local = A_local.t() * u_local;
		double a_local_arr[m]; for (int i = 0; i < m; i++) { a_local_arr[i] = a_local[i];}

		MPI_Allreduce(a_local_arr, a_arr, m, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); 
		stdvec a_tempvec(a_arr, a_arr + sizeof(a_arr)/(sizeof(a_arr[0]))); 
		a_stdvec = a_tempvec;

		A_hallow_old = stdvec(A_hallow);
		l_A_hallow = A_hallow.size();
		l_A_hallow_old = A_hallow_old.size();

	}
	time_end = steady_clock::now(); 

	if (p == 0) {
		double time_taken = duration_cast<milliseconds>(time_end - time_start).count();
		final_time += time_taken; 

		// std::ofstream myoutput;
		// myoutput.open("cb" + std::to_string((int) og_block) + "t" + std::to_string(size), std::ios::app);
		// myoutput << final_time << "\n";
		// myoutput.close();

		// std::ofstream myoutput;
		// myoutput.open("cb" + std::to_string((int) og_block) + "t" + std::to_string(size), std::ios::app);
		// myoutput << final_time << "\n";
		// myoutput.close();

		// std::ofstream myoutput;
		// myoutput.open("cb" + std::to_string((int) og_block) + "t" + std::to_string(size), std::ios::app);
		// myoutput << final_time << "\n";
		// myoutput.close();

		// std::ofstream myoutput;
		// myoutput.open("cb" + std::to_string((int) og_block) + "t" + std::to_string(size), std::ios::app);
		// myoutput << final_time << "\n";
		// myoutput.close();

		std::cout << final_time << " ms\n";
		printVec(error_Ahallow);
		printVec(linfty_result);
		printVec(l2_result);
		printVec(A_hallow);
		std::cout << "--------\n";

		std::ofstream error;
		error.open("ct" + std::to_string(size) + "b" + std::to_string((int) og_block) + "error", std::ios::app);
		error << vec2str(error_Ahallow) << "\n";
		error.close();

		std::ofstream linfty;
		linfty.open("ct" + std::to_string(size) + "b" + std::to_string((int) og_block) + "linfty", std::ios::app);
		linfty << vec2str(linfty_result) << "\n";
		linfty.close();

		std::ofstream l2;
		l2.open("ct" + std::to_string(size) + "b" + std::to_string((int) og_block) + "l2", std::ios::app);
		l2 << vec2str(l2_result) << "\n";
		l2.close();

		std::ofstream time_output;
		time_output.open("ct" + std::to_string(size) + "b" + std::to_string((int) og_block) + "time", std::ios::app);
		time_output << final_time << "\n";
		time_output.close();



	}
	if (block == max_block) {
		MPI_Finalize();
	}
	return A_hallow;
}

int main(int argc, char** argv) {
	std::vector<int> mn = mnRet();
	int m = mn[0]; int n = mn[1];
	int max_block = std::floor(std::min(m, n) / 2);
	
	stdvec hallow; 
	for (int i = 1; i <= max_block; i++) {
		stdvec to_ret = parallel_bLARS_svm_data(i, hallow);
		if (i == 1) {
			hallow = to_ret;
		}
	}
	return 0;
}

