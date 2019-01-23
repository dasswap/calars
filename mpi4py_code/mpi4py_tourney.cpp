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
						arma::mat submatrix = rhs.submat(j, 0, j, i);
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

arma::mat selectRows(arma::mat X, stdvec lst) {
	arma::mat toRet = arma::mat(lst.size(), X.n_cols);
	for (stdvec::size_type i = 0; i != lst.size(); i++) {
		toRet.row(i) = X.row(lst[i]);
	}
	return toRet;	
}

void printVec(stdvec vec) {
	for (int x = 0; x < vec.size(); x++) {
		std::cout << vec[x] << " ";
	}
	std::cout << "\n";
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
	std::string demo = index + ".txt";
	std::ifstream myfile (demo);
	stdvec toret;
	
	if (myfile.is_open()) {
		std::string line;
		std::getline(myfile, line);
		int iterations = std::stoi(line); // aData
		while (std::getline(myfile, line))	{
			/* std::cout << line << "\n"; */
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

arma::mat src; arma::mat dst; int chosen_cols;
int n_g; int r_g; int c_g; arma::mat L; int p;
arma::mat A_chosen; arma::vec b; arma::vec b_sol;
arma::mat L_append_1; arma::mat L_append_2;
arma::mat A_B; bool test_var = false;

void print_shape(arma::mat matrix) {
	std::cout << "Rows: " << matrix.n_rows << " | Columns:" << matrix.n_cols << "\n";
}

void print_thread(std::string str, int p_check) {
	if (p == p_check) {
		std::cout << str << "\n";
	}
}

int sign(double num) {
	if (num > 0) {
		return 1;
	} else if (num < 0) {
		return -1;
	} else {
		return 0;
	}
}

double abs(double num) {
	double to_ret = num;
	if (sign(num) < 0) {
		to_ret = to_ret * -1;
	}
	return to_ret;
}

double stepLARS(double c_const, double h, arma::vec c, arma::vec a, double idx) {
	double val1; double val2; double min_val;
	double c_idx = c[idx]; double a_idx = a[idx];

	if (c_const >= abs(c_idx)) {
		if (sign(c_idx) == sign(a_idx)) {
			val1 = (c_const - c_idx) / (h * c_const - a_idx);
			val2 = (c_const + c_idx) / (h * c_const + a_idx);
			if (val1 < 0) {
				val1 = std::numeric_limits<double>::infinity();
			} else if (val2 < 0) {
				val2 = std::numeric_limits<double>::infinity();
			}
			min_val = std::min(val1, val2);
		} else {
			min_val = (c_const - c_idx) / (h * c_const + a_idx);
		}
	} else {
		if (sign(c_idx) == sign(a_idx)) {
			if ((c_idx * h) >= (c_const * a_idx)) {
				min_val = (c_const + c_idx) / (h * c_const + a_idx);
			} else {
				min_val = 1 / h;
			}
		} else {
			min_val = 0;
		}
	}
	return min_val;
}




// Runs parallel BLARS. void parallel_bLARS_svm_data(stdvec LARS_Ahallow, double block, std::string ff)
stdmat bLARS_tournament(arma::mat A, arma::mat A_c, arma::vec b, double block, arma::vec b_sol, arma::mat L) {
	double l_A_c = A_c.n_cols; int iterations = 0;
	if (l_A_c < 1) {
		stdmat to_ret; stdvec empty; arma::vec x = arma::vec(empty);
		arma::mat last_elem = arma::mat(b.n_elem, block); last_elem.zeros();
		to_ret.push_back(b_sol); to_ret.push_back(arma::vec(x));
		to_ret.push_back(arma::vec(x)); to_ret.push_back(arma::vec(x));
		to_ret.push_back(last_elem); return to_ret;
	}
	arma::vec r = b - b_sol; int l_A; arma::mat A_full;
	if (chosen_cols == 0) {
		l_A = 0;
		A_full = A_c;
	} else {
		l_A = A.n_cols;
		A_full = arma::join_rows(A, A_c);
	}
    arma::vec c = A_full.t() * r; int B; stdvec A_hallow;
    if (chosen_cols == 0) {
        arma::vec temp_vec = -1 * arma::abs(c);
		B = argsort(arma::conv_to<stdvec>::from(temp_vec))[0];        
        A_hallow.push_back(B); iterations = std::min(block - 1, l_A_c);
    } else {
        A_hallow = range(l_A); iterations = std::min(block, l_A_c);
    }
    stdvec A_hallow_c = diff(range(l_A + l_A_c), A_hallow);
    stdvec A_hallow_old = A_hallow;
    
    double l_A_hallow = A_hallow.size();
   	double l_A_hallow_c = A_hallow_c.size();
    double l_A_hallow_old = A_hallow_old.size();   
    double c_const = (arma::abs(c)).max();
 	
    if (l_A_hallow == 1) {
        arma::mat tempG = selectCols(A_full, A_hallow);
        arma::mat G = (tempG.t()) * tempG; L = chol(G);
    }

    arma::vec w = arma::vec(l_A); w.zeros();
    stdvec s; arma::mat q; double h; double gamma = 0;
    arma::vec u; arma::vec a; 


    for (int counter = 0; counter < iterations; counter++) {
    	s = specificSlice(arma::conv_to<stdvec>::from(c), A_hallow);
    	arma::mat q = solve_linear(L, s); h = arma::as_scalar(1 / sqrt(arma::vec(s).t() * q));
        w = arma::as_scalar(h) * q; u = selectCols(A_full, A_hallow) * w; a = A_full.t() * u;


        arma::vec min_pos = arma::vec(l_A_hallow_c); min_pos.zeros(); double idx;
		for (int j = 0; j < l_A_hallow_c; j++) {
			idx = A_hallow_c[j];
			double val1 = (c_const - c[idx]) / (h * c_const - a[idx]);
			double val2 = (c_const + c[idx]) / (h * c_const + a[idx]);
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
        idx = (argsort(arma::conv_to<stdvec>::from(min_pos)))[0];
        B = A_hallow_c[idx]; gamma = 0.9 * min_pos[idx]; 
        b_sol = b_sol + u * gamma; c = c - gamma * a; c_const = (arma::abs(c)).max();
        
        A_hallow.push_back(B); l_A_hallow = A_hallow.size();
        A_hallow_c = diff(range(l_A + l_A_c), A_hallow); l_A_hallow_c = A_hallow_c.size();	

        stdvec B_vec; B_vec.push_back(B);
        arma::mat A_full_hallow = selectCols(A_full, A_hallow_old); arma::mat A_B = selectCols(A_full, B_vec);
        arma::mat A_full_hallow_t_old_A_B = A_full_hallow.t() * A_B; arma::mat A_full_B_t_A_full_B = A_B.t() * A_B;  

        arma::mat H = arma::solve(L, A_full_hallow_t_old_A_B);
        arma::mat rhs_Omega = A_full_B_t_A_full_B - H.t() * H;
        arma::mat Omega = solve_quadratic(rhs_Omega);
        arma::mat temp4 = arma::join_rows(H.t(), Omega);

        arma::mat tempX = arma::mat(l_A_hallow_old, 1); tempX.zeros();
        arma::mat L2 = arma::join_rows(L, tempX);
        L = arma::join_cols(L2, temp4); A_hallow_old = stdvec(A_hallow);
        l_A_hallow_old = (A_hallow_old.size());
    }  
    if (l_A == 0) {
    	L_append_1 = arma::mat(1, 1); L_append_1(0, 0) = 0;
    } else {
    	L_append_1 = L.submat(l_A, 0, l_A + block - 1, l_A - 1); 
    } 
    
    int index = l_A + block - 1; L_append_2 = L.submat(l_A, l_A, index, index);
    stdvec which_col_stdvec = slice(A_hallow, l_A, l_A + block);
    arma::mat sliced = selectCols(A_full, which_col_stdvec);
    for (int i = 0; i < which_col_stdvec.size(); i++) {
    	which_col_stdvec[i] = which_col_stdvec[i] - l_A;
    }
    arma::vec which_col = arma::vec(which_col_stdvec);

    stdmat to_ret; 
    to_ret.push_back(b_sol); to_ret.push_back(L_append_1); to_ret.push_back(L_append_2);
    to_ret.push_back(which_col); to_ret.push_back(sliced); return to_ret;
}


	// arma::mat x_main = x_mat.submat(0, 0, n_g - 1, hmn1 - 1);
	// arma::mat x_piece_1 = x_mat.submat(n_g, 0, n_g + 1 - 1, hmn1 - 1);
	// arma::mat x_piece_2 = x_mat.submat(n_g + 1, 0, n_g + 2 - 1, hmn1 - 1);
	// arma::mat x_piece_3 = x_mat.submat(n_g + 2, 0, n_g + 3 - 1, hmn1 - 1);

	// arma::mat y_main = y_mat.submat(0, 0, n_g - 1, hmn2 - 1);
	// arma::mat y_piece_1 = y_mat.submat(n_g, 0, n_g + 1 - 1, hmn2 - 1);
	// arma::mat y_piece_2 = y_mat.submat(n_g + 1, 0, n_g + 2 - 1, hmn2 - 1);
	// arma::mat y_piece_3 = y_mat.submat(n_g + 2, 0, n_g + 3 - 1, hmn2 - 1);

	// arma::mat A_2B = arma::join_rows(y_main, x_main);
	// arma::mat proc_v = arma::join_cols(y_piece_1, x_piece_1);
	// arma::mat indices = arma::join_cols(y_piece_2, x_piece_2);
	// arma::mat real_indices = arma::join_cols(y_piece_3, x_piece_3);

// 0 cols, 1 rows
// rows is hstack
void custom_reduce(void *in, void *inout, int *len, MPI_Datatype *dptr) {
	int i; stdvec x; stdvec y;
	double* ind = (double *) in;
	double* inoutd = (double*) inout;
	double* inout_ret = inoutd;
	for (i = 0; i < *len; i++) {
		x.push_back((*ind));
		y.push_back((*inoutd)); 
		ind++; inoutd++;
	}
	arma::vec x_vec = arma::vec(x);
	arma::vec y_vec = arma::vec(y);

	arma::mat x_mat = arma::reshape(x_vec, r_g, c_g);
	arma::mat y_mat = arma::reshape(y_vec, r_g, c_g);
	double hm1 = x_mat(n_g + 3, 0); 
	double hm2 = y_mat(n_g + 3, 0); 
	double block = c_g - 3;
	double hm3 = std::min(hm1 + hm2, block);

	if (hm3 < 0) {
		return;
	}

	double hmn1 = std::min(block, hm1);
	double hmn2 = std::min(block, hm2);

	// 

	arma::mat x_main; arma::mat x_piece_1; arma::mat x_piece_2; arma::mat x_piece_3;
	arma::mat y_main; arma::mat y_piece_1; arma::mat y_piece_2; arma::mat y_piece_3;
	arma::mat A_2B; arma::mat proc_v; arma::mat indices; arma::mat real_indices;

	if (hmn1 > 0) {
		x_main = x_mat.submat(0, 0, n_g - 1, hmn1 - 1);
		x_piece_1 = x_mat.submat(n_g, 0, n_g + 1 - 1, hmn1 - 1).t();
		x_piece_2 = x_mat.submat(n_g + 1, 0, n_g + 2 - 1, hmn1 - 1).t();
		x_piece_3 = x_mat.submat(n_g + 2, 0, n_g + 3 - 1, hmn1 - 1).t();
	}

	if (hmn2 > 0) {
		y_main = y_mat.submat(0, 0, n_g - 1, hmn2 - 1);
		y_piece_1 = y_mat.submat(n_g, 0, n_g + 1 - 1, hmn2 - 1).t();
		y_piece_2 = y_mat.submat(n_g + 1, 0, n_g + 2 - 1, hmn2 - 1).t();
		y_piece_3 = y_mat.submat(n_g + 2, 0, n_g + 3 - 1, hmn2 - 1).t();
	}

	if (hmn1 <= 0 && hmn2 <= 0) {
		arma::vec ret_vec = arma::reshape(src, src.n_elem, 1); 
	    for (i = 0; i < *len; i++) {
			*inout_ret = ret_vec[i];
			inout_ret++;
		}
		return;
	}

	if (hmn1 > 0  && hmn2 <= 0) {
		A_2B = x_main; proc_v = x_piece_1; indices = x_piece_2; real_indices = x_piece_3;
	} else if (hmn1 <= 0 && hmn2 > 0) {
		A_2B = y_main; proc_v = y_piece_1; indices = y_piece_2; real_indices = y_piece_3;
	} else {
		A_2B = arma::join_rows(y_main, x_main);
		proc_v = arma::join_cols(y_piece_1, x_piece_1);
		indices = arma::join_cols(y_piece_2, x_piece_2);
		real_indices = arma::join_cols(y_piece_3, x_piece_3);
	}

	stdmat d = bLARS_tournament(A_chosen, A_2B, b, hm3, b_sol, L);
    arma::vec b_sol__ = d[0]; arma::mat L_append_1 = d[1]; arma::mat L_append_2 = d[2]; 
    arma::vec B_ = d[3]; arma::mat A_B_ = d[4]; 
 
    src(n_g + 3, 0) = A_B_.n_cols; 
    proc_v.reshape(1, proc_v.n_elem); arma::mat proc_v_B = selectCols(proc_v, arma::conv_to<stdvec>::from(B_));
    indices.reshape(1, indices.n_elem); arma::mat indices_B = selectCols(indices, arma::conv_to<stdvec>::from(B_));
    real_indices.reshape(1, real_indices.n_elem); arma::mat real_indices_B = selectCols(real_indices, arma::conv_to<stdvec>::from(B_));
    arma::mat temp100 = arma::join_cols(proc_v_B, arma::join_cols(indices_B, real_indices_B));

    src.submat(0, 0, n_g + 3 - 1, A_B_.n_cols - 1) = arma::join_cols(A_B_, temp100);
    src.submat(0, block, n_g - 1, block + 1 - 1) = b_sol__;

    int l_L_append_1 = L_append_1.n_elem; int l_L_append_2 = L_append_2.n_elem;

    src.submat(0, block + 1, l_L_append_1 - 1, block + 2 - 1) = arma::reshape(L_append_1.t(), l_L_append_1, 1);
    src.submat(0, block + 2, l_L_append_2 - 1, block + 3 - 1) = arma::reshape(L_append_2.t(), l_L_append_2, 1);

    arma::vec ret_vec = arma::reshape(src, src.n_elem, 1); 
    for (i = 0; i < *len; i++) {
		*inout_ret = ret_vec[i];
		inout_ret++;
	}

}

//         src[0:n+3,0:A_B_.shape[1]] = np.concatenate((A_B_,[proc_v[B_],indices[B_],real_indices[B_]]),axis=0)
//         src[0:n,block:block+1] = b_sol__

//         l_L_append_1 = L_append_1.shape[0]*L_append_1.shape[1]
//         src[0:l_L_append_1,block+1:block+2] = L_append_1.reshape(l_L_append_1,1)

//         l_L_append_2 = L_append_2.shape[0]*L_append_2.shape[1]
//         src[0:l_L_append_2,block+2:block+3] = L_append_2.reshape(l_L_append_2,1)
        
int size;

std::string vec2str(stdvec vec) {
	std::string to_ret;
	for (int x = 0; x < vec.size(); x++) {
		to_ret = to_ret + std::to_string(vec[x]) + " ";
	}
	return to_ret;
}

stdvec mpi4py_tournament_bLARS_svm_clever(double block, stdvec LARS_Ahallow) {
    
    if (block == 1) {
	    MPI_Init(NULL, NULL);
		MPI_Comm_size(MPI_COMM_WORLD, &size);
		MPI_Comm_rank(MPI_COMM_WORLD, &p);
	}
	std::string pStr = std::to_string(p);
	std::vector<stdvec> aDataP = aRet("dT" + pStr);
	std::vector<stdvec> aRowP = aRet("rT" + pStr);
	stdvec aRowI = bRet("arI" + pStr);
	stdvec b_vec_temp = bRet("bT0"); std::vector<int> mn = mnRet();
	b = arma::vec(b_vec_temp);
	int m = mn[0]; int n = mn[1]; int min_temp = std::min(m, n); 
	int iterations = std::ceil(min_temp/block);
	int og_block = (int) block; n_g = n;
	int max_block = std::floor(std::min(m, n) / 2);


	int m_local = aDataP.size(); int ct; int counting = 0;
	arma::mat A_local_T = arma::mat(m_local, n);
	for (int row = 0; row < m_local; row++) {
		ct = 0;
		for (int col = 0; col < aRowP[row].size(); col++) {
			A_local_T(row, aRowP[row][col]) = aDataP[row][ct];
			ct = ct + 1;
		}
	}
	arma::mat A_local = A_local_T.t(); double final_time = 0;
	auto time_start = steady_clock::now(); auto time_end = steady_clock::now();

	stdvec error_Ahallow;
	stdvec linfty_result;
	stdvec l2_result;

    A_chosen = arma::mat(n, 1); chosen_cols = 0;
    b_sol = arma::vec(n); b_sol.zeros();
    stdvec A_hallow_winners; stdvec A_hallow; stdvec A_hallow_local;

    L = arma::mat(1, 1); L.zeros();
    stdvec A_hallow_local_c = range(m_local);
    
    int dim1 = n + 4; int dim2 = block * (chosen_cols + block);
    int dim = std::max(dim1, dim2); int how_many;

    for (int counter = 0; counter < iterations; counter++) {
    	double l_A_hallow_l_c = A_hallow_local_c.size();
        how_many = std::min(block, l_A_hallow_l_c); 
        stdmat d = bLARS_tournament(A_chosen, selectCols(A_local, A_hallow_local_c), b, how_many, b_sol, L);

        // how many = block

        arma::vec b_sol_ = d[0]; L_append_1 = d[1]; L_append_2 = d[2]; 
        arma::vec B = d[3]; A_B = d[4]; 
        
		MPI_Op op; MPI_Op_create(custom_reduce, 0, &op);
        dim1 = n + 4; dim2 = block * (chosen_cols + block);
        dim = std::max(dim1, dim2); r_g = dim; c_g = block + 3; 
        dst = arma::mat(dim, block + 3); dst.zeros();    
        src = arma::mat(dim, block + 3); src.zeros();

        arma::vec temp;
        if (B.n_elem > 0) {
        	temp = arma::vec(B.n_elem); 
        	for (int i = 0; i < B.n_elem; i++) {
        		temp[i] = aRowI[A_hallow_local_c[B[i]]];
        	}
            src.at(n+3, 0) = A_B.n_cols;
        } else {
        	temp = arma::vec(A_B.n_cols); temp.zeros();
            B = arma::vec(A_B.n_cols); temp.zeros();
            src.at(n+3, 0) = 0;
        }
        arma::vec X_0 = arma::vec(A_B.n_cols); 
        for (int i = 0; i < A_B.n_cols; i++) {
        	X_0[i] = p;
        }
        arma::mat X_0_T = X_0.t();
        arma::mat B_T = B.t();   
        arma::mat temp_T = temp.t();  
        arma::mat X_1 = arma::join_cols(A_B, arma::mat(X_0_T));
        arma::mat X_2 = arma::join_cols(B_T, temp_T);
        arma::mat X_3 = arma::join_cols(X_1, X_2);
       
        src.submat(0, block, n - 1, block + 1 - 1) = arma::mat(b_sol_);
        if (A_B.n_cols != 0) {
        	src.submat(0, 0, n + 3 - 1, int(A_B.n_cols) - 1) = X_3; 
        } // src(n + 3, 0) = block

        stdvec src_vector = mat2vec(src); double src_arr[src_vector.size()]; double dst_arr[src_vector.size()];
        for (int i = 0; i < src_vector.size(); i++) {src_arr[i] = src_vector[i];}

        MPI_Allreduce(src_arr, dst_arr, src_vector.size(), MPI_DOUBLE, op, MPI_COMM_WORLD); int ct_temp = 0;

        for (int i = 0; i < (block + 3); i++) {
        	for (int j = 0; j < dim; j++) {
        		dst(j, i) = dst_arr[ct_temp]; ct_temp += 1;
        	}
        }

        MPI_Op_free(&op);
        if (size > 1) {
        	how_many = (dst(n + 3, 0));
        } else {
        	how_many = A_B.n_cols;
        }

        arma::mat L_temp;
        if (chosen_cols > 0) {
        	arma::mat zero_temp = arma::mat(chosen_cols, how_many); zero_temp.zeros();
        	L_temp = arma::join_rows(L, zero_temp); L = L_temp;
        	if (size > 1) {
        		arma::mat temp1 = dst.submat(0, how_many + 1, how_many * chosen_cols - 1, how_many + 2 - 1);
        		L_append_1 = arma::reshape(temp1, chosen_cols, how_many).t();
        		arma::mat temp2 = dst.submat(0, how_many + 2, how_many * how_many - 1, how_many + 3 - 1);
        		L_append_2 = arma::reshape(temp2, how_many, how_many).t();

        	}
        	L_temp = arma::join_cols(L, arma::join_rows(L_append_1, L_append_2)); L = L_temp;
        } else {
        	if (size > 1) {
        		arma::mat temp3 = dst.submat(0, how_many + 2, how_many * how_many - 1, how_many + 3 - 1);
        		L = arma::reshape(temp3, how_many, how_many).t();
        	} else {
        		L = L_append_2;
        	}
        }

        arma::mat to_choose = dst.submat(0, 0, n - 1, how_many - 1);
        if (chosen_cols == 0) {
        	A_chosen = to_choose;
        } else {
        	arma::mat temp74 = arma::join_rows(A_chosen, to_choose);
        	A_chosen = temp74;
        }
        chosen_cols = A_chosen.n_cols;

        b_sol = dst.submat(0, block, n - 1, block + 1 - 1);

        int proc; int idx; int idx2;
        for (int ct = 0; ct < how_many; ct ++) {
            proc = dst.at(n, ct);
            idx = dst.at(n + 1, ct);
            idx2 = dst.at(n + 2, ct);
            A_hallow_winners.push_back(idx2);        
            if (p == proc) {
                A_hallow_local.push_back(A_hallow_local_c[idx]);
                A_hallow.push_back(aRowI[idx]);
            }
        }
        A_hallow_local_c = diff(range(m_local), A_hallow_local);

        if (p == 0) {
        	time_end = steady_clock::now();
			double time_taken = duration_cast<milliseconds>(time_end - time_start).count();
			final_time += time_taken;
		}

        double cc[m]; for (int i = 0; i < m; i++) {cc[i] = 0;}
        arma::vec a_cc_local = A_local.t() * (b - b_sol);
        double cc_local[a_cc_local.n_elem]; for (int i = 0; i < m; i++) {cc_local[i] = a_cc_local[i];}
        int recvcounts[size]; int length = a_cc_local.n_elem;
        MPI_Gather(&length, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);


        int displs[size];
        if (p == 0) {
	        for (int i = 0; i < size; i++) {
	        	if (i == 0) {
	        		displs[i] = 0;
	        	} else {
	        		displs[i] = displs[i - 1] + recvcounts[i - 1]; 
	        	}
	        } 
	    }

        MPI_Gatherv(cc_local, (int) a_cc_local.n_elem, MPI_DOUBLE, cc, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        stdvec cc_vec(cc, cc + sizeof(cc)/(sizeof(cc[0]))); arma::vec cc_a_vec = arma::vec(cc_vec);


        if (p == 0) {
            linfty_result.push_back(arma::max(arma::abs(cc_a_vec)));
            double x = arma::norm(cc_a_vec, 2);
			l2_result.push_back(x);

            if (LARS_Ahallow.size() != 0) {
				if (block == 1) {
					stdvec temp;
					for (int k = 0; k < block; k++) {
						temp.push_back(LARS_Ahallow[k]);
					}
					int difference = diff(temp, A_hallow_winners).size();
					error_Ahallow.push_back(difference);
				}
				else {
					int diffSize = diff(slice(LARS_Ahallow, 0, block), A_hallow_winners).size();
					error_Ahallow.push_back(diffSize);
				}
			}
			time_start = steady_clock::now();
        }
    }
    if (p == 0) {
	    std::cout << final_time << " ms.\n";
	   	printVec(error_Ahallow);
		printVec(linfty_result);
		printVec(l2_result);
	    printVec(A_hallow_winners);
	    std::cout << "--------\n";

	    std::ofstream error;
		error.open("ctourt" + std::to_string(size) + "b" + std::to_string((int) og_block) + "error", std::ios::app);
		error << vec2str(error_Ahallow) << "\n";
		error.close();

		std::ofstream linfty;
		linfty.open("ctourt" + std::to_string(size) + "b" + std::to_string((int) og_block) + "linfty", std::ios::app);
		linfty << vec2str(linfty_result) << "\n";
		linfty.close();

		std::ofstream l2;
		l2.open("ctourt" + std::to_string(size) + "b" + std::to_string((int) og_block) + "l2", std::ios::app);
		l2 << vec2str(l2_result) << "\n";
		l2.close();

		std::ofstream time_output;
		time_output.open("ctourt" + std::to_string(size) + "b" + std::to_string((int) og_block) + "time", std::ios::app);
		time_output << final_time << "\n";
		time_output.close();

	}
	if (block == max_block) {
	    MPI_Finalize(); 
	}
    return A_hallow_winners;
}

int main(int argc, char** argv) {
	std::vector<int> mn = mnRet();
	int m = mn[0]; int n = mn[1];
	int max_block = std::floor(std::min(m, n) / 2);

	stdvec hallow; 
	for (int i = 1; i <= max_block; i++) {
		stdvec to_ret = mpi4py_tournament_bLARS_svm_clever(i, hallow);
		if (i == 1) {
			hallow = to_ret;
		}
	}
	return 0;
}

