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
#include <sstream>
#include <fstream>
#include <ctime>

// http://arma.sourceforge.net
// Used to solve linear systems.
#include "/usr/local/include/armadillo"
#include "/usr/lib/openmpi/include/mpi.h"

typedef std::vector<double> stdvec;
typedef std::vector<arma::mat> stdmat;

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
arma::mat A_B; bool test_var = false; int size;

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
stdmat bLARS_tournament(arma::mat A, arma::mat A_c, arma::vec b, double block, arma::vec b_sol, arma::mat L, bool L_send) {
	double l_A_c = A_c.n_cols; int iterations = 0;
	if (l_A_c < 1) {
		stdmat to_ret; stdvec empty; arma::vec x = arma::vec(empty);
		arma::mat last_elem = arma::mat(b.n_elem, block); last_elem.zeros();
		to_ret.push_back(b_sol); to_ret.push_back(arma::vec(x));
		to_ret.push_back(last_elem); to_ret.push_back(arma::vec(x));
		to_ret.push_back(arma::vec(x)); return to_ret;
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
    to_ret.push_back(b_sol); to_ret.push_back(which_col); to_ret.push_back(sliced); 

    if (L_send) {
    	to_ret.push_back(L_append_1); to_ret.push_back(L_append_2);
    }

    return to_ret;
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

stdvec buffer; 

void custom_reduce(void *in, void *inout, int *len, MPI_Datatype *dptr) {
	int i; stdvec x; stdvec y;
	double* ind = (double *) in;
	double* inoutd = (double*) inout;
	double* inout_ret = inoutd;
	double* inout_reset = inoutd;

	// for (i = 0; i < *len; i++) {
	// 	x.push_back((*ind));
	// 	y.push_back((*inoutd)); 
	// 	ind++; inoutd++;
	// }
	// arma::vec x_vec = arma::vec(x);
	// arma::vec y_vec = arma::vec(y);

	// arma::mat x_mat = arma::reshape(x_vec, r_g, c_g);
	// arma::mat y_mat = arma::reshape(y_vec, r_g, c_g);

	arma::mat x_mat = arma::mat(r_g, c_g); x_mat.zeros();
	arma::mat y_mat = arma::mat(r_g, c_g); y_mat.zeros(); int haha = 0;

	for (i = 0; i < *len; i+=3) {
		double x_num = *ind; double y_num = *inoutd; *inoutd = 0; ind++; inoutd++;
		double x_row = *ind; double y_row = *inoutd; *inoutd = 0; ind++; inoutd++;
		double x_col = *ind; double y_col = *inoutd; *inoutd = 0; ind++; inoutd++;
		if (i > 0) { // The list can go onwards
			if ((x_row == 0) && (x_col == 0)) {
				x_num = x_mat(0, 0);
			} else if ((y_row == 0) && (y_col == 0)) {
				y_num = y_mat(0, 0);
			}
		} 
		x_mat(x_row, x_col) = x_num; y_mat(y_row, y_col) = y_num;
	}

	// if (p == 3) {
	// 	x_mat.submat(0, 0, 9, c_g - 1).print(); std::cout << "\n---\n";
	// 	x_mat.submat(r_g - 10, 0, r_g - 1, c_g - 1).print(); std::cout << "\n---\n";

	// 	y_mat.submat(0, 0, 9, c_g - 1).print(); std::cout << "\n---\n";
	// 	y_mat.submat(r_g - 10, 0, r_g - 1, c_g - 1).print(); std::cout << "\n---\n";
	// }
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
		arma::sp_mat src_sparse = arma::sp_mat(src); 
		arma::sp_mat::iterator it = src_sparse.begin(); arma::sp_mat::iterator it_end = src_sparse.end();
        for(; it != it_end; ++it) {
        	*inout_ret = *it; inout_ret++;
        	*inout_ret = it.row(); inout_ret++;
        	*inout_ret = it.col(); inout_ret++;
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

	bool condition = (p == size - 1); 
	// bool condition = true;
	stdmat d = bLARS_tournament(A_chosen, A_2B, b, hm3, b_sol, L, condition);
    arma::vec b_sol__ = d[0]; arma::vec B_ = d[1]; arma::mat A_B_ = d[2]; 

    if (condition) {
    	arma::mat L_append_1 = d[3]; arma::mat L_append_2 = d[4]; 
    }

    proc_v.reshape(1, proc_v.n_elem); arma::mat proc_v_B = selectCols(proc_v, arma::conv_to<stdvec>::from(B_));
    indices.reshape(1, indices.n_elem); arma::mat indices_B = selectCols(indices, arma::conv_to<stdvec>::from(B_));
    real_indices.reshape(1, real_indices.n_elem); arma::mat real_indices_B = selectCols(real_indices, arma::conv_to<stdvec>::from(B_));
    arma::mat temp100 = arma::join_cols(proc_v_B, arma::join_cols(indices_B, real_indices_B)); int counter_curr = 0;

 //    for (int index_c = 0; index_c < A_B_.n_cols; index_c++) {
 //    	for (int index_r = 0; index_r < (n_g + 3); index_r++) {
 //    		inout_ret = inout_ret + (src.n_rows * index_c + index_r);
 //    		*inout_ret = src(index_r, index_c); 
 //    		inout_ret = inout_reset;
 //    	}
 //    }

 //    src(n_g + 3, 0) = A_B_.n_cols; 
 //    inout_ret += (n_g + 3); *inout_ret = A_B_.n_cols;
 //    inout_ret = inout_reset;

 //    src.submat(0, block, n_g - 1, block + 1 - 1) = b_sol__;
 //    for (int index_r = 0; index_r < n_g; index_r++) {
 //    	int data = src.n_rows * block + index_r; inout_ret = inout_ret + data;
	// 	*inout_ret = src(index_r, block); inout_ret = inout_reset;
	// }

	// if (condition) {
	//     int l_L_append_1 = L_append_1.n_elem; int l_L_append_2 = L_append_2.n_elem;
	//     src.submat(0, block + 1, l_L_append_1 - 1, block + 2 - 1) = arma::reshape(L_append_1.t(), l_L_append_1, 1);
	//     for (int index_r = 0; index_r < l_L_append_1; index_r++) {
	//     	int data = src.n_rows * (block + 1) + index_r; inout_ret = inout_ret + data;
	// 		*inout_ret = src(index_r, block + 1); inout_ret = inout_reset;
	// 	}
	//     src.submat(0, block + 2, l_L_append_2 - 1, block + 3 - 1) = arma::reshape(L_append_2.t(), l_L_append_2, 1);
	//     for (int index_r = 0; index_r < l_L_append_2; index_r++) {
	//     	int data = src.n_rows * (block + 2) + index_r; inout_ret = inout_ret + data;
	// 		*inout_ret = src(index_r, block + 2); inout_ret = inout_reset;
	// 	}
	// }

    src.submat(0, 0, n_g + 3 - 1, A_B_.n_cols - 1) = arma::join_cols(A_B_, temp100);
    // src.submat(0, 0, 9, 2).print(); std::cout << "lmaoooo\n";
    for (int index_c = 0; index_c < A_B_.n_cols; index_c++) {
    	for (int index_r = 0; index_r < (n_g + 3); index_r++) {
    		double element_curr = src(index_r, index_c);
    		if (element_curr != 0 && counter_curr < *len) { 
	    		*inout_ret = element_curr; inout_ret++;
	    		*inout_ret = index_r; inout_ret++; *inout_ret = index_c; inout_ret++;
	    		counter_curr += 3;
	    	} else if (element_curr != 0 && counter_curr >= *len) {
	    		// std::cout << "biggest box, " << counter_curr << ": " << element_curr << "\n"; 
	    		buffer.push_back(element_curr); buffer.push_back(index_r); buffer.push_back(index_c);
	    		counter_curr += 3;
	    	}
    	}
    }	

    src(n_g + 3, 0) = A_B_.n_cols; 
    *inout_ret = A_B_.n_cols; inout_ret++;
    *inout_ret = n_g + 3; inout_ret++;
    *inout_ret = 0; inout_ret++;
    counter_curr += 3;

 //    for (i = 0; i < *len; i+=3) {
	// 	std::cout << *inout_reset; inout_reset++;
	// 	std::cout << " | row: " << *inout_reset; inout_reset++;
	// 	std::cout << ", col: " << *inout_reset << "\n"; inout_reset++;
	// }

    src.submat(0, block, n_g - 1, block + 1 - 1) = b_sol__;
    for (int index_r = 0; index_r < n_g; index_r++) {
    	double element_curr = src(index_r, block);
    	if (element_curr != 0 && counter_curr < *len) {
			*inout_ret = element_curr; inout_ret++;
			*inout_ret = index_r; inout_ret++; *inout_ret = block; inout_ret++;
			counter_curr += 3;
		} else if (element_curr != 0 && counter_curr >= *len) {
	    	// std::cout << "b_sol, " << counter_curr << ": " << element_curr << "\n";
	    	buffer.push_back(element_curr); buffer.push_back(index_r); buffer.push_back(block);
	    	counter_curr += 3;
    	}
	}

	if (condition) {
	    int l_L_append_1 = L_append_1.n_elem; int l_L_append_2 = L_append_2.n_elem;

	    src.submat(0, block + 1, l_L_append_1 - 1, block + 2 - 1) = arma::reshape(L_append_1.t(), l_L_append_1, 1);
	    for (int index_r = 0; index_r < l_L_append_1; index_r++) {
	    	double element_curr = src(index_r, block + 1);
	    	if (element_curr != 0 && counter_curr < *len) {
	    		*inout_ret = element_curr; inout_ret++;
	    		*inout_ret = index_r; inout_ret++; *inout_ret = block + 1; inout_ret++;
	    		counter_curr += 3;
    		} else if (element_curr != 0 && counter_curr >= *len) {
	    		// std::cout << "l_append_1, " << counter_curr << ": " << element_curr << "\n";
	    		buffer.push_back(element_curr); buffer.push_back(index_r); buffer.push_back(block + 1);
	    		counter_curr += 3;
	    	}
		}
	    src.submat(0, block + 2, l_L_append_2 - 1, block + 3 - 1) = arma::reshape(L_append_2.t(), l_L_append_2, 1);
	    // src.submat(0, block + 2, l_L_append_2 - 1, block + 3 - 1).print();
	    for (int index_r = 0; index_r < l_L_append_2; index_r++) {
	    	double element_curr = src(index_r, block + 2);
	    	// std::cout << "yo it's " << p << " and i gots the " << element_curr << " at " << index_r << " and " << block + 2 
	    	// << " at " << counter_curr << " with len = " << *len << "\n";
	    	if (element_curr != 0 && counter_curr < *len) {
	    		*inout_ret = element_curr; inout_ret++;
	    		*inout_ret = index_r; inout_ret++; *inout_ret = block + 2; inout_ret++;
	    		counter_curr += 3;
	    	} else if (element_curr != 0 && counter_curr >= *len) {
	    		// std::cout << "l_append_2, " << counter_curr << ": " << element_curr <<  "\n";
	    		buffer.push_back(element_curr); buffer.push_back(index_r); buffer.push_back(block + 2);
	    		counter_curr += 3;
	    	}
		}

		// src.submat(0, 0, 9, src.n_cols - 1).print(); std::cout << "\n-\n";
		// src.submat(src.n_rows - 10, 0, src.n_rows - 1, src.n_cols - 1).print(); std::cout << "\n---\n";
	}
	// for (i = 0; i < *len * 2; i+=3) {
	// 	std::cout << *inout_reset; inout_reset++;
	// 	std::cout << " | row: " << *inout_reset; inout_reset++;
	// 	std::cout << ", col: " << *inout_reset << "\n"; inout_reset++;
	// }

	// if (buffer.size() != 0 and p == (size - 1)) {
	// 	printVec(buffer);
	// }
}

//         src[0:n+3,0:A_B_.shape[1]] = np.concatenate((A_B_,[proc_v[B_],indices[B_],real_indices[B_]]),axis=0)
//         src[0:n,block:block+1] = b_sol__

//         l_L_append_1 = L_append_1.shape[0]*L_append_1.shape[1]
//         src[0:l_L_append_1,block+1:block+2] = L_append_1.reshape(l_L_append_1,1)

//         l_L_append_2 = L_append_2.shape[0]*L_append_2.shape[1]
//         src[0:l_L_append_2,block+2:block+3] = L_append_2.reshape(l_L_append_2,1)
        


stdvec mpi4py_tournament_bLARS_svm_clever(double block) {
    MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &p);
	std::string pStr = std::to_string(p);
	std::vector<stdvec> aDataP = aRet("dT" + pStr);
	std::vector<stdvec> aRowP = aRet("rT" + pStr);
	stdvec aRowI = bRet("arI" + pStr);
	stdvec b_vec_temp = bRet("bT0"); std::vector<int> mn = mnRet();
	b = arma::vec(b_vec_temp);
	int m = mn[0]; int n = mn[1]; int min_temp = std::min(m, n); 
	int iterations = std::ceil(min_temp/block);
	int og_block = (int) block; n_g = n;


	int m_local = aDataP.size(); int ct; int counting = 0;
	arma::mat A_local_T = arma::mat(m_local, n);
	for (int row = 0; row < m_local; row++) {
		ct = 0;
		for (int col = 0; col < aRowP[row].size(); col++) {
			A_local_T(row, aRowP[row][col]) = aDataP[row][ct];
			ct = ct + 1;
		}
	}
	arma::mat A_local = A_local_T.t();

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
        if (p == 0) {
        	std::cout << "Iteration No: " << counter << "\n";
        }
        stdmat d = bLARS_tournament(A_chosen, selectCols(A_local, A_hallow_local_c), b, how_many, b_sol, L, true);

        // how many = block

        arma::vec b_sol_ = d[0]; arma::vec B = d[1]; A_B = d[2]; 
        L_append_1 = d[3]; L_append_2 = d[4];
        
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

        arma::sp_mat src_sparse = arma::sp_mat(src); stdvec src_vector;
        arma::sp_mat::iterator it = src_sparse.begin(); arma::sp_mat::iterator it_end = src_sparse.end();

        for(; it != it_end; ++it) {
        	src_vector.push_back(*it); src_vector.push_back(it.row()); src_vector.push_back(it.col()); 
		}

		// if (p == 0) {
		// 	for (int i = 0; i < 30; i++) {
		// 		std::cout << src_vector[i] << " ";
		// 		if (i % 3 == 2) {
		// 			std::cout << "\n";
		// 		}
		// 	}
		// }

		int size_of_src[1]; size_of_src[0] = src_vector.size();	int req_size[1];
		MPI_Allreduce(size_of_src, req_size, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

		while (src_vector.size() < (req_size[0])) {
			src_vector.push_back(0);
		}

        double src_arr[req_size[0]]; double dst_arr[req_size[0]];
        for (int i = 0; i < src_vector.size(); i++) {src_arr[i] = src_vector[i];}

        MPI_Allreduce(src_arr, dst_arr, src_vector.size(), MPI_DOUBLE, op, MPI_COMM_WORLD); 

		for (int i = 0; i < req_size[0]; i+=3) {
			if (dst_arr[i + 1] == 0 && dst_arr[i + 2] == 0) {
				if (dst(dst_arr[i + 1], dst_arr[i + 2]) == 0) {
					dst(dst_arr[i + 1], dst_arr[i + 2]) = dst_arr[i];
				} 
			}
        	dst(dst_arr[i + 1], dst_arr[i + 2]) = dst_arr[i];
        }

        int size_container[1]; size_container[0] = buffer.size();
        MPI_Bcast(size_container, 1, MPI_INT, size - 1, MPI_COMM_WORLD);

        if (size_container[0] != 0) {
        	double buffer_extra[size_container[0]];
        	if (p == (size - 1)) {
	        	for (int i = 0; i < buffer.size(); i++) {
	        		buffer_extra[i] = buffer[i];
	        	}
        	}
        	MPI_Bcast(buffer_extra, size_container[0], MPI_DOUBLE, size - 1, MPI_COMM_WORLD);

	        for (int i = 0; i < size_container[0]; i += 3) {
	        	if (dst(buffer_extra[i + 1], buffer_extra[i + 2]) == 0) {
	        		dst(buffer_extra[i + 1], buffer_extra[i + 2]) = buffer_extra[i];	  
	        	} 		
	    	}
	    	stdvec resetVec; buffer = resetVec;
    	}

  //       if (p == 0) {
		// 	dst.submat(0, 0, 9, dst.n_cols - 1).print(); std::cout << "\n-\n";
		// 	dst.submat(dst.n_rows - 10, 0, dst.n_rows - 1, dst.n_cols - 1).print(); std::cout << "\n---\n";
		// }

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
    }
    printVec(A_hallow_winners);
    MPI_Finalize();
    return A_hallow_winners;
}

int main(int argc, char** argv) {
	// arma::mat lmao = arma::mat(5, 5);
	// lmao.randu(); lmao.print();
	// std::cout << "!!!!!!\n";
	// arma::vec ret_vec = arma::reshape(lmao, lmao.n_elem, 1);
	// ret_vec.print();
	// return 0;


	std::stringstream convert(argv[1]); 
	int block_size;
	if (!(convert >> block_size)) {
		block_size = 1;
		std::cout << "Unable to parse number! Default LARS executed.\n";
	} else {
		std::cout << "We have a block size of " << block_size << ".\n";
	}
	stdvec to_ret = mpi4py_tournament_bLARS_svm_clever(block_size);
	printVec(to_ret);
	return 0;
}

