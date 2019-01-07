// sample to test out root finding for poly 

#include <stdio.h>
#include <iostream>
#include <math.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

using namespace Eigen;

std::vector<std::complex<double> > poly_roots(VectorXd coeffs){
    int sz = coeffs.size() - 1;
    std::vector<std::complex<double> > vret;

    MatrixXd companion_mat(sz, sz);

    for(int n = 0; n < sz; n++){
        for(int m = 0; m < sz; m++){
            if(n == m + 1){
                companion_mat(n,m) = 1.0;
            }
            if(m == sz - 1){
                companion_mat(n,m) = -coeffs(n)/coeffs(sz);
            }
        }
    }

    MatrixXcd eig = companion_mat.eigenvalues();

    for(int i = 0; i < sz; i++)
        vret.push_back( eig(i) );

    return vret;
}

int main(int argc, char** argv){
    VectorXd coeff = VectorXd::Zero(argc-1);
    for (int i = 1; i < argc; i++){
        coeff(i-1) = atof(argv[i]);
    }
    std::vector<std::complex<double> > result = poly_roots(coeff);
    std::cout << "roots: ";
    for (int i = 0; i < result.size(); i++){
        std::cout << result[i];
    }
    std::cout << std::endl; 
}
