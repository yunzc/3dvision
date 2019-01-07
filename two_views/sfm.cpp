/*
Corresponding to chapter 7 of book 
Given a collection of image pairs correspoinding to 
an unknown number of indepedent and rigidly moving 
objects, estimate the number of independent motions, 
the fundamental matrices, and the object to which 
each image pair belongs
*/ 

#include <stdio.h>
#include <iostream>
#include <limits>
#include <math.h>
#include <cmath>
#include <time.h>  
#include <assert.h>    

#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

using namespace Eigen;

void readme(){
	std::cout << " Usage: ./2_views/sfm <img1> <img2>" << std::endl; 
}

void match_pts(cv::Mat img1, cv::Mat img2, std::vector<cv::KeyPoint>& keypoints1, 
			std::vector<cv::KeyPoint>& keypoints2, std::vector<cv::DMatch>& good_matches){

	// SIFT detector for keypoint detection 
	cv::Ptr<cv::xfeatures2d::SIFT> detector = cv::xfeatures2d::SIFT::create(); 

	cv::Mat descriptors1, descriptors2; 

	detector->detectAndCompute(img1, cv::Mat(), keypoints1, descriptors1);
	detector->detectAndCompute(img2, cv::Mat(), keypoints2, descriptors2);

	// FLANN based matcher
	cv::FlannBasedMatcher matcher; 
	std::vector<cv::DMatch> matches; 
	matcher.match(descriptors1, descriptors2, matches); 
	double max_dist = 0; double min_dist = 100;
	//-- Quick calculation of max and min distances between keypoints
	for( int i = 0; i < descriptors1.rows; i++ ){ 
		double dist = matches[i].distance;
		if( dist < min_dist ) min_dist = dist;
		if( dist > max_dist ) max_dist = dist;
	}

	for (int i = 0; i < descriptors1.rows; i++){
		if (matches[i].distance <= MAX(5.0*min_dist, 0.02)){
			good_matches.push_back(matches[i]);
		}
	}
}

void get_Veronese_map(Vector3d ptx, VectorXd& v, int deg){
	// assume image pt x y z has x y corr to keypoint and z = 1
	double x = ptx(0); double y = ptx(1); double z = ptx(2); 
	if (deg == 0){
		v = VectorXd::Ones(0); 
		return;
	}else if (deg == 1){
		v = VectorXd::Zero(3);
		v(0) = x; v(1) = y; v(2) = z; 
		return; 
	}else{
		int len = (deg + 2)*(deg + 1)/2;
		v = VectorXd::Zero(len);
		// get veronses for one lesser degree 
		VectorXd v_less;
		get_Veronese_map(ptx, v_less, deg-1);
		int len_less = v_less.size(); // get len of the lesser degree veronese 
		assert (len == len_less + deg + 1); // this condition should hold 
		for (int i = 0; i < len_less; i++){
			v(i) = x*v_less(i);
		}
		// fill in the ones without the x but has y 
		for (int j = 0; j < deg; j++){
			v(len_less + j) = y*v_less(len_less - deg + j);
		}
		// the one that only has z
		v(len_less + deg) = z*v_less(len_less - 1);
		return; 
	}
}

void get_Kronecker_product(VectorXd v1, VectorXd v2, VectorXd& result){
	int v1_size = v1.size();
	int v2_size = v2.size();
	result = VectorXd::Zero(v1_size*v2_size);
	for (int i = 0; i < v1_size; i++){
		result.segment(v2_size*i, v2_size) = v1(i)*v2;
	}
	return;
}

void get_matrix_A(std::vector<cv::Point2f> x1, std::vector<cv::Point2f> x2, int n, MatrixXd& A){
	int N = x1.size(); 
	int Cn = (n+2)*(n+1)/2;
	A = MatrixXd::Zero(N,Cn*Cn);
	for (int i = 0; i < N; i++){
		VectorXd v1, v2, row; 
		Vector3d x1i, x2i;
		x1i(0) = x1[i].x; x1i(1) = x1[i].y; x1i(2) = 1; 
		x2i(0) = x2[i].x; x2i(1) = x2[i].y; x2i(2) = 1; 
		get_Veronese_map(x2i, v2, n);
		get_Veronese_map(x1i, v1, n);
		get_Kronecker_product(v2,v1,row);
		A.row(i) = row; 
	}
	return; 
}

std::vector<std::complex<double> > poly_roots(VectorXd coeffs){
	// note coeff is orders so lower order first 
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

int getNoM(std::vector<cv::Point2f> x1, std::vector<cv::Point2f> x2){
	// image pairs of points keyponts1 and keypoints 2
	// return number of motions 
	int N = x1.size();
	for (int i = 1; i < N + 1; i++){
		if (x1.size() < pow(i,4)){
			std::cout << "****** Not Enough Image Pairs *****" << std::endl; 
			return -1; 
		}
		MatrixXd Ai; 
		get_matrix_A(x1, x2, i, Ai);
		FullPivLU<MatrixXd> lu_decomp(Ai);
		int Ci = (i + 2)*(i + 1)/2; 
		int rankA = lu_decomp.rank();
		if (rankA == Ci*Ci - 1){
			return i; 
		}
	}
	return -1;
}

void computeFundamentalMtrx(MatrixXd A, MatrixXd& F){
	// computes the multibody fundamental matrix 
	FullPivLU<MatrixXd> lu_decomp(A);
	// find kernel 
	MatrixXd K = lu_decomp.kernel(); 
	// assume only one col 
	VectorXd Fs = K.col(0); // stacked F
	int Cn = (int) sqrt(A.cols());
	F = MatrixXd::Zero(Cn, Cn); 
	// unstack Fs into F 
	for (int i = 0; i < Cn; i++){
		for (int j = 0; j < Cn; j++){
			F(j,i) = Fs(Cn*i + j); 
		}
	}
	return;
}

void findEpipolarLines(MatrixXd F, int n){
	// use factorization algorithm 
	int Cn = (n + 2)*(n + 1)/2; 
	// pick N >= Cn - 1 vectors 
	// with [1,0,0], [0,1,0], and [0,0,1] included 
	std::vector<MatrixXd> l; // the many epipolar lines  
	// first look at the 3 basic vectors 
	for (int i = 0; i < 3; i++){
		Vector3d x = Vector3d::Zero(3);
		x(i) = 1;
		VectorXd v, l_mb; 
		get_Veronese_map(x, v, n);
		// get multibody epipolar line 
		l_mb = F*v;
		std::cout << "solve: " << std::endl; 
		// reverse since lower order first in poly_roots function 
		std::vector<std::complex<double> > wi = poly_roots(l_mb.tail(n+1).reverse());
		std::cout << "roots: ";
    	for (int i = 0; i < wi.size(); i++){
        	std::cout << wi[i];
        	// wi.size() should equl n 
    		double l2 = 1; double l3 = (double)-wi[i].real();

    	}
    	std::cout << std::endl; 
	}
	while (l.size() < Cn - 1){ // N >= cn - 1
		Vector3d x = Vector3d::Zero(3);
		// randomy create vector 
		VectorXd v, l_mb; 
		get_Veronese_map(x, v, n); 
		l_mb = F*v;
	}
	
}

int main(int argc, char**argv){
	if (argc != 3){
		readme(); 
		return -1; 
	}

	cv::Mat img_1 = cv::imread(argv[1]);
	cv::Mat img_2 = cv::imread(argv[2]);
	cv::Mat img_1_gray, img_2_gray; 
	cv::cvtColor(img_1, img_1_gray, cv::COLOR_BGR2GRAY);
	cv::cvtColor(img_2, img_2_gray, cv::COLOR_BGR2GRAY);
	std::vector<cv::KeyPoint> kp1, kp2;
	std::vector<cv::DMatch> matches; 
	match_pts(img_1_gray, img_2_gray, kp1, kp2, matches);

	// draw good matches 
	cv::Mat img_matches;
	cv::drawMatches(img_1, kp1, img_2, kp2, matches, img_matches, cv::Scalar::all(-1), 
				cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
	//-- Show detected matches
	// cv::imshow( "Good Matches", img_matches );
	// cv::waitKey(0);
	// x1 are img pts from first image 
	// x2 are img pts from snd image
	std::vector<cv::Point2f> x1;
	std::vector<cv::Point2f> x2;
	for( int i = 0; i < matches.size(); i++ ){
    	//-- Get the keypoints from the good matches
    	x1.push_back(kp1[matches[i].queryIdx].pt);
    	x2.push_back(kp2[matches[i].trainIdx].pt);
	}

	int NOM = getNoM(x1, x2); // find number of motions 
	std::cout << "Number of motions: " << NOM << std::endl; 
	MatrixXd An, F; 
	get_matrix_A(x1, x2, NOM, An);
	computeFundamentalMtrx(An, F);
	findEpipolarLines(F, NOM);
}