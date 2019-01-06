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

void get_Veronese_map(cv::Point2f ptx, VectorXd& v, int deg){
	// assume image pt x y z has x y corr to keypoint and z = 1
	double x = ptx.x; double y = ptx.y; double z = ptx.z; 
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
		get_Veronese_map(x2[i], v2, n);
		get_Veronese_map(x1[i], v1, n);
		get_Kronecker_product(v2,v1,row);
		A.row(i) = row; 
	}
	return; 
}

int getNoM(std::vector<cv::Point2f> x1, std::vector<cv::Point2f> x2){
	// image pairs of points keyponts1 and keypoints 2
	// return number of motions 
	int N = x1.size();
	for (int i = 1; i < N + 1; i++){
		MatrixXd Ai; 
		get_matrix_A(x1, x2, i, Ai);
		FullPivLU<MatrixXd> lu_decomp(Ai);
		Ci = (i + 2)*(i + 1)/2; 
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
	MaxtrixXd K = lu_decomp.kernel(); 
	// assume only one col 
	VectorXd Fs = K.col(0); // stacked F
	int Cn = (int) sqrt(A.cols());
	F = MaxtrixXd::Zero(Cn, Cn); 
	// unstack Fs into F 
	for (int i = 0; i < Cn; i++){
		for (int j = 0; j < Cn; j++){
			F(j,i) = Fs(Cn*i + j); 
		}
	}
	return;
}

void findEpipolarLines()