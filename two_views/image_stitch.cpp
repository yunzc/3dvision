/*
First use SURF FlannMatcher for matching keypoints 
between two images 
*/ 

#include <stdio.h>
#include <iostream>
#include <limits>
#include <math.h>
#include <time.h>      

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
	std::cout << " Usage: ./2_views/image_stitch.cpp <img1> <img2>" << std::endl; 
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

void stitch_images(cv::Mat left_img, cv::Mat warpped_image, cv::Mat& result){
	cv::Size ls = left_img.size();
	cv::Size ws = warpped_image.size();
	result = warpped_image; 
	for (int i = 0; i < ls.width; i++){
		for (int j = 0; j < ls.height; j++){
			cv::Vec3b color = left_img.at<cv::Vec3b>(cv::Point(i,j));
			if (color[0] != 0 || color[1] != 0 || color[2] != 0){
				result.at<cv::Vec3b>(cv::Point(i,j)) = color;
			}
		}
	}
}

void findSampleHomography(std::vector<cv::Point2f> x1, std::vector<cv::Point2f> x2, cv::Mat& H){
	// first find X matrix X = [a1, a2, ..., an]^T
	MatrixXd X = MatrixXd::Zero(3*x1.size(), 9);
	for (int i = 0; i < x1.size(); i++){
		// ai = x1i kronecker mult hat(x2i) [9 by 3]
		// x1i = [x1, y1, 1]^T, hat(x2i) = [0 -1 y2; 1 0 -x2; -y2 x2 0]
		Matrix3d x2i_hat = Matrix3d::Zero(3,3);
		x2i_hat(0,1) = -1; x2i_hat(0,2) = x2[i].y; 
		x2i_hat(1,0) = 1; x2i_hat(1,2) = -x2[i].x; 
		x2i_hat(2,0) = -x2[i].y; x2i_hat(2,1) = x2[i].x; 
		MatrixXd ai = MatrixXd::Zero(9,3); 
		ai.block(0,0,3,3) = x1[i].x*x2i_hat; 
		ai.block(3,0,3,3) = x1[i].y*x2i_hat;
		ai.block(6,0,3,3) = x2i_hat;
		// Fill in the matrix 
		X.block(3*i,0,3,9) = ai.transpose(); 
	}
	// SVD of X
	JacobiSVD<MatrixXd> svd(X, ComputeThinU | ComputeThinV);
	MatrixXd Vx = svd.matrixV();
    MatrixXd Dx = svd.singularValues().asDiagonal();
    // find index of smallest singular value
   	double min_val = std::numeric_limits<double>::infinity();
   	int min_idx; 
    for (int i = 0; i < Dx.cols(); i++){
    	if (Dx(i,i) < min_val){
    		min_idx = i; 
    		min_val = Dx(i,i);
    	}
    }
    Matrix3d Hl = Matrix3d::Zero(3,3);
    // unstack 
    Hl(0,0) = Vx(0,min_idx); 
    Hl(1,0) = Vx(1,min_idx); 
    Hl(2,0) = Vx(2,min_idx); 
    Hl(0,1) = Vx(3,min_idx); 
    Hl(1,1) = Vx(4,min_idx); 
    Hl(2,1) = Vx(5,min_idx); 
    Hl(0,2) = Vx(6,min_idx); 
    Hl(1,2) = Vx(7,min_idx); 
    Hl(2,2) = Vx(8,min_idx);  

    // test sign of Hl 
    Vector3d x10; Vector3d x20; 
    x10(0) = x1[0].x; x10(1) = x1[0].y; x10(2) = 1; 
    x20(0) = x2[0].x; x20(1) = x2[0].y; x20(2) = 1; 
    // std::cout << "x1: " << Hl*x10/Hl(2,2) << "x2: " << x20 << std::endl; 
    if (x20.transpose()*Hl*x10 < 0){
    	Hl = -Hl; 
    }
    // normalize for H 
    JacobiSVD<MatrixXd> svdHl(Hl, ComputeThinU | ComputeThinV);
	Matrix3d Dhl = svdHl.singularValues().asDiagonal();
	H = cv::Mat(3,3, CV_64F, double(0));
	for (int i = 0; i < 3; i++){
		for (int j = 0; j < 3; j++){
			H.at<double>(i,j) = Hl(i,j)/Hl(2,2);
		}
	}
}

void findHomography(std::vector<cv::Point2f> x1, std::vector<cv::Point2f> x2, cv::Mat& H){
	// RANSAC find homography 
	int iters = 500; 
	double epsilon = 10; 
	cv::Mat best_H; 
	int max_num_inliers = 0;
	// randomly choose 4 points and find sample homography 
	std::vector<int> indices; 
	for (int n = 0; n < iters; n++){
		indices.clear(); 
		// randomly choose 4 points 
		for (int i = 0; i < 4; i++){
			bool dup = false; // check for duplicates
			int idx; 
			do{
				idx = rand() % x1.size(); 
				for (int j = 0; j < indices.size(); j++){
					if (indices[j] == idx){
						dup = true; 
						break; 
					}
				}
			} while(dup);
			indices.push_back(idx);
		}
		// find sample homgraphy
		cv::Mat H_samp; 
		std::vector<cv::Point2f> x1_pts; 
		std::vector<cv::Point2f> x2_pts; 
		for (int i = 0; i < indices.size(); i++){
			x1_pts.push_back(x1[indices[i]]);
			x2_pts.push_back(x2[indices[i]]);
		}
		findSampleHomography(x1_pts, x2_pts, H_samp);
		// count number of inliers 
		int num_inliers = 0;
		for (int k = 0; k < x1.size(); k++){
			// x1_ x1 transformed 
			double x1_x = H_samp.at<double>(0,0)*x1[k].x + H_samp.at<double>(0,1)*x1[k].y 
							+ H_samp.at<double>(0,2);
			double x1_y = H_samp.at<double>(1,0)*x1[k].x + H_samp.at<double>(1,1)*x1[k].y
							+ H_samp.at<double>(1,2); 
			double x1_z = H_samp.at<double>(2,0)*x1[k].x + H_samp.at<double>(2,1)*x1[k].y
							+ H_samp.at<double>(2,2); 
			x1_x = x1_x/x1_z; x1_y = x1_y/x1_z; 
			// std::cout << x1_x << " " << x1_y << " " << x2[k].x << " " << x2[k].y << std::endl; 
			if (sqrt((x1_x - x2[k].x)*(x1_x - x2[k].x) + (x1_y - x2[k].y)*(x1_y - x2[k].y)) < epsilon){
				num_inliers = num_inliers + 1; 
			}
		}
		std::cout << "iter: " << n << " num inliers: " << num_inliers << std::endl; 
		if (num_inliers > max_num_inliers){
			max_num_inliers = num_inliers; 
			best_H = H_samp; 
		}
	}
	H = best_H; 
}

int main(int argc, char**argv){
	if (argc != 3){
		readme(); 
		return -1; 
	}
	// random module 
	srand (time(NULL));

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

	// for( int i = 0; i < (int)matches.size(); i++ ){ 
	// 	printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, matches[i].queryIdx, matches[i].trainIdx );
	// 	std::cout << kp1[matches[i].queryIdx].pt << " " << kp2[matches[i].trainIdx].pt << std::endl; 
	// }

	// Find homography (fill in with something else later)
	std::vector<cv::Point2f> scene1;
	std::vector<cv::Point2f> scene2;
	for( int i = 0; i < matches.size(); i++ ){
    	//-- Get the keypoints from the good matches
    	scene1.push_back(kp1[matches[i].queryIdx].pt);
    	scene2.push_back(kp2[matches[i].trainIdx].pt);
	}

	cv::Mat H;// = cv::findHomography(scene2, scene1, cv::RANSAC );
	// H transforms scene2 to plane of scene1
	findHomography(scene2, scene1, H);
	double tx, ty, tz; 
	tx = H.at<double>(0,0)*img_2.size().width 
				+ H.at<double>(1,0)*img_2.size().height + H.at<double>(2,0); 
	ty = H.at<double>(0,1)*img_2.size().width 
				+ H.at<double>(1,1)*img_2.size().height + H.at<double>(2,1); 
	tz = H.at<double>(0,2)*img_2.size().width 
				+ H.at<double>(1,2)*img_2.size().height + H.at<double>(2,2); 

	std::cout << "Tx: " << tx << "ty: " << ty << std::endl; 

	cv::Size s; 
	s.width = tx + img_1.size().width; s.height = ty + img_1.size().height;
	cv::Mat warp2; 
	cv::warpPerspective(img_2, warp2, H, s);
	// stitch 
	cv::Mat result; 
	stitch_images(img_1, warp2, result);
	cv::imshow("result", result);
	cv::waitKey(0);
      
	return 0;
}