/*
This module is used to construct some simple stereo images 
given some world points. 
From a text file: 
- the first line reads the world coordinates center of a circle 
with radius of 1 (all units are meters)
- the second line reads the world coordinates center of a line
with length 2 (so two endpoints each 1m away from center)
- 3rd is a triangle with each vertex 1m away from center
- etc. etc. so on and so forth 

wx wy wz is rotation direction vector 

Assume for the first view all the shapes are planar to image 
*/ 

#include <stdio.h>
#include <string>
#include <fstream>
#include <iostream>
#include <limits>
#include <utility>
#include <math.h>
#include <time.h>    

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#define _USE_MATH_DEFINES
using namespace Eigen;

void readme(){
	std::cout << 
	" Usage: ./2_views/image_construct <textfile> <Tx> <Ty> <Tz> <wx> <wy> <wz> <theta>" 
	<< std::endl; 
}

class ImageConstruction{
	Vector3d w; // omega vector for rotation direction (unit vector)
	Vector3d T; // translation between two views 
	double theta; // basically magnitude of rotation 
	Matrix3d K; // intrinsic 
	MatrixXd PI; // perspective projection matrix 

	// screen size 
	int window_h, window_w; 

	// images 
	cv::Mat first_image; 
	cv::Mat second_image; 

	std::vector<std::vector<double> > obj_coords; 
	void gen_first_view_vertices(std::vector<std::vector<cv::Point> >& vertices);
	void gen_second_view_vertices(std::vector<std::vector<cv::Point> >& second_view_vertices);
public:
	ImageConstruction(double tx, double ty, double tz, double wx, double wy, double wz, double theta);
	void read_file(std::string textfile);
	void generate_images();
	void save_images(std::string fstv_filename, std::string scdv_filename);
};

ImageConstruction::ImageConstruction(double tx, double ty, double tz, double wx, 
									double wy, double wz, double theta){
	// store values 
	T(0) = tx; T(1) = ty; T(2) = tz; 
	w(0) = wx; w(1) = wy; w(2) = wz;
	theta = theta; 

	// get screen size 
    window_w  = 3840;
    window_h = 2160;

    // Init camera matrices 
    // First K
    double f = 600; // can be tuned but this for now 
    double ox = (int) window_w/2; 
    double oy = (int) window_h/2;
    K = Matrix3d::Zero(3,3);
    K(0,0) = f; K(0,1) = f; K(1,1) = f; 
    K(0,2) = ox; K(1,2) = oy; 
    K(2,2) = 1; 
    // Then PI 
    PI = MatrixXd::Zero(3,4); 
    PI.block(0,0,3,3) = Matrix3d::Identity(3,3); 
}

void ImageConstruction::read_file(std::string textfile){
	// read from file to get coordinates 
	std::ifstream inFile; 
	inFile.open(textfile.c_str());
	if (!inFile.is_open()){
		std::cout << "error opening textfile" << std::endl; 
		return;
	}
	// first get first line for number of vertices 
	std::string line; 
	while (std::getline(inFile, line)){
		std::stringstream ss(line); 
		double x, y, z; 
		ss >> x >> y >> z;
		std::vector<double> pt; 
		pt.push_back(x); 
		pt.push_back(y);
		pt.push_back(z);
		obj_coords.push_back(pt);
	}	
}

void pts_to_param_ellipse(std::vector<cv::Point> pts,
			double& width, double& height, double& ang){
	// given 3 pts, generate params to dranw with opencv 
	double cx = pts[0].x; double cy = pts[0].y; 
	double x1 = pts[1].x; double y1 = pts[1].y; 
	double x2 = pts[2].x; double y2 = pts[2].y; 
	ang = atan2(y1, x1); // get angle 
	width = sqrt((x1-cx)*(x1-cx) + (y1-cy)*(y1-cy)); 
	height = sqrt((x2-cx)*(x2-cx) + (y2-cy)*(y2-cy));
}

void ImageConstruction::gen_first_view_vertices(std::vector<std::vector<cv::Point> >& vertices){
	if (obj_coords.size() > 0){
		// first object will be a circle give 3 pts as if defining an ellipse 
		Vector4d c, a, b; 
		Vector3d img_c, img_a, img_b; 
		c(0) = obj_coords[0][0]; c(1) = obj_coords[0][1]; 
		c(2) = obj_coords[0][2]; c(3) = 1; // homogeneous 
		a = c; a(0) = c(0) + 1; 
		b = c; b(1) = c(1) + 1; 
		// camera perspective transform
		img_c = K*PI*c; img_a = K*PI*a; img_b = K*PI*b; 
		cv::Point pc, pa, pb; 
		pc.x = img_c(0); pc.y = img_c(1); pa.x = img_a(0); 
		pa.y = img_a(1); pb.x = img_b(0); pb.y = img_b(1);
		std::vector<cv::Point> pts;
		pts.push_back(pc); pts.push_back(pa); pts.push_back(pb);
		vertices.push_back(pts);
	}
	if (obj_coords.size() > 1){
		// line and polygons 
		for (int i = 1; i < obj_coords.size(); i++){
			std::vector<cv::Point> pts;
			Vector4d c; // center 
			c(0) = obj_coords[i][0]; c(1) = obj_coords[i][1]; 
			c(2) = obj_coords[i][2]; c(3) = 1; 
			double in_ang = 2*M_PI/(i + 1); // inner angle 
			for (int k = 0; k < (i+1); k++){
				double ang = in_ang*k; 
				Vector4d pt; // point in homogenious coords
				pt(0) = c(0) + cos(ang); pt(1) = c(1) + sin(ang);
				Vector3d img_pt; 
				img_pt = K*PI*pt; 
				cv::Point p; p.x = img_pt(0); p.y = img_pt(1); 
				pts.push_back(p); // add to point of line/polygon
			}
			vertices.push_back(pts); 
		}	
	}
}

void ImageConstruction::gen_second_view_vertices(std::vector<std::vector<cv::Point> >& vertices){
	// pretty similar to gen first view but with additional extrinsic parameter matrix 
	// create extrinsic camera matrix
	Matrix4d g; 
	// first find R 
	Matrix3d R, w_hat; 
	w = theta*w.normalized(); 
	// constrict w_hat
	w_hat(0,1) = -w(2); w_hat(0,2) = w(1); 
	w_hat(1,0) = w(2); w_hat(1,2) = -w(0);
	w_hat(2,0) = -w(1); w_hat(2,1) = w(0);
	R = Matrix3d::Identity(3,3) + (1/theta)*w_hat*sin(theta) 
				+ (1/theta*theta)*(w_hat*w_hat)*(1 - cos(theta));
	// construct g
	g.block(0,0,3,3) = R; 
	g.block(0,3,3,1) = T; 
	g(3,3) = 1; 
	if (obj_coords.size() > 0){
		// first object will be a circle give 3 pts as if defining an ellipse 
		Vector4d c, a, b; 
		Vector3d img_c, img_a, img_b; 
		c(0) = obj_coords[0][0]; c(1) = obj_coords[0][1]; 
		c(2) = obj_coords[0][2]; c(3) = 1; // homogeneous 
		a = c; a(0) = c(0) + 1; 
		b = c; b(1) = c(1) + 1; 
		// camera perspective transform
		img_c = K*PI*g*c; img_a = K*PI*g*a; img_b = K*PI*g*b;
		cv::Point pc, pa, pb; 
		pc.x = img_c(0); pc.y = img_c(1); pa.x = img_a(0); 
		pa.y = img_a(1); pb.x = img_b(0); pb.y = img_b(1);
		std::vector<cv::Point> pts;
		pts.push_back(pc); pts.push_back(pa); pts.push_back(pb);
		vertices.push_back(pts);
	}
	if (obj_coords.size() > 1){
		// line and polygons 
		for (int i = 1; i < obj_coords.size(); i++){
			std::vector<cv::Point> pts;
			Vector4d c; // center 
			c(0) = obj_coords[i][0]; c(1) = obj_coords[i][1]; 
			c(2) = obj_coords[i][2]; c(3) = 1; 
			double in_ang = 2*M_PI/(i + 1); // inner angle 
			for (int k = 0; k < (i+1); k++){
				double ang = in_ang*k; 
				Vector4d pt; // point in homogenious coords
				pt(0) = c(0) + cos(ang); pt(1) = c(1) + sin(ang);
				Vector3d img_pt; 
				img_pt = K*PI*g*pt; 
				cv::Point p; p.x = img_pt(0); p.y = img_pt(1); 
				pts.push_back(p); // add to point of line/polygon
			}
			vertices.push_back(pts); 
		}	
	}
}

void ImageConstruction::generate_images(){
	std::vector<std::vector<cv::Point> > vertices_1; 
	std::vector<std::vector<cv::Point> > vertices_2; 
	// generate the vertices 
	gen_first_view_vertices(vertices_1);
	gen_second_view_vertices(vertices_2);
	
	first_image = cv::Mat::zeros(window_h, window_h, CV_8UC3);
	second_image = cv::Mat::zeros(window_h, window_h, CV_8UC3);
	if (vertices_1.size() > 0){
		// [later] first vertex is of the ellipse (circle in first view)
		// gen vertices will give center, and two points 
		// (coor to minor axis and major axis respc)
		double w1, h1, ang1, w2, h2, ang2; 
		pts_to_param_ellipse(vertices_1[0], w1, h1, ang1);
		pts_to_param_ellipse(vertices_2[0], w2, h2, ang2);
		cv::ellipse(first_image, vertices_1[0][0], cv::Size(w1, h1), ang1,
           		0, 360, cv::Scalar( 255, 0, 0 ), 3, 8);
		cv::ellipse(second_image, vertices_2[0][0], cv::Size(w2, h2), ang2,
           		0, 360, cv::Scalar( 255, 0, 0 ), 3, 8);
	}
	// second set of vertices are lines 
	if (vertices_1.size() > 1){
		cv::line(first_image, vertices_1[1][0], vertices_1[1][1], cv::Scalar(0,0,255),3,8);
		cv::line(second_image, vertices_2[1][0], vertices_2[1][1], cv::Scalar(0,0,255),3,8);
	}
	// then we have the polygons 
	if (vertices_1.size() > 2){
		for (int i=2; i < vertices_1.size(); i++){
			int numpts = (int)vertices_1[i].size(); 
			cv::Point points_1[1][numpts];
			cv::Point points_2[1][numpts];
			for (int j=0; j < vertices_1[i].size(); j++){
				points_1[0][j] = vertices_1[i][j];
				points_2[0][j] = vertices_2[i][j];
			}
			const cv::Point* ppt_1[1] = {points_1[0]};
			int npt_1[] = {numpts};
			const cv::Point* ppt_2[1] = {points_2[0]};
			int npt_2[] = {numpts};
			cv::fillPoly(first_image, ppt_1, npt_1, 1, cv::Scalar(0,0,255),8);
			cv::fillPoly(second_image, ppt_2, npt_2, 1, cv::Scalar(0,0,255),8);		
		}
	}
}

void ImageConstruction::save_images(std::string fstv_filename, std::string scdv_filename){
	cv::imwrite(fstv_filename.c_str(), first_image);
	cv::imwrite(scdv_filename.c_str(), second_image);
}


int main(int argc, char**argv){
	if (argc != 9){
		readme(); 
		return -1; 
	}
	// initialize and read arg 
	ImageConstruction cons(atof(argv[2]), atof(argv[3]), atof(argv[4]), 
			atof(argv[5]), atof(argv[6]), atof(argv[7]), atof(argv[8]));

	cons.read_file(argv[1]);
	cons.generate_images();
	cons.save_images("scene1.jpg", "scene2.jpg");
}