cmake_minimum_required(VERSION 2.8)
project(3dvision)
find_package( OpenCV REQUIRED )
find_package(Eigen3 REQUIRED)
include_directories( 
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS} 
)

add_executable(two_views/image_stitch two_views/image_stitch.cpp)
target_link_libraries(two_views/image_stitch ${OpenCV_LIBS} ${Eigen3_LIBRARIES})