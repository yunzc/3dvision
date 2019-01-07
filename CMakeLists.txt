cmake_minimum_required(VERSION 2.8)
project(3dvision)
add_compile_options(-std=c++11)

find_package( OpenCV REQUIRED )
find_package(Eigen3 REQUIRED)

include_directories( 
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS} 
)

add_executable(two_views/image_stitch two_views/image_stitch.cpp)
add_executable(two_views/image_construct two_views/image_construct.cpp)
add_executable(two_views/sfm two_views/sfm.cpp)
add_executable(two_views/root_finding_test two_views/root_finding.cpp)
target_link_libraries(two_views/image_stitch ${OpenCV_LIBS} ${Eigen3_LIBRARIES})
target_link_libraries(two_views/image_construct ${OpenCV_LIBS} ${Eigen3_LIBRARIES})
target_link_libraries(two_views/sfm ${OpenCV_LIBS} ${Eigen3_LIBRARIES})
target_link_libraries(two_views/root_finding_test ${OpenCV_LIBS} ${Eigen3_LIBRARIES})
