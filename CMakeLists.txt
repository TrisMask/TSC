cmake_minimum_required(VERSION 3.22)

project(dbslcd LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(PCL 1.10 REQUIRED COMPONENTS common io visualization filters)
find_package(OpenCV 4 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS} ${OpenCV_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS} ${OpenCV_DEFINITIONS})

add_executable(m main.cpp kitti.cc pcd2.cc)
target_link_libraries(m ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(sc sc.cpp pcd2.cc kitti.cc)
target_link_libraries(sc ${PCL_LIBRARIES})