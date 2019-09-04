# cmake needs this line
cmake_minimum_required(VERSION 2.8)

project(main LANGUAGES C CXX)

add_compile_options(-std=c++11)

INCLUDE_DIRECTORIES(inc)
add_subdirectory(src) 

find_package(OpenCV 3 REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "version: ${OpenCV_VERSION}")
message(STATUS "libraries: ${OpenCV_LIBS}")
message(STATUS "include path: ${OpenCV_INCLUDE_DIRS}")

#target_link_libraries(main ${OpenCV_LIBS} librealsense2.so pthread)