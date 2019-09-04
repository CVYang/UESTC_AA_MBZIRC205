find_package(OpenCV 3 REQUIRED)

add_compile_options(-std=c++11)
aux_source_directory(. DIR_SRCS)
add_executable(main ${DIR_SRCS})
target_link_libraries(main  ${OpenCV_LIBS} librealsense2.so pthread)

