# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /root/Desktop/opencv/MBZIRC/taskplanning

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /root/Desktop/opencv/MBZIRC/taskplanning

# Include any dependencies generated for this target.
include src/CMakeFiles/main.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/main.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/main.dir/flags.make

src/CMakeFiles/main.dir/main.cpp.o: src/CMakeFiles/main.dir/flags.make
src/CMakeFiles/main.dir/main.cpp.o: src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/Desktop/opencv/MBZIRC/taskplanning/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/main.dir/main.cpp.o"
	cd /root/Desktop/opencv/MBZIRC/taskplanning/src && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/main.dir/main.cpp.o -c /root/Desktop/opencv/MBZIRC/taskplanning/src/main.cpp

src/CMakeFiles/main.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/main.cpp.i"
	cd /root/Desktop/opencv/MBZIRC/taskplanning/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/Desktop/opencv/MBZIRC/taskplanning/src/main.cpp > CMakeFiles/main.dir/main.cpp.i

src/CMakeFiles/main.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/main.cpp.s"
	cd /root/Desktop/opencv/MBZIRC/taskplanning/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/Desktop/opencv/MBZIRC/taskplanning/src/main.cpp -o CMakeFiles/main.dir/main.cpp.s

src/CMakeFiles/main.dir/main.cpp.o.requires:

.PHONY : src/CMakeFiles/main.dir/main.cpp.o.requires

src/CMakeFiles/main.dir/main.cpp.o.provides: src/CMakeFiles/main.dir/main.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/main.dir/build.make src/CMakeFiles/main.dir/main.cpp.o.provides.build
.PHONY : src/CMakeFiles/main.dir/main.cpp.o.provides

src/CMakeFiles/main.dir/main.cpp.o.provides.build: src/CMakeFiles/main.dir/main.cpp.o


src/CMakeFiles/main.dir/detect.cpp.o: src/CMakeFiles/main.dir/flags.make
src/CMakeFiles/main.dir/detect.cpp.o: src/detect.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/Desktop/opencv/MBZIRC/taskplanning/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/CMakeFiles/main.dir/detect.cpp.o"
	cd /root/Desktop/opencv/MBZIRC/taskplanning/src && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/main.dir/detect.cpp.o -c /root/Desktop/opencv/MBZIRC/taskplanning/src/detect.cpp

src/CMakeFiles/main.dir/detect.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/detect.cpp.i"
	cd /root/Desktop/opencv/MBZIRC/taskplanning/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/Desktop/opencv/MBZIRC/taskplanning/src/detect.cpp > CMakeFiles/main.dir/detect.cpp.i

src/CMakeFiles/main.dir/detect.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/detect.cpp.s"
	cd /root/Desktop/opencv/MBZIRC/taskplanning/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/Desktop/opencv/MBZIRC/taskplanning/src/detect.cpp -o CMakeFiles/main.dir/detect.cpp.s

src/CMakeFiles/main.dir/detect.cpp.o.requires:

.PHONY : src/CMakeFiles/main.dir/detect.cpp.o.requires

src/CMakeFiles/main.dir/detect.cpp.o.provides: src/CMakeFiles/main.dir/detect.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/main.dir/build.make src/CMakeFiles/main.dir/detect.cpp.o.provides.build
.PHONY : src/CMakeFiles/main.dir/detect.cpp.o.provides

src/CMakeFiles/main.dir/detect.cpp.o.provides.build: src/CMakeFiles/main.dir/detect.cpp.o


# Object files for target main
main_OBJECTS = \
"CMakeFiles/main.dir/main.cpp.o" \
"CMakeFiles/main.dir/detect.cpp.o"

# External object files for target main
main_EXTERNAL_OBJECTS =

src/main: src/CMakeFiles/main.dir/main.cpp.o
src/main: src/CMakeFiles/main.dir/detect.cpp.o
src/main: src/CMakeFiles/main.dir/build.make
src/main: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_stitching3.so.3.3.1
src/main: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_superres3.so.3.3.1
src/main: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_videostab3.so.3.3.1
src/main: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_aruco3.so.3.3.1
src/main: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_bgsegm3.so.3.3.1
src/main: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_bioinspired3.so.3.3.1
src/main: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_ccalib3.so.3.3.1
src/main: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_cvv3.so.3.3.1
src/main: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_dpm3.so.3.3.1
src/main: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_face3.so.3.3.1
src/main: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_fuzzy3.so.3.3.1
src/main: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_hdf3.so.3.3.1
src/main: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_img_hash3.so.3.3.1
src/main: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_line_descriptor3.so.3.3.1
src/main: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_optflow3.so.3.3.1
src/main: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_reg3.so.3.3.1
src/main: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_rgbd3.so.3.3.1
src/main: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_saliency3.so.3.3.1
src/main: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_stereo3.so.3.3.1
src/main: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_structured_light3.so.3.3.1
src/main: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_surface_matching3.so.3.3.1
src/main: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_tracking3.so.3.3.1
src/main: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_xfeatures2d3.so.3.3.1
src/main: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_ximgproc3.so.3.3.1
src/main: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_xobjdetect3.so.3.3.1
src/main: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_xphoto3.so.3.3.1
src/main: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_shape3.so.3.3.1
src/main: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_photo3.so.3.3.1
src/main: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_datasets3.so.3.3.1
src/main: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_plot3.so.3.3.1
src/main: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_text3.so.3.3.1
src/main: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_dnn3.so.3.3.1
src/main: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_ml3.so.3.3.1
src/main: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_video3.so.3.3.1
src/main: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_calib3d3.so.3.3.1
src/main: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_features2d3.so.3.3.1
src/main: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_highgui3.so.3.3.1
src/main: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_videoio3.so.3.3.1
src/main: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_viz3.so.3.3.1
src/main: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_phase_unwrapping3.so.3.3.1
src/main: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_flann3.so.3.3.1
src/main: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_imgcodecs3.so.3.3.1
src/main: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_objdetect3.so.3.3.1
src/main: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_imgproc3.so.3.3.1
src/main: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_core3.so.3.3.1
src/main: src/CMakeFiles/main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/Desktop/opencv/MBZIRC/taskplanning/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable main"
	cd /root/Desktop/opencv/MBZIRC/taskplanning/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/main.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/main.dir/build: src/main

.PHONY : src/CMakeFiles/main.dir/build

src/CMakeFiles/main.dir/requires: src/CMakeFiles/main.dir/main.cpp.o.requires
src/CMakeFiles/main.dir/requires: src/CMakeFiles/main.dir/detect.cpp.o.requires

.PHONY : src/CMakeFiles/main.dir/requires

src/CMakeFiles/main.dir/clean:
	cd /root/Desktop/opencv/MBZIRC/taskplanning/src && $(CMAKE_COMMAND) -P CMakeFiles/main.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/main.dir/clean

src/CMakeFiles/main.dir/depend:
	cd /root/Desktop/opencv/MBZIRC/taskplanning && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/Desktop/opencv/MBZIRC/taskplanning /root/Desktop/opencv/MBZIRC/taskplanning/src /root/Desktop/opencv/MBZIRC/taskplanning /root/Desktop/opencv/MBZIRC/taskplanning/src /root/Desktop/opencv/MBZIRC/taskplanning/src/CMakeFiles/main.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/main.dir/depend

