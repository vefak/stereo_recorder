cmake_minimum_required(VERSION 2.8.3)
project(stereocam_recorder)

set(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_FLAGS "-O0")
SET(CMAKE_C_FLAGS "-O0")

#use: -std=c++14
#set (CMAKE_CXX_STANDARD 11)
#set (CMAKE_CXX_STANDARD_REQUIRED ON)
#set (CMAKE_CXX_EXTENSIONS OFF)



find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  OpenCV REQUIRED
)

include_directories(
  include
  ${PROJECT_SOURCE_DIR}/include/
  ${catkin_INCLUDE_DIRS}
  "/usr/include/flycapture"
  "/usr/include/triclops4"
  "/usr/lib/x86_64-linux-gnu/triclops4" 
  ${OpenCV_INCLUDE_DIRS}
)
link_directories( "/usr/lib/x86_64-linux-gnu/triclops4" )

catkin_package(LIBRARIES ros_${PROJECT_NAME}
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp rospy
)

add_library(ros_${PROJECT_NAME}
  src/stereocore_rt.cpp
)

file (GLOB flycapture_lib_files "/usr/lib/libfly*.*")
file (GLOB triclops_lib_files "/usr/lib/x86_64-linux-gnu/triclops4/*.*")



add_executable(stereo_recorder_node src/stereo_alpha_rt.cpp)
target_link_libraries(stereo_recorder_node ros_${PROJECT_NAME} 
                                      ${catkin_LIBRARIES} 
                                      ${OpenCV_LIBRARIES} 
                                      ${flycapture_lib_files} 
                                      ${triclops_lib_files})

add_dependencies(stereo_recorder_node stereocam_recorder_generate_messages_cpp)



catkin_package(
  INCLUDE_DIRS include
)



