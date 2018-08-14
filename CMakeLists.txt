cmake_minimum_required(VERSION 2.8.9)
project(Prepare_Set)

add_definitions("-std=c++0x -O2")

find_package(OpenCV 3 REQUIRED core imgproc video highgui imgcodecs)

find_package(Boost 1.56 REQUIRED COMPONENTS system filesystem program_options)

set(SOURCES ${PROJECT_SOURCE_DIR}/prepare_set.cpp
            ${PROJECT_SOURCE_DIR}/SegNetSet.cpp)

add_executable(prepare_set ${SOURCES})
target_link_libraries(prepare_set ${OpenCV_LIBS} ${Boost_LIBRARIES})