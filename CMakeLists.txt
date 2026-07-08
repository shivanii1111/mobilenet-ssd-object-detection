cmake_minimum_required(VERSION 3.10)
project(mobilenet_ssd_detect CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release build by default for realistic latency numbers
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED)

add_executable(mobilenet_ssd_detect src/main.cpp)

target_include_directories(mobilenet_ssd_detect PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(mobilenet_ssd_detect PRIVATE ${OpenCV_LIBS})
