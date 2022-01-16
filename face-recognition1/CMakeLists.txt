cmake_minimum_required(VERSION 3.14)

set(LIBRARY_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/lib")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/exe")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(OpenCV_DIR "C:/DevTools/OpenCVFull/output")

project (Face-Recognition VERSION 1.0 LANGUAGES CXX)
find_package(OpenCV)

add_executable (face-recognition1 "face-recognition1.cpp")
target_sources(face-recognition1 PRIVATE "face-recognition1.cpp")
target_include_directories(face-recognition1 PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(face-recognition1 PRIVATE ${OpenCV_LIBS})