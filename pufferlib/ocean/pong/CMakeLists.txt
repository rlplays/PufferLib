cmake_minimum_required(VERSION 3.10)

project(pong)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)



file(GLOB SOURCES "*.cpp" "*.c")
file(GLOB HEADERS "*.hpp" "*.h")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/NumCpp/include)
add_executable(pong ${SOURCES} ${HEADERS})


target_include_directories(pong PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})