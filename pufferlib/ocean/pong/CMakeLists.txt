cmake_minimum_required(VERSION 3.14)

project(pong)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(RELEASE)
  add_compile_definitions(RELEASE=1)
  set(CMAKE_BUILD_TYPE Release)
  if(NOT WIN32)
    add_compile_options(-O3 -DNDEBUG)
  endif()  
endif()

file(GLOB SOURCES "*.cpp")
if(MSVC)
  add_compile_options(/wd4576)
endif()
if(DEBUG)
  add_compile_definitions(DEBUG=1)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/NumCpp/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../extensions)
add_executable(pong ${SOURCES} ${HEADERS})
add_compile_definitions(NUMCPP_NO_USE_BOOST=1)

target_include_directories(pong PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Fetch raylib dependency
include(FetchContent)
set(RAYLIB_VERSION 5.0)

FetchContent_Declare(
    raylib
    DOWNLOAD_EXTRACT_TIMESTAMP OFF
    URL https://github.com/raysan5/raylib/archive/refs/tags/${RAYLIB_VERSION}.tar.gz
    FIND_PACKAGE_ARGS
)

FetchContent_MakeAvailable(raylib)
target_link_libraries(pong raylib)