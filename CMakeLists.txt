cmake_minimum_required(VERSION 3.16)
project(GravityLangCpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(gravity cpp/gravity_interpreter.cpp)
add_executable(gravityc cpp/gravity_compiler.cpp)

enable_testing()
add_test(NAME gravity_moon_orbit COMMAND gravity run ${CMAKE_SOURCE_DIR}/examples/moon_orbit.gravity)
add_test(NAME gravity_earth_moon COMMAND gravity run ${CMAKE_SOURCE_DIR}/examples/earth_moon.gravity)
add_test(NAME gravity_scientific_loop COMMAND gravity run ${CMAKE_SOURCE_DIR}/examples/scientific_loop.gravity)
