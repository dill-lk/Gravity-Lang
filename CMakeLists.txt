cmake_minimum_required(VERSION 3.16)
project(GravityLangCpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

add_executable(gravity cpp/gravity_interpreter.cpp)
target_link_libraries(gravity PRIVATE Threads::Threads)
add_executable(gravityc cpp/gravity_compiler.cpp)

enable_testing()
add_test(NAME gravity_moon_orbit COMMAND gravity run ${CMAKE_SOURCE_DIR}/examples/moon_orbit.gravity)
add_test(NAME gravity_earth_moon COMMAND gravity run ${CMAKE_SOURCE_DIR}/examples/earth_moon.gravity)
add_test(NAME gravity_scientific_loop COMMAND gravity run ${CMAKE_SOURCE_DIR}/examples/scientific_loop.gravity)
add_test(NAME gravity_threaded_three_body COMMAND gravity run ${CMAKE_SOURCE_DIR}/examples/threaded_three_body.gravity)
add_test(NAME gravity_help COMMAND gravity --help)
add_test(NAME gravity_check_moon_orbit COMMAND gravity check ${CMAKE_SOURCE_DIR}/examples/moon_orbit.gravity)
add_test(NAME gravityc_help COMMAND gravityc --help)
