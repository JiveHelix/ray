cmake_minimum_required(VERSION 3.15)

project(ray CXX)

include(${CMAKE_CURRENT_LIST_DIR}/cmake_includes/setup_project.cmake)
setup_project()

add_subdirectory(ray)

include(${CMAKE_CURRENT_LIST_DIR}/cmake_includes/enable_extras.cmake)
enable_extras()
