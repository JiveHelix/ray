add_library(ray)

if (${fPIC})
    set_property(TARGET ray PROPERTY POSITION_INDEPENDENT_CODE ON)
endif ()

FindPackages(Tau)

# Projects that include this project must #include "ray/<header-name>"
target_include_directories(ray PUBLIC ${PROJECT_SOURCE_DIR})

target_link_libraries(
    ray
    PUBLIC
    tau::tau)

target_sources(
    ray
    PRIVATE
    camera_parameters.cpp
    homography.cpp
    homography_settings.cpp
    named_vertex.cpp
    pose.cpp)


install(TARGETS ray DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(
    DIRECTORY ${PROJECT_SOURCE_DIR}/ray
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
