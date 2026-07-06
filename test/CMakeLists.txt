add_catch2_test(
    NAME ray_tests
    SOURCES
        distortion_tests.cpp
        extrinsics_tests.cpp
        homography_tests.cpp
        intrinsics_tests.cpp
        projection_tests.cpp
    LINK
        ray)
