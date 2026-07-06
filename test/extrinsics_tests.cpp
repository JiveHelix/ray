#include <catch2/catch.hpp>
#include <tau/eigen.h>
#include <tau/literals.h>

#include <ray/pose.h>
#include <ray/intrinsics.h>
#include <ray/projection.h>

using namespace tau::literals;


using Pixel = tau::Point2d<float>;


TEST_CASE("Pose origin projection", "[pose]")
{
    ray::Intrinsics<float> intrinsics{{
        10_f,
        25_f,
        25_f,
        1920_f / 2_f,
        1080_f / 2_f,
        0}};

    // At origin, no rotation.
    // Camera is is facing the x-axis
    auto originPose = ray::Pose<float>();

    auto projectionFromWorld =
        ray::Projection<float>(intrinsics, originPose);

    auto pixel = Pixel{1920_f / 2_f, 1080_f / 2_f};
    auto lineFromWorld = projectionFromWorld.GetLine_m(pixel);
    auto expected = tau::Line3d<float>({0_f, 0_f, 0_f}, {1_f, 0_f, 0_f});

    REQUIRE(lineFromWorld.IsColinear(expected));

    auto lowerPixel = Pixel{1920_f / 2_f, 1079};

    std::cout << "lowerPixel: "
        << projectionFromWorld.GetLine(lowerPixel).GetAngleAboutY()
        << std::endl;

    auto upperPixel = Pixel{1920_f / 2_f, 0};

    std::cout << "upperPixel: "
        << projectionFromWorld.GetLine(upperPixel).GetAngleAboutY()
        << std::endl;

    auto leftPixel = Pixel{0_f, 1080_f / 2_f};

    std::cout << "leftPixel: "
        << projectionFromWorld.GetLine_m(leftPixel).GetAngleAboutZ()
        << std::endl;

    auto rightPixel = Pixel{1919_f, 1080_f / 2_f};

    std::cout << "rightPixel: "
        << projectionFromWorld.GetLine_m(rightPixel).GetAngleAboutZ()
        << std::endl;

    // TODO Add a REQUIRE
}


TEST_CASE("Pose shifted projection", "[pose]")
{
    ray::Intrinsics<float> intrinsics{{
        10_f,
        25_f,
        25_f,
        1920_f / 2_f,
        1080_f / 2_f,
        0_f}};

    // At origin, no rotation.
    // Camera is is facing the x-axis
    auto pose = ray::Pose<float>();
    pose.point_m.y = 2_f;

    auto projection = ray::Projection<float>(intrinsics, pose);

    auto pixel = Pixel{1920_f / 2_f, 1080_f / 2_f};
    auto lineFromPose = projection.GetLine_m(pixel);
    auto expected = tau::Line3d<float>({0_f, 2_f, 0_f}, {1_f, 0_f, 0_f});

    REQUIRE(lineFromPose.IsColinear(expected));
}


TEST_CASE("Pose rotated projection", "[pose]")
{
    ray::Intrinsics<float> intrinsics{{
        10_f,
        25_f,
        25_f,
        1920_f / 2_f,
        1080_f / 2_f,
        0}};

    auto pose = ray::Pose<float>(
        {45_f, 0_f, 0_f},
        {});

    auto projection = ray::Projection<float>(intrinsics, pose);

    auto pixel = Pixel{1920_f / 2_f, 1080_f / 2_f};
    auto lineFromPose = projection.GetLine(pixel);
    auto expectedAngle = 45_f;

    REQUIRE(lineFromPose.GetAngleAboutZ() == Approx(expectedAngle));
}
