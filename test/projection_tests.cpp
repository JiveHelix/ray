#include <catch2/catch.hpp>

#include <tau/literals.h>
#include <ray/projection.h>


using namespace tau::literals;


TEST_CASE("Image to World to Image round trip.", "[projection]")
{
    auto intrinsics = ray::Intrinsics<float>{};
    auto opticalCenter = tau::Point3d<float>(1_f, -3_f, 2_f);

    ray::Pose<float> pose(
        {
            -10_f, // yaw --> rotation about z
            13.0_f, // pitch --> rotation about y
            0.0_f, // roll --> rotation about x
            tau::AxisOrder{2, 1, 0}}, // rotation about x
        opticalCenter);

    ray::Projection projection(intrinsics, pose);
    auto pixel = tau::Point2d<float>{1547_f, 567_f};
    auto world = projection.GetLine_m(pixel).ScaleToPoint(5);
    auto roundTrip = projection.WorldToImage(world);

    REQUIRE(roundTrip.isApprox(pixel.GetHomogeneous()));
}


TEST_CASE("Image to world direction check", "[projection]")
{
    auto intrinsics = ray::Intrinsics<float>{};
    auto opticalCenter = tau::Point3d<float>(-1_f, 6_f, 4_f);

    ray::Pose<float> pose(
        {
            30_f, // yaw -->rotation about z
            -20.0_f, // pitch --> rotation about y
            -59.3_f, // roll --> rotation about x
            tau::AxisOrder{2, 1, 0}}, // rotation about x
        opticalCenter);

    ray::Projection projection(intrinsics, pose);
    auto pixel0 = tau::Point2d<float>{0_f, 200_f};
    auto pixel1 = tau::Point2d<float>{1919_f, 800_f};
    auto line0 = projection.GetLine_m(pixel0);
    auto line1 = projection.GetLine_m(pixel1);
    auto point0 = line0.ScaleToPoint(5);
    auto point1 = line1.ScaleToPoint(5);

    auto checkDirection0 = (point0 - opticalCenter).ToEigen().normalized();
    auto checkDirection1 = (point1 - opticalCenter).ToEigen().normalized();

    REQUIRE(line0.point == opticalCenter);

    if (!checkDirection0.isApprox(line0.direction))
    {
        std::cout << "checkDirection0: " << checkDirection0 << std::endl;
        std::cout << "line0.direction: " << line0.direction << std::endl;
    }

    REQUIRE(checkDirection0.isApprox(line0.direction));
    REQUIRE(checkDirection1.isApprox(line1.direction));

    auto checkPixel0 = projection.WorldToImage(point0);
    auto checkPixel1 = projection.WorldToImage(point1);

    if (!checkPixel0.isApprox(pixel0.GetHomogeneous()))
    {
        std::cout << "checkPixel0: " << checkPixel0 << std::endl;
        std::cout << "pixel0: " << pixel0 << std::endl;
    }

    if (!checkPixel1.isApprox(pixel1.GetHomogeneous()))
    {
        std::cout << "checkPixel1: " << checkPixel1 << std::endl;
        std::cout << "pixel1: " << pixel1 << std::endl;
    }

    REQUIRE(checkPixel0.isApprox(pixel0.GetHomogeneous()));
    REQUIRE(checkPixel1.isApprox(pixel1.GetHomogeneous()));
}


TEST_CASE("Simple image to world", "[projection]")
{
    auto intrinsics = ray::Intrinsics<float>{};
    auto opticalCenter = tau::Point3d<float>(0_f, 0_f, 0_f);

    ray::Pose<float> pose(
        {
            0_f, // yaw -->rotation about z
            0_f, // pitch --> rotation about y
            0_f, // roll --> rotation about x
            tau::AxisOrder{2, 1, 0}}, // rotation about x
        opticalCenter);

    ray::Projection projection(intrinsics, pose);
    auto pixel0 = tau::Point2d<float>{960_f, 0_f};
    auto pixel1 = tau::Point2d<float>{960_f, 1079_f};
    auto pixel2 = tau::Point2d<float>{0_f, 540_f};
    auto pixel3 = tau::Point2d<float>{1919_f, 540_f};

    auto line0 = projection.GetLine_m(pixel0);
    auto line1 = projection.GetLine_m(pixel1);
    auto line2 = projection.GetLine_m(pixel2);
    auto line3 = projection.GetLine_m(pixel3);

    auto point0 = line0.ScaleToPoint(5);
    auto point1 = line1.ScaleToPoint(5);
    auto point2 = line2.ScaleToPoint(5);
    auto point3 = line3.ScaleToPoint(5);

    // All projected points should be in front of the camera at positive x.
    REQUIRE(point0.x > 1_f);
    REQUIRE(point1.x > 1_f);
    REQUIRE(point2.x > 1_f);
    REQUIRE(point3.x > 1_f);

    // Points 0 and 1 should be above and below the x-y plane, respectively.
    REQUIRE(point0.z > 0);
    REQUIRE(point1.z < 0);

    // Points 2 and 3 should be left and right of the z-x plane, respectively.
    REQUIRE(point2.y > 0);
    REQUIRE(point3.y < 0);
}


TEST_CASE("Trivial world to Camera", "[projection]")
{
    auto intrinsics = ray::Intrinsics<float>{};
    auto pose = ray::Pose<float>{};

    tau::Vector3<float> world(5.0_f, 0.0_f, 0.0_f);
    ray::Projection projection(intrinsics, pose);
    tau::Vector3<float> pixel = projection.WorldToImage(world);

    REQUIRE(pixel(0) == Approx(960.0_f));
    REQUIRE(pixel(1) == Approx(540.0_f));
    REQUIRE(pixel(2) == Approx(1.0_f));
}


TEST_CASE("Shifted world to Camera", "[projection]")
{
    auto intrinsics = ray::Intrinsics<float>{};

    auto noShift = ray::Pose<float>{};

    auto pose = ray::Pose<float>{};
    pose.point_m.y = 1.0_f;

    tau::Vector3<float> world(5.0_f, 0.0_f, 0.0_f);
    ray::Projection noShiftProjection(intrinsics, noShift);
    ray::Projection projection(intrinsics, pose);

    tau::Vector3<float> noShiftPixel =
        noShiftProjection.WorldToImage(world);

    tau::Vector3<float> pixel = projection.WorldToImage(world);

    std::cout << "Shifted world to camera: " << pixel << std::endl;
    std::cout << "No Shift world to camera: " << noShiftPixel << std::endl;

    // TODO: Add a REQUIRE
}
