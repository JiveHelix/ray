#include <catch2/catch.hpp>
#include <tau/eigen_shim.h>


#include <tau/literals.h>
#include <ray/intrinsics.h>


using namespace tau::literals;


TEST_CASE("Meters To Pixel", "[intrinsics]")
{
    using Intrinsics = ray::Intrinsics<double>;
    using PixelConvert = ray::PixelConvert<double>;
    Intrinsics intrinsics{};

    intrinsics.pixelSize_um = 10;

    double metersToPixel =
        intrinsics.pixelSize_um * PixelConvert::metersPerMicron;

    REQUIRE(intrinsics.MetersToPixels(0.0_d) == 0_d);
    REQUIRE(intrinsics.MetersToPixels(2.0_d) == 2.0_d / metersToPixel);
    REQUIRE(intrinsics.MetersToPixels(-2.0_d) == -2.0_d / metersToPixel);
}

TEST_CASE("Pixels To Meters", "[intrinsics]")
{
    using Intrinsics = ray::Intrinsics<double>;
    using PixelConvert = ray::PixelConvert<double>;
    Intrinsics intrinsics{};

    intrinsics.pixelSize_um = 10;

    double metersToPixel =
        intrinsics.pixelSize_um * PixelConvert::metersPerMicron;

    REQUIRE(intrinsics.PixelsToMeters(0.0_d) == 0_d);
    REQUIRE(intrinsics.PixelsToMeters(2.0_d) == 2.0_d * metersToPixel);
    REQUIRE(intrinsics.PixelsToMeters(-2.0_d) == -2.0_d * metersToPixel);
}
