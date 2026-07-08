#include <catch2/catch.hpp>
#include <ray/normalize_pixel.h>


static constexpr double testTolerance = 1e-4;


TEST_CASE("Normalized center", "[homography]")
{
    auto sensorSize = tau::Size<double>{1920, 1080};
    auto normalize = ray::NormalizePixel(sensorSize);

    auto center = tau::Point2d<double>{1920.0 / 2, 1080 / 2};
    auto normalized = normalize.ToNormalized(center);
    auto maximumRadius = 1.0;

    REQUIRE(normalized.Magnitude() <= maximumRadius);

    auto roundTrip = normalize.ToPixels(normalized);

    REQUIRE(jive::Roughly(roundTrip.x, testTolerance) == center.x);
    REQUIRE(jive::Roughly(roundTrip.y, testTolerance) == center.y);
}


TEST_CASE("Normalized topLeft", "[homography]")
{
    auto sensorSize = tau::Size<double>{1920, 1080};
    auto normalize = ray::NormalizePixel(sensorSize);

    auto topLeft = tau::Point2d<double>{0, 0};
    auto normalized = normalize.ToNormalized(topLeft);
    auto maximumRadius = 1.0;

    REQUIRE(normalized.Magnitude() <= maximumRadius);

    auto roundTrip = normalize.ToPixels(normalized);

    REQUIRE(jive::Roughly(roundTrip.x, testTolerance) == topLeft.x);
    REQUIRE(jive::Roughly(roundTrip.y, testTolerance) == topLeft.y);
}


TEST_CASE("Normalized topRight", "[homography]")
{
    auto sensorSize = tau::Size<double>{1920, 1080};
    auto normalize = ray::NormalizePixel(sensorSize);

    auto topRight = tau::Point2d<double>{1919, 0};
    auto normalized = normalize.ToNormalized(topRight);
    auto maximumRadius = 1.0;

    REQUIRE(normalized.Magnitude() <= maximumRadius);

    auto roundTrip = normalize.ToPixels(normalized);

    REQUIRE(jive::Roughly(roundTrip.x, testTolerance) == topRight.x);
    REQUIRE(jive::Roughly(roundTrip.y, testTolerance) == topRight.y);
}


TEST_CASE("Normalized bottomLeft", "[homography]")
{
    auto sensorSize = tau::Size<double>{1920, 1080};
    auto normalize = ray::NormalizePixel(sensorSize);

    auto bottomLeft = tau::Point2d<double>{0, 1079};
    auto normalized = normalize.ToNormalized(bottomLeft);
    auto maximumRadius = 1.0;

    REQUIRE(normalized.Magnitude() <= maximumRadius);

    auto roundTrip = normalize.ToPixels(normalized);

    REQUIRE(jive::Roughly(roundTrip.x, testTolerance) == bottomLeft.x);
    REQUIRE(jive::Roughly(roundTrip.y, testTolerance) == bottomLeft.y);
}

TEST_CASE("Normalized bottomRight", "[homography]")
{
    auto sensorSize = tau::Size<double>{1920, 1080};
    auto normalize = ray::NormalizePixel(sensorSize);

    auto bottomRight = tau::Point2d<double>{1919, 1079};
    auto normalized = normalize.ToNormalized(bottomRight);
    auto maximumRadius = 1.0;

    REQUIRE(normalized.Magnitude() <= maximumRadius);

    auto roundTrip = normalize.ToPixels(normalized);

    REQUIRE(jive::Roughly(roundTrip.x, testTolerance) == bottomRight.x);
    REQUIRE(jive::Roughly(roundTrip.y, testTolerance) == bottomRight.y);
}


