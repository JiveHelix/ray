#include <catch2/catch.hpp>
#include <ray/normalize_pixel.h>


static constexpr double testTolerance = 1e-4;


TEST_CASE("Normalized center", "[homography]")
{
    auto sensorSize = tau::Size<double>{1920, 1080};
    auto normalize = ray::NormalizePixel(sensorSize);

    auto center = tau::Point2d<double>{1920.0 / 2, 1080 / 2};
    auto normalized = normalize.ToNormalized(center);
    auto expected = tau::Point2d<double>{0.0, 0.0};

    REQUIRE(jive::Roughly(expected.x, testTolerance) == normalized.x);
    REQUIRE(jive::Roughly(expected.y, testTolerance) == normalized.y);

    auto roundTrip = normalize.ToPixel(normalized);
    REQUIRE(jive::Roughly(roundTrip.x) == center.x);
    REQUIRE(jive::Roughly(roundTrip.y) == center.y);
}


TEST_CASE("Normalized topLeft", "[homography]")
{
    auto sensorSize = tau::Size<double>{1920, 1080};
    auto normalize = ray::NormalizePixel(sensorSize);

    auto topLeft = tau::Point2d<double>{0, 0};
    auto normalized = normalize.ToNormalized(topLeft);
    auto expected = tau::Point2d<double>{-1.0, -1.0};

    REQUIRE(jive::Roughly(expected.x, testTolerance) == normalized.x);
    REQUIRE(jive::Roughly(expected.y, testTolerance) == normalized.y);

    auto roundTrip = normalize.ToPixel(normalized);
    REQUIRE(jive::Roughly(roundTrip.x) == topLeft.x);
    REQUIRE(jive::Roughly(roundTrip.y) == topLeft.y);
}


TEST_CASE("Normalized topRight", "[homography]")
{
    auto sensorSize = tau::Size<double>{1920, 1080};
    auto normalize = ray::NormalizePixel(sensorSize);

    auto topRight = tau::Point2d<double>{1919, 0};
    auto normalized = normalize.ToNormalized(topRight);
    auto expected = tau::Point2d<double>{0.9990, -1.0};

    REQUIRE(jive::Roughly(expected.x, testTolerance) == normalized.x);
    REQUIRE(jive::Roughly(expected.y, testTolerance) == normalized.y);

    auto roundTrip = normalize.ToPixel(normalized);
    REQUIRE(jive::Roughly(roundTrip.x) == topRight.x);
    REQUIRE(jive::Roughly(roundTrip.y) == topRight.y);
}


TEST_CASE("Normalized bottomLeft", "[homography]")
{
    auto sensorSize = tau::Size<double>{1920, 1080};
    auto normalize = ray::NormalizePixel(sensorSize);

    auto bottomLeft = tau::Point2d<double>{0, 1079};
    auto normalized = normalize.ToNormalized(bottomLeft);
    auto expected = tau::Point2d<double>{-1.0, 0.9981};

    REQUIRE(jive::Roughly(expected.x, testTolerance) == normalized.x);
    REQUIRE(jive::Roughly(expected.y, testTolerance) == normalized.y);

    auto roundTrip = normalize.ToPixel(normalized);
    REQUIRE(jive::Roughly(roundTrip.x) == bottomLeft.x);
    REQUIRE(jive::Roughly(roundTrip.y) == bottomLeft.y);
}

TEST_CASE("Normalized bottomRight", "[homography]")
{
    auto sensorSize = tau::Size<double>{1920, 1080};
    auto normalize = ray::NormalizePixel(sensorSize);

    auto bottomRight = tau::Point2d<double>{1919, 1079};
    auto normalized = normalize.ToNormalized(bottomRight);
    auto expected = tau::Point2d<double>{0.9990, 0.9981};

    REQUIRE(jive::Roughly(expected.x, testTolerance) == normalized.x);
    REQUIRE(jive::Roughly(expected.y, testTolerance) == normalized.y);

    auto roundTrip = normalize.ToPixel(normalized);
    REQUIRE(jive::Roughly(roundTrip.x) == bottomRight.x);
    REQUIRE(jive::Roughly(roundTrip.y) == bottomRight.y);
}


