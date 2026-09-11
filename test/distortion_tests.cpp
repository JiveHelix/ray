#include <catch2/catch.hpp>
#include <iostream>
#include <fields/serialize.h>
#include <tau/random.h>
#include <ray/distortion.h>
#include <ray/intrinsics.h>


template<typename T>
bool IsSaneRadialDistortion(const ray::distortion::BrownConrady<T> &d, T rMax)
{
    constexpr int sampleCount = 64;

    for (int i = 0; i <= sampleCount; ++i)
    {
        T r = rMax * T(i) / T(sampleCount);
        T r2 = r * r;
        T r4 = r2 * r2;
        T r6 = r4 * r2;

        T scale =
            T(1)
            + d.k1 * r2
            + d.k2 * r4
            + d.k3 * r6;

        if (!(scale > T(0.25)))
        {
            return false;
        }

        T radialDerivative =
            T(1)
            + T(3) * d.k1 * r2
            + T(5) * d.k2 * r4
            + T(7) * d.k3 * r6;

        if (!(radialDerivative > T(0.10)))
        {
            return false;
        }
    }

    return true;
}


template<typename T>
ray::distortion::BrownConrady<T>
GenerateDistortion(tau::UniformRandom<T> &uniformRandom, T rMax)
{
    T rMax2 = rMax * rMax;
    T rMax4 = rMax2 * rMax2;
    T rMax6 = rMax4 * rMax2;

    while (true)
    {
        T c1 = uniformRandom.SetRange(T(-0.60), T(0.40))();
        T c2 = uniformRandom.SetRange(T(-0.25), T(0.25))();
        T c3 = uniformRandom.SetRange(T(-0.08), T(0.08))();

        ray::distortion::BrownConrady<T> d;

        d.k1 = c1 / rMax2;
        d.k2 = c2 / rMax4;
        d.k3 = c3 / rMax6;

        d.p1 = uniformRandom.SetRange(T(-0.01), T(0.01))() / rMax2;
        d.p2 = uniformRandom.SetRange(T(-0.01), T(0.01))() / rMax2;

        if (IsSaneRadialDistortion(d, rMax))
        {
            return d;
        }
    }
}


template<typename T>
void RunRoundTripTest(
    const ray::IntrinsicsAsPixels<T> &intrinsics_pixels,
    const ray::distortion::BrownConrady<T> &distortion,
    const tau::Point2d<T> &point)
{
    auto normalizedPoint = intrinsics_pixels.ToNormalizedPixel(point);

    auto normalizedDistorted = distortion.Apply(normalizedPoint);
    auto distorted = intrinsics_pixels.ToSensorPixel(normalizedDistorted);

    auto result = distortion.Undo(
        intrinsics_pixels.ToNormalizedPixel(distorted));

    REQUIRE(!result.singularJacobian);
    REQUIRE(result.converged);

    auto resultPoint_pixels = intrinsics_pixels.ToSensorPixel(result.point);

    REQUIRE(std::abs(resultPoint_pixels.x - point.x) <= 0.002);
    REQUIRE(std::abs(resultPoint_pixels.y - point.y) <= 0.002);
}


TEST_CASE("Distort/Undistort round trip succeeds", "[distortion]")
{
    auto seed = GENERATE(
        take(10, random(0u, std::numeric_limits<unsigned int>::max())));

    tau::UniformRandom<double> uniformRandom(seed, -1, 1);

    auto distortion = GenerateDistortion(uniformRandom, 1.7);

    ray::Intrinsics<double> intrinsics{};

    intrinsics.focalLengthX_mm = 8.0;
    intrinsics.focalLengthY_mm = 8.0;

    auto intrinsics_pixels = intrinsics.GetAsPixels();

    RunRoundTripTest(
        intrinsics_pixels,
        distortion,
        tau::Point2d<double>(0, 0));

    RunRoundTripTest(
        intrinsics_pixels,
        distortion,
        tau::Point2d<double>(1919, 0));

    RunRoundTripTest(
        intrinsics_pixels,
        distortion,
        tau::Point2d<double>(1919, 1079));

    RunRoundTripTest(
        intrinsics_pixels,
        distortion,
        tau::Point2d<double>(0, 1079));

    RunRoundTripTest(
        intrinsics_pixels,
        distortion,
        tau::Point2d<double>(960, 540));

    uniformRandom.SetRange(0, 900);

    for (size_t i = 0; i < 100; ++i)
    {
        RunRoundTripTest(
            intrinsics_pixels,
            distortion,
            tau::Point2d<double>(uniformRandom(), uniformRandom()));
    }
}


TEST_CASE("Distortion with numeric direction in json", "[distortion]")
{
    static constexpr auto testString = R"(
        {
            "direction": 1,
            "k1": 0.17033830279190668,
            "k2": -0.06492882857243383,
            "k3": -0.03854759207230837,
            "p1": -0.0010766094881574997,
            "p2": 0.000791311604773549
        }
    )";

    auto distortion =
        fields::FromJson<ray::distortion::BrownConrady<double>>(testString);

    REQUIRE(distortion.direction == ray::distortion::Direction::inverse);
}


TEST_CASE("Distortion with string direction in json", "[distortion]")
{
    static constexpr auto testString = R"(
        {
            "direction": "forward",
            "k1": 0.17033830279190668,
            "k2": -0.06492882857243383,
            "k3": -0.03854759207230837,
            "p1": -0.0010766094881574997,
            "p2": 0.000791311604773549
        }
    )";

    auto distortion =
        fields::FromJson<ray::distortion::BrownConrady<double>>(testString);

    REQUIRE(distortion.direction == ray::distortion::Direction::forward);
}
