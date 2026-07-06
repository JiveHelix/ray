#pragma once


#include <pex/group.h>
#include <ray/intrinsics.h>


namespace ray
{


namespace distortion
{


template<typename T>
struct BrownConradyFields
{
    static constexpr auto fields = std::make_tuple(
        fields::Field(&T::k1, "k1"),
        fields::Field(&T::k2, "k2"),
        fields::Field(&T::p1, "p1"),
        fields::Field(&T::p2, "p2"),
        fields::Field(&T::k3, "k3"));
};


template<typename Float>
struct BrownConradyTemplate
{
    template<template<typename> typename T>
    struct Template
    {
        T<Float> k1;
        T<Float> k2;
        T<Float> p1;
        T<Float> p2;
        T<Float> k3;

        static constexpr auto fields =
            BrownConradyFields<Template>::fields;

        static constexpr auto fieldsTypeName = "BrownConrady";
    };
};


template<typename Float>
using BrownConradyBase =
    BrownConradyTemplate<Float>::template Template<pex::Identity>;


template<typename Float>
struct BrownConrady: public BrownConradyBase<Float>
{
    template<typename U, typename Style = tau::Round>
    BrownConrady<U> Cast() const
    {
        return tau::CastFields<BrownConrady<U>, U, Style>(*this);
    }
};


template<typename T>
using BrownConradyGroup =
    pex::Group
    <
        BrownConradyFields,
        BrownConradyTemplate<T>::template Template,
        pex::PlainT<BrownConrady<T>>
    >;


template<typename T>
using BrownConradyModel =
    typename BrownConradyGroup<T>::Model;

template<typename T>
using BrownConradyControl =
    typename BrownConradyGroup<T>::DefaultControl;


DECLARE_OUTPUT_STREAM_OPERATOR(BrownConrady<float>)
DECLARE_OUTPUT_STREAM_OPERATOR(BrownConrady<double>)
DECLARE_EQUALITY_OPERATORS(BrownConrady<float>)
DECLARE_EQUALITY_OPERATORS(BrownConrady<double>)


template<typename T>
tau::Point2d<T> DistortPoint(
    const BrownConrady<T> &distortion,
    const tau::Point2d<T> &point)
{
    T xPow2 = point.x * point.x;
    T yPow2 = point.y * point.y;

    T radiusPow2 = xPow2 + yPow2;
    T radiusPow4 = radiusPow2 * radiusPow2;
    T radiusPow6 = radiusPow4 * radiusPow2;

    T radialDistortion =
        T(1)
        + distortion.k1 * radiusPow2
        + distortion.k2 * radiusPow4
        + distortion.k3 * radiusPow6;

    T xy = point.x * point.y;

    tau::Point2d<T> result;

    result.x =
        point.x * radialDistortion
        + T(2) * distortion.p1 * xy
        + distortion.p2 * (radiusPow2 + T(2) * xPow2);

    result.y =
        point.y * radialDistortion
        + distortion.p1 * (radiusPow2 + T(2) * yPow2)
        + T(2) * distortion.p2 * xy;

    return result;
}


template<typename T>
constexpr T GetTolerance(
    const IntrinsicsAsPixels<T> &intrinsics_pixels,
    T pixelTolerance)
{
    T maxFocalLength_pixels =
        std::max(
            intrinsics_pixels.focalLengthX,
            intrinsics_pixels.focalLengthY);

    return pixelTolerance / maxFocalLength_pixels;
}


template<typename T>
std::optional<tau::Point2d<T>> UndistortPoint(
    const IntrinsicsAsPixels<T> &intrinsics_pixels,
    const BrownConrady<T> &distortion,
    const tau::Point2d<T> &distortedPoint,
    int maxIterations = 10,
    T pixelTolerance = T(1e-3))
{
    const T tolerance = GetTolerance(intrinsics_pixels, pixelTolerance);
    const T toleranceSquared = tolerance * tolerance;
    auto normalizedPoint = intrinsics_pixels.ToNormalizedPixel(distortedPoint);
    T xd = normalizedPoint.x;
    T yd = normalizedPoint.y;

    T x = xd;
    T y = yd;

    for (int i = 0; i < maxIterations; ++i)
    {
        T xSquared = x * x;
        T ySquared = y * y;
        T xy = x * y;

        T rSquared = xSquared + ySquared;
        T rPower4 = rSquared * rSquared;
        T rPower6 = rPower4 * rSquared;

        T radial =
            T(1)
            + distortion.k1 * rSquared
            + distortion.k2 * rPower4
            + distortion.k3 * rPower6;

        T residualX =
            x * radial
            + T(2) * distortion.p1 * xy
            + distortion.p2 * (rSquared + T(2) * xSquared)
            - xd;

        T residualY =
            y * radial
            + distortion.p1 * (rSquared + T(2) * ySquared)
            + T(2) * distortion.p2 * xy
            - yd;

        T residualSquared = residualX * residualX + residualY * residualY;

        if (residualSquared < toleranceSquared)
        {
            break;
        }

        T radialDerivative =
            distortion.k1
            + T(2) * distortion.k2 * rSquared
            + T(3) * distortion.k3 * rPower4;

        T common = T(2) * xy * radialDerivative;

        T j00 =
            radial
            + T(2) * xSquared * radialDerivative
            + T(2) * distortion.p1 * y
            + T(6) * distortion.p2 * x;

        T j01 =
            common
            + T(2) * distortion.p1 * x
            + T(2) * distortion.p2 * y;

        T j10 =
            common
            + T(2) * distortion.p1 * x
            + T(2) * distortion.p2 * y;

        T j11 =
            radial
            + T(2) * ySquared * radialDerivative
            + T(6) * distortion.p1 * y
            + T(2) * distortion.p2 * x;

        T determinant = j00 * j11 - j01 * j10;

        T determinantScale =
            std::max(
                std::max(std::abs(j00), std::abs(j01)),
                std::max(std::abs(j10), std::abs(j11)));

        if (determinantScale == T(0))
        {
            std::cerr << "Jacobian is 0." << std::endl;

            return std::nullopt;
        }

        static constexpr T determinantTolerance =
            T(1000) * std::numeric_limits<T>::epsilon();

        T minimumDeterminant =
            determinantTolerance * determinantScale * determinantScale;

        if (abs(determinant) <= minimumDeterminant)
        {
            std::cerr << "Determinant is below limit" << std::endl;

            return std::nullopt;
        }

        T dx = ( j11 * residualX - j01 * residualY) / determinant;
        T dy = (-j10 * residualX + j00 * residualY) / determinant;

        x -= dx;
        y -= dy;

#if 0
        if (dx * dx + dy * dy < toleranceSquared)
        {
            break;
        }
#endif
    }

    return intrinsics_pixels.ToSensorPixel(tau::Point2d<T>{x, y});
}


} // end namespace distortion


} // end namespace ray
