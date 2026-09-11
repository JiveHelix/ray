#pragma once


#include <fields/formatter.h>
#include <pex/group.h>
#include <ray/intrinsics.h>
#include <nlohmann/json.hpp>


namespace ray
{


namespace distortion
{



enum class Direction: int
{
    forward = 0,
    inverse
};


struct DirectionChoices
{
    using Type = Direction;
    static std::vector<Direction> GetChoices();
};


using DirectionSelect = pex::MakeSelect<DirectionChoices>;
using DirectionModel = pex::ModelSelector<DirectionSelect>;
using DirectionControl = pex::ControlSelector<DirectionSelect>;

struct DirectionConverter
{
    static std::string ToString(Direction direction);
};


std::ostream & operator<<(std::ostream &, Direction);

void to_json(nlohmann::json &json, Direction direction);

void from_json(const nlohmann::json &json, Direction &direction);


template<typename T>
struct BrownConradyFields
{
    static constexpr auto fields = std::make_tuple(
        fields::Field(&T::direction, "direction"),
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
        T<DirectionSelect> direction;
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


template<typename T>
struct BrownConrady: public BrownConradyBase<T>
{
    template<typename U, typename Style = tau::Round>
    BrownConrady<U> Cast() const
    {
        return tau::CastFields<BrownConrady<U>, U, Style>(*this);
    }

    BrownConrady()
        :
        BrownConradyBase<T>{}
    {

    }

    BrownConrady(const BrownConradyBase<T> &base)
        :
        BrownConradyBase<T>(base)
    {

    }

    BrownConrady(Direction direction_, const T *parameters)
        :
        BrownConradyBase<T>{
            .direction = direction_,
            .k1 = parameters[0],
            .k2 = parameters[1],
            .p1 = parameters[2],
            .p2 = parameters[3],
            .k3 = parameters[4]}
    {

    }

    // Apply_ functor stores intermediate results that can be re-used in the
    // "Undo" iterative solver.
    struct Apply_
    {
        BrownConrady brownConrady;
        tau::Point2d<T> point;
        T xPow2;
        T yPow2;
        T radiusPow2;
        T radiusPow4;
        T radiusPow6;
        T radialDistortion;
        T xy;

        Apply_(
            const BrownConrady &brownConrady_,
            const tau::Point2d<T> &point_)
            :
            brownConrady(brownConrady_),
            point(point_),
            xPow2(point_.x * point_.x),
            yPow2(point_.y * point_.y),
            radiusPow2(this->xPow2 + this->yPow2),
            radiusPow4(this->radiusPow2 * this->radiusPow2),
            radiusPow6(this->radiusPow4 * this->radiusPow2),

            radialDistortion(
                T(1)
                + brownConrady_.k1 * this->radiusPow2
                + brownConrady_.k2 * this->radiusPow4
                + brownConrady_.k3 * this->radiusPow6),

            xy(point_.x * point_.y)
        {

        }

        tau::Point2d<T> operator()()
        {
            tau::Point2d<T> result;

            result.x =
                (this->point.x * this->radialDistortion)
                + (T(2) * this->brownConrady.p1 * this->xy)
                + (this->brownConrady.p2
                    * (this->radiusPow2 + T(2) * this->xPow2));

            result.y =
                (this->point.y * this->radialDistortion)
                + (this->brownConrady.p1
                    * (this->radiusPow2 + T(2) * this->yPow2))
                + (T(2) * this->brownConrady.p2 * this->xy);

            return result;
        }
    };

    tau::Point2d<T> Apply(
        const tau::Point2d<T> &point) const
    {
        return Apply_(*this, point)();
    }

    Eigen::Vector<T, 3> Apply(
        const Eigen::Vector<T, 3> &point) const
    {
        Eigen::Vector<T, 3> result{};
        result(2) = T(1);

        result.template head<2>() =
            Apply_(*this, tau::Point2d<T>(point.template head<2>()))()
                .ToEigen();

        return result;
    }

    struct UndoResult
    {
        tau::Point2d<T> point;
        T residualSquared;
        bool singularJacobian;
        bool converged;
    };

    UndoResult Undo(
        const tau::Point2d<T> &point,
        int maxIterations = 10,
        T tolerance = T(1e-6),
        T minimumAreaScale = T(0.01)) const
    {
        const T toleranceSquared = tolerance * tolerance;
        auto resultPoint = point;
        bool converged = false;
        T residualSquared = std::numeric_limits<T>::infinity();

        for (int i = 0; i < maxIterations; ++i)
        {
            auto apply = Apply_(*this, resultPoint);
            auto estimate = apply();
            auto residual = estimate - point;

            residualSquared =
                residual.x * residual.x + residual.y * residual.y;

            if (residualSquared < toleranceSquared)
            {
                converged = true;
                break;
            }

            // Find the derivative of the radialDistortion w.r.t r^2
            T dRadialWrtRadiusSquared =
                this->k1
                + T(2) * this->k2 * apply.radiusPow2
                + T(3) * this->k3 * apply.radiusPow4;

           /**
                // Showing the work for
                    d(p2 * (radiusPow2 + T(2) * xPow2)) / dx

                p2 * (radiusPow2 + T(2) * xPow2)
                    ==

                p2 * (xPow2 + yPow2) + p2 * T(2) * xPow2
                    ==
                p2 * xPow2
                + .p2 * yPow2
                + p2 * T(2) * xPow2

                d/dx ==
                    2 * x * p2 + 0 + 4 * x * p2
                    ==
                    6 * x * p2

                // Similarly for
                    d(p1 * (radiusPow2 + T(2) * yPow2)) / dy
                    ==
                    6 * y * p1

            **/

            Eigen::Matrix<T, 2, 2> jacobian{};

            // d distortionX / dx
            jacobian(0, 0) =
                // Apply the product rule then the chain rule
                apply.radialDistortion
                + T(2) * apply.xPow2 * dRadialWrtRadiusSquared
                + T(2) * this->p1 * resultPoint.y
                + T(6) * this->p2 * resultPoint.x;

            // d distortionX / dy and d distortionY / dx have the same value for
            // normalized correction.
            //
            // d distortionX / dy
            jacobian(0, 1) =
                // Apply the product rule then the chain rule
                T(2) * apply.xy * dRadialWrtRadiusSquared
                + T(2) * this->p1 * resultPoint.x
                + T(2) * this->p2 * resultPoint.y;

            // d distortionY / dx
            jacobian(1, 0) = jacobian(0, 1);

            // d distortionY / dy
            jacobian(1, 1) =
                apply.radialDistortion
                + T(2) * apply.yPow2 * dRadialWrtRadiusSquared
                + T(6) * this->p1 * resultPoint.y
                + T(2) * this->p2 * resultPoint.x;

            // We care that the jacobian is invertible numerically, but we hold
            // it to a higher standard based on geometric interpretation.
            if (jacobian.determinant() < minimumAreaScale)
            {
                return {
                    .point = resultPoint,
                    .residualSquared = residualSquared,
                    .singularJacobian = true,
                    .converged = false};
            }

            // jacobian * step = residual
            // step = jacobian^-1 * residual
            Eigen::Vector<T, 2> delta = jacobian.inverse() * residual.ToEigen();

            resultPoint -= tau::Point2d<T>(delta);
        }

        return {
            .point = resultPoint,
            .residualSquared = residualSquared,
            .singularJacobian = false,
            .converged = converged};
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
constexpr T GetNormalizedTolerance(
    const IntrinsicsAsPixels<T> &intrinsics_pixels,
    T pixelTolerance)
{
    T maxFocalLength_pixels =
        std::max(
            intrinsics_pixels.focalLengthX,
            intrinsics_pixels.focalLengthY);

    return pixelTolerance / maxFocalLength_pixels;
}



} // end namespace distortion


} // end namespace ray


template<>
struct fmt::formatter<ray::distortion::Direction>
{
    constexpr auto parse(fmt::format_parse_context & context)
    {
        return context.begin();
    }

    template<typename FormatContext>
    auto format(
        const ray::distortion::Direction &direction,
        FormatContext &context) const
    {
        return fmt::format_to(
            context.out(),
            "{}",
            ray::distortion::DirectionConverter::ToString(direction));
    }
};


template<>
struct fmt::formatter<ray::distortion::BrownConrady<double>>
    : fields::Formatter<ray::distortion::BrownConrady<double>, double> {};

template<>
struct fmt::formatter<ray::distortion::BrownConrady<float>>
    : fields::Formatter<ray::distortion::BrownConrady<float>, float> {};
