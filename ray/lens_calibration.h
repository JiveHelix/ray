#pragma once


#include <ray/intrinsics.h>
#include <ray/distortion.h>


namespace ray
{


template<typename T>
struct LensCalibrationFields
{
    static constexpr auto fields = std::make_tuple(
        fields::Field(&T::intrinsics, "intrinsics"),
        fields::Field(&T::distortion, "distortion"));
};


template<typename Float>
struct LensCalibrationTemplate
{
    template<template<typename> typename T>
    struct Template
    {
        T<IntrinsicsGroup<Float>> intrinsics;
        T<distortion::BrownConradyGroup<Float>> distortion;

        static constexpr auto fields =
            LensCalibrationFields<Template>::fields;

        static constexpr auto fieldsTypeName = "LensCalibration";
    };
};


template<typename T>
struct LensCalibration:
    public LensCalibrationTemplate<T>::template Template<pex::Identity>
{
    using Base =
        typename LensCalibrationTemplate<T>::template Template<pex::Identity>;

    template<typename U, typename Style = tau::Round>
    LensCalibration<U> Cast() const
    {
        return tau::CastFields<LensCalibration<U>, U, Style>(*this);
    }
};


template<typename T>
using LensCalibrationGroup =
    pex::Group
    <
        LensCalibrationFields,
        LensCalibrationTemplate<T>::template Template,
        pex::PlainT<LensCalibration<T>>
    >;


template<typename T>
using LensCalibrationModel =
    typename LensCalibrationGroup<T>::Model;

template<typename T>
using LensCalibrationControl =
    typename LensCalibrationGroup<T>::DefaultControl;


DECLARE_OUTPUT_STREAM_OPERATOR(LensCalibration<float>)
DECLARE_OUTPUT_STREAM_OPERATOR(LensCalibration<double>)
DECLARE_EQUALITY_OPERATORS(LensCalibration<float>)
DECLARE_EQUALITY_OPERATORS(LensCalibration<double>)



} // end namespace ray
