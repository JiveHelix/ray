#pragma once


#include <ray/intrinsics.h>
#include <ray/distortion.h>


namespace ray
{


template<typename Float>
struct LensCalibrationSchema
{
    template<template<typename> typename T>
    struct Schema
    {
        T<IntrinsicsGroup<Float>> intrinsics;
        T<distortion::BrownConradyGroup<Float>> distortion;

        static constexpr auto fieldsTypeName = "LensCalibration";
    };
};


template<typename T>
struct LensCalibration:
    public LensCalibrationSchema<T>::template Schema<pex::Identity>
{
    using Base =
        typename LensCalibrationSchema<T>::template Schema<pex::Identity>;

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
        LensCalibrationSchema<T>::template Schema,
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
