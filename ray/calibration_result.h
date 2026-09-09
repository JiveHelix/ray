#pragma once


#include <ray/lens_calibration.h>


namespace ray
{


template<typename T>
struct CalibrationResultFields
{
    static constexpr auto fields = std::make_tuple(
        fields::Field(&T::lensCalibration, "lensCalibration"),

        fields::Field(
            &T::rmsReprojectionError_pixels,
            "rmsReprojectionError_pixels"));
};


template<typename Float>
struct CalibrationResultTemplate
{
    template<template<typename> typename T>
    struct Template
    {
        T<LensCalibrationGroup<Float>> lensCalibration;
        T<Float> rmsReprojectionError_pixels;

        static constexpr auto fields =
            CalibrationResultFields<Template>::fields;

        static constexpr auto fieldsTypeName = "CalibrationResult";
    };
};


template<typename Float>
using CalibrationResultGroup =
    pex::Group
    <
        CalibrationResultFields,
        CalibrationResultTemplate<Float>::template Template
    >;

template<typename Float>
using CalibrationResultModel = typename CalibrationResultGroup<Float>::Model;;

template<typename Float>
using CalibrationResult = typename CalibrationResultGroup<Float>::Plain;

template<typename Float>
using CalibrationResultControl =
    typename CalibrationResultGroup<Float>::DefaultControl;


DECLARE_OUTPUT_STREAM_OPERATOR(CalibrationResult<float>)
DECLARE_OUTPUT_STREAM_OPERATOR(CalibrationResult<double>)
DECLARE_EQUALITY_OPERATORS(CalibrationResult<float>)
DECLARE_EQUALITY_OPERATORS(CalibrationResult<double>)


} // end namespace ray
