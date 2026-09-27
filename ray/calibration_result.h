#pragma once


#include <ray/lens_calibration.h>


namespace ray
{


template<typename Float>
struct CalibrationResultSchema
{
    template<template<typename> typename T>
    struct Schema
    {
        T<LensCalibrationGroup<Float>> lensCalibration;
        T<Float> rmsReprojectionError_pixels;

        static constexpr auto fieldsTypeName = "CalibrationResult";
    };
};


template<typename Float>
using CalibrationResultGroup =
    pex::Group
    <
        CalibrationResultSchema<Float>::template Schema
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
