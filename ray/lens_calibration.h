#pragma once


#include <jive/version.h>
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

    static constexpr auto version = jive::Version<uint8_t>(1, 0, 0);

    static LensCalibration Deserialize(std::istream &inputStream)
    {
        auto unstructured = nlohmann::json::parse(inputStream);
        auto fileVersion = jive::Version<uint8_t>(unstructured["version"]);
        auto minimumVersion = jive::Version<uint8_t>(1, 0, 0);

        if (fileVersion < minimumVersion)
        {
            throw std::runtime_error("Incompatible file version");
        }

        return fields::Structure<LensCalibration>(unstructured);
    }

    std::string Serialize() const
    {
        auto unstructured = fields::Unstructure<nlohmann::json>(*this);
        unstructured["version"] = LensCalibration::version.ToString();
        return unstructured.dump(4);
    }

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


template<typename T>
using CalibrationResultGroup =
    pex::Group
    <
        CalibrationResultFields,
        CalibrationResultTemplate<T>::template Template
    >;

template<typename T>
using CalibrationResult = typename CalibrationResultGroup<T>::Plain;

template<typename T>
using CalibrationResultModel =
    typename CalibrationResultGroup<T>::Model;

template<typename T>
using CalibrationResultControl =
    typename CalibrationResultGroup<T>::DefaultControl;


DECLARE_OUTPUT_STREAM_OPERATOR(CalibrationResult<float>)
DECLARE_OUTPUT_STREAM_OPERATOR(CalibrationResult<double>)
DECLARE_EQUALITY_OPERATORS(CalibrationResult<float>)
DECLARE_EQUALITY_OPERATORS(CalibrationResult<double>)


} // end namespace ray
