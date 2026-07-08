#pragma once


#include <tau/size.h>
#include <fields/fields.h>
#include <fields/describe.h>
#include <pex/group.h>


namespace ray
{


template<typename T>
struct HomographyFields
{
    static constexpr auto fields = std::make_tuple(
        fields::Field(&T::sensorSize_pixels, "sensorSize_pixels"),
        fields::Field(&T::pixelSize_microns, "pixelSize_microns"),
        fields::Field(&T::squareSize_mm, "squareSize_mm"));
};


template<template<typename> typename T>
struct HomographyTemplate
{
    T<tau::SizeGroup<double>> sensorSize_pixels;
    T<double> pixelSize_microns;
    T<double> squareSize_mm;

    static constexpr auto fields =
        HomographyFields<HomographyTemplate>::fields;

    static constexpr auto fieldsTypeName = "Homography";
};


struct HomographySettings: public HomographyTemplate<pex::Identity>
{
    static constexpr tau::Size<double> defaultImageSize{1920, 1080};
    static constexpr double defaultPixelSize = 10.0;
    static constexpr double defaultSquareSize = 150.0;

    HomographySettings()
        :
        HomographyTemplate<pex::Identity>{
            defaultImageSize,
            defaultPixelSize,
            defaultSquareSize}
    {

    }
};


DECLARE_EQUALITY_OPERATORS(HomographySettings)
DECLARE_OUTPUT_STREAM_OPERATOR(HomographySettings)


using HomographyGroup =
    pex::Group
    <
        HomographyFields,
        HomographyTemplate,
        pex::PlainT<HomographySettings>
    >;

using HomographyControl = typename HomographyGroup::DefaultControl;


} // end namespace ray


extern template struct pex::Group
    <
        ray::HomographyFields,
        ray::HomographyTemplate,
        pex::PlainT<ray::HomographySettings>
    >;
