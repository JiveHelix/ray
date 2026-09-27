#pragma once


#include <tau/size.h>
#include <fields/fields.h>
#include <fields/describe.h>
#include <pex/group.h>


namespace ray
{


template<template<typename> typename T>
struct HomographySchema
{
    T<tau::SizeGroup<double>> sensorSize_pixels;
    T<double> pixelSize_microns;
    T<double> squareSize_mm;

    static constexpr auto fieldsTypeName = "Homography";
};


struct HomographySettings: public HomographySchema<pex::Identity>
{
    static constexpr tau::Size<double> defaultImageSize{1920, 1080};
    static constexpr double defaultPixelSize = 10.0;
    static constexpr double defaultSquareSize = 150.0;

    HomographySettings()
        :
        HomographySchema<pex::Identity>{
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
        HomographySchema,
        pex::PlainT<HomographySettings>
    >;

using HomographyControl = typename HomographyGroup::DefaultControl;


} // end namespace ray


extern template struct pex::Group
    <
        ray::HomographySchema,
        pex::PlainT<ray::HomographySettings>
    >;
