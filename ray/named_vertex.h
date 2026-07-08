#pragma once


#include <fields/fields.h>
#include <tau/vector2d.h>
#include <ray/normalize_pixel.h>


namespace ray
{


struct NamedVertex
{
    tau::Point2d<size_t> logical;
    tau::Point2d<double> pixel;

    static constexpr auto fields = std::make_tuple(
        fields::Field(&NamedVertex::logical, "logical"),
        fields::Field(&NamedVertex::pixel, "pixel"));

    static constexpr auto fieldsTypeName = "NamedVertex";

    NamedVertex GetNormalized(const ray::NormalizePixel &normalizePixel) const;
};


DECLARE_OUTPUT_STREAM_OPERATOR(NamedVertex)


using PlanarVertices = std::vector<NamedVertex>;


PlanarVertices GetNormalized(
    const ray::NormalizePixel &normalize,
    const PlanarVertices &planarVertices);


std::vector<PlanarVertices> GetNormalized(
    const ray::NormalizePixel &normalize,
    const std::vector<PlanarVertices> &planarVertices);


std::vector<tau::Point2d<double>> PlanarVerticesToPixels(
    const PlanarVertices &namedVertices);


std::vector<tau::Point2d<size_t>> PlanarVerticesToLogicals(
    const PlanarVertices &namedVertices);


} // end namespace ray
