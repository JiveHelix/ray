#pragma once


#include <fields/fields.h>
#include <tau/vector2d.h>


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
};


DECLARE_OUTPUT_STREAM_OPERATOR(NamedVertex)


using NamedVertices = std::vector<NamedVertex>;


std::vector<tau::Point2d<double>> NamedVerticesToPixels(
    const NamedVertices &namedVertices);


std::vector<tau::Point2d<size_t>> NamedVerticesToLogicals(
    const NamedVertices &namedVertices);


} // end namespace ray
