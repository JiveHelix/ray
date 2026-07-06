#include <ray/named_vertex.h>
#include <algorithm>


namespace ray
{


std::vector<tau::Point2d<double>> NamedVerticesToPixels(
    const NamedVertices &namedVertices)
{
    std::vector<tau::Point2d<double>> pixels;
    pixels.reserve(namedVertices.size());

    std::transform(
        std::begin(namedVertices),
        std::end(namedVertices),
        std::back_inserter(pixels),
        [](const NamedVertex &vertex)
        {
            return vertex.pixel;
        });

    return pixels;
}


std::vector<tau::Point2d<size_t>> NamedVerticesToLogicals(
    const NamedVertices &namedVertices)
{
    std::vector<tau::Point2d<size_t>> logicals;
    logicals.reserve(namedVertices.size());

    std::transform(
        std::begin(namedVertices),
        std::end(namedVertices),
        std::back_inserter(logicals),
        [](const NamedVertex &vertex)
        {
            return vertex.logical;
        });

    return logicals;
}


} // end namespace ray
