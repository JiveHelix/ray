#include <ray/named_vertex.h>
#include <algorithm>


namespace ray
{


NamedVertex NamedVertex::GetNormalized(
    const ray::NormalizePixel &normalizePixel) const
{
    NamedVertex result;

    // The logical vertex is not normalized.
    result.logical = this->logical;

    result.pixel = normalizePixel(this->pixel);

    return result;
}


PlanarVertices GetNormalized(
    const ray::NormalizePixel &normalize,
    const PlanarVertices &planarVertices)
{
    PlanarVertices result;

    for (const auto &vertex: planarVertices)
    {
        result.push_back(vertex.GetNormalized(normalize));
    }

    return result;
}


std::vector<PlanarVertices> GetNormalized(
    const ray::NormalizePixel &normalize,
    const std::vector<PlanarVertices> &planarVertices)
{
    std::vector<PlanarVertices> result;

    for (const auto &vertices: planarVertices)
    {
        result.push_back(GetNormalized(normalize, vertices));
    }

    return result;
}


std::vector<tau::Point2d<double>> PlanarVerticesToPixels(
    const PlanarVertices &namedVertices)
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


std::vector<tau::Point2d<size_t>> PlanarVerticesToLogicals(
    const PlanarVertices &namedVertices)
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
