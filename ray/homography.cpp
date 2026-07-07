#include <ray/homography.h>
#include <ray/error.h>
#include <jive/range.h>
#include <tau/svd.h>
#include <cmath>

#include <ceres/ceres.h>


namespace ray
{


ConstrainedElements GetConstrainedElements(
    const HomographyMatrix &homography,
    Eigen::Index i,
    Eigen::Index j)
{
    ConstrainedElements result{};

    result(0) = homography(0, i) * homography(0, j);

    result(1) = homography(0, i) * homography(1, j)
        + homography(1, i) * homography(0, j);

    result(2) = homography(2, i) * homography(0, j)
        + homography(0, i) * homography(2, j);

    result(3) = homography(1, i) * homography(1, j);

    result(4) = homography(2, i) * homography(1, j)
        + homography(1, i) * homography(2, j);

    result(5) = homography(2, i) * homography(2, j);

    return result;
}


ConstrainedFactors GetConstrainedFactors(const HomographyMatrix &homography)
{
    ConstrainedFactors result;

    result.block<1, 6>(0, 0) = GetConstrainedElements(homography, 0, 1);

    result.block<1, 6>(1, 0) =
        GetConstrainedElements(homography, 0, 0)
        - GetConstrainedElements(homography, 1, 1);

    return result;
}


Homography::Homography(const HomographySettings &settings)
    :
    settings_(settings),
    world_(settings.squareSize_mm),
    sensorSize_(settings.sensorSize_pixels),
    normalize_(settings.sensorSize_pixels)
{

}


Eigen::Matrix<double, 2, 9>
Homography::GetHomographyFactors(const NamedVertex &vertex)
{
    auto world = this->world_(vertex.logical);
    auto sensor = this->normalize_(vertex.pixel);

    return tau::Matrix<2, 9, double>(
        // The factors dependent on sensor x coordinates
        -world.x,
        -world.y,
        -1,
        0,
        0,
        0,
        sensor.x * world.x,
        sensor.x * world.y,
        sensor.x,

        // The factors dependent on sensor y coordinates
        0,
        0,
        0,
        -world.x,
        -world.y,
        -1,
        sensor.y * world.x,
        sensor.y * world.y,
        sensor.y);
}


Homography::Factors Homography::CombineHomographyFactors(
    const std::vector<NamedVertex> &vertices)
{
    using Index = Eigen::Index;
    auto vertexCount = static_cast<Index>(vertices.size());
    Homography::Factors result(2 * vertexCount, 9);

    for (auto i: jive::Range<Index>(0, vertexCount))
    {
        const auto &vertex = vertices[static_cast<size_t>(i)];
        result.block<2, 9>(i * 2, 0) = this->GetHomographyFactors(vertex);
    }

    return result;
}


HomographyMatrix Homography::GetHomographyMatrix(
    const std::vector<NamedVertex> &vertices)
{
    auto factors = this->CombineHomographyFactors(vertices);

    HomographyMatrix homographyMatrix =
        tau::SvdSolve(factors).reshaped<Eigen::RowMajor>(3, 3);

    return homographyMatrix;
}


IntrinsicsMatrix Homography::EstimateIntrinsics(
    const std::vector<NamedVertices> &namedVertices)
{
    if (namedVertices.size() < 3)
    {
        throw RayError("Underdetermined vertices");
    }

    using ConstrainedFactorGroup = Eigen::Matrix<double, Eigen::Dynamic, 6>;

    Eigen::Index solutionCount =
        static_cast<Eigen::Index>(namedVertices.size());

    ConstrainedFactorGroup factors(2 * solutionCount, 6);

    for (auto i: jive::Range<Eigen::Index>(0, solutionCount))
    {
        const auto &vertices = namedVertices[static_cast<size_t>(i)];

        factors.block<2, 6>(2 * i, 0) = GetConstrainedFactors(
            this->GetHomographyMatrix(vertices));
    }

    Eigen::Vector<double, 6> solution = tau::SvdSolve(factors);

    using Beta = Eigen::Matrix<double, 3, 3>;

    Beta beta{};
    beta(0, 0) = solution(0);
    beta(1, 0) = solution(1);
    beta(2, 0) = solution(2);

    beta(0, 1) = solution(1);
    beta(1, 1) = solution(3);
    beta(2, 1) = solution(4);

    beta(0, 2) = solution(2);
    beta(1, 2) = solution(4);
    beta(2, 2) = solution(5);

    using Cholesky = Eigen::LLT<Beta, Eigen::Upper>;
    Cholesky cholesky(beta);

    if (cholesky.info() != Eigen::Success)
    {
        throw RayError("Degenerate intrinsics solution");
    }

    Eigen::Matrix<double, 3, 3> kInverseTranspose = cholesky.matrixL();

    IntrinsicsMatrix intrinsics
        = kInverseTranspose.transpose().inverse();

    // Canonicalize the intrinsics matrix by forcing K_33 to be 1.
    intrinsics.array() /= intrinsics(2, 2);

    const auto &n = this->normalize_;

    intrinsics(0, 0) = n.Unscale(intrinsics(0, 0), true);
    intrinsics(1, 1) = n.Unscale(intrinsics(1, 1), false);
    intrinsics(0, 2) = n.ToPixel(intrinsics(0, 2), true);
    intrinsics(1, 2) = n.ToPixel(intrinsics(1, 2), false);

    return intrinsics;
}


} // end namespace ray
