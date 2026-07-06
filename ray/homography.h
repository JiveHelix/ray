#pragma once


#include <tau/normalize_pixel.h>
#include <ray/lens_calibration.h>
#include <ray/named_vertex.h>
#include <ray/homography_settings.h>


namespace ray
{


using HomographyMatrix = Eigen::Matrix<double, 3, 3>;
using ConstrainedElements = Eigen::RowVector<double, 6>;
using ConstrainedFactors = Eigen::Matrix<double, 2, 6>;


ConstrainedElements GetConstrainedElements(
    const HomographyMatrix &homography,
    Eigen::Index i,
    Eigen::Index j);


ConstrainedFactors GetConstrainedFactors(const HomographyMatrix &homography);


class World
{
public:
    static constexpr double metersPerMillimeter = 1e-3;

    World(double chessSquareSize_mm)
        :
        chessSquareSize_m_(chessSquareSize_mm * metersPerMillimeter)
    {

    }

    tau::Point2d<double> operator()(const tau::Point2d<size_t> &logical) const
    {
        return logical.template Cast<double>() * this->chessSquareSize_m_;
    }

private:
    double chessSquareSize_m_;
};


using IntrinsicsMatrix = Eigen::Matrix<double, 3, 3>;


class Homography
{
public:
    Homography(const HomographySettings &settings);

    Eigen::Matrix<double, 2, 9>
    GetHomographyFactors(const NamedVertex &vertext);

    using Factors = Eigen::Matrix<double, Eigen::Dynamic, 9>;

    Factors CombineHomographyFactors(const NamedVertices &vertices);

    HomographyMatrix GetHomographyMatrix(const NamedVertices &vertices);

    IntrinsicsMatrix EstimateIntrinsics(
        const std::vector<NamedVertices> &namedVertices);

    CalibrationResult<double> RefineIntrinsics(
        const IntrinsicsMatrix &intrinsics,
        const std::vector<NamedVertices> &namedVertices);

private:
    HomographySettings settings_;
    World world_;
    tau::Size<double> sensorSize_;
    tau::NormalizePixel normalize_;
};


} // end namespace ray
