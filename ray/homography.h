#pragma once


#include <ray/normalize_pixel.h>
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


class LogicalToMeters
{
public:
    static constexpr double metersPerMillimeter = 1e-3;

    LogicalToMeters(double chessSquareSize_mm)
        :
        chessSquareSize_m_(chessSquareSize_mm * metersPerMillimeter)
    {

    }

    tau::Point3d<double> operator()(const tau::Point2d<size_t> &logical) const
    {
        tau::Point3d<double> result;

        auto xAndY = logical.template Cast<double>() * this->chessSquareSize_m_;
        result.x = xAndY.x;
        result.y = xAndY.y;

        return result;
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

    Factors CombineHomographyFactors(const PlanarVertices &vertices);

    HomographyMatrix GetHomographyMatrix(const PlanarVertices &vertices);

    IntrinsicsMatrix EstimateIntrinsics(
        const std::vector<PlanarVertices> &namedVertices);

    CalibrationResult<double> RefineIntrinsics(
        const IntrinsicsMatrix &intrinsics,
        const std::vector<PlanarVertices> &namedVertices);

    CalibrationResult<double> Calibrate(
        const std::vector<PlanarVertices> &namedVertices);

    const ray::NormalizePixel & GetNormalizePixel() const
    {
        return this->normalize_;
    }

private:
    HomographySettings settings_;
    LogicalToMeters logicalToMeters_;
    tau::Size<double> sensorSize_;
    ray::NormalizePixel normalize_;
};


} // end namespace ray
