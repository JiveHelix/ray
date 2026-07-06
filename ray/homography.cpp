#include <ray/homography.h>
#include <ray/error.h>
#include <jive/range.h>
#include <tau/svd.h>
#include <cmath>


namespace ray
{


namespace
{


struct ReprojectionParameters
{
    double fx;
    double fy;
    double cx;
    double cy;
    distortion::BrownConrady<double> distortion;
};


using ParameterVector = Eigen::Vector<double, Eigen::Dynamic>;


double GetRmsResidual_pixels(
    const Eigen::Vector<double, Eigen::Dynamic> &residuals)
{
    if (residuals.size() == 0)
    {
        return 0.0;
    }

    const auto pointCount = static_cast<double>(residuals.size()) / 2.0;

    return std::sqrt(residuals.squaredNorm() / pointCount);
}


ParameterVector ToVector(const ReprojectionParameters &parameters)
{
    ParameterVector result(9);

    const auto &distortion = parameters.distortion;

    result <<
        parameters.fx,
        parameters.fy,
        parameters.cx,
        parameters.cy,
        distortion.k1,
        distortion.k2,
        distortion.p1,
        distortion.p2,
        distortion.k3;

    return result;
}


ReprojectionParameters ToReprojectionParameters(
    const ParameterVector &parameters)
{
    return {
        parameters(0),
        parameters(1),
        parameters(2),
        parameters(3),
        {
            parameters(4),
            parameters(5),
            parameters(6),
            parameters(7),
            parameters(8)}};
}


ReprojectionParameters ToReprojectionParameters(
    const IntrinsicsMatrix &intrinsics)
{
    return {
        intrinsics(0, 0),
        intrinsics(1, 1),
        intrinsics(0, 2),
        intrinsics(1, 2),
        {}};
}


IntrinsicsMatrix ToIntrinsicsMatrix(const ReprojectionParameters &parameters)
{
    IntrinsicsMatrix result = IntrinsicsMatrix::Identity();

    result(0, 0) = parameters.fx;
    result(1, 1) = parameters.fy;
    result(0, 2) = parameters.cx;
    result(1, 2) = parameters.cy;

    assert(result(0, 1) == 0.0);

    return result;
}


HomographyMatrix ToPixelHomography(
    const HomographyMatrix &normalizedHomography,
    const tau::Size<double> &sensorSize)
{
    HomographyMatrix result = normalizedHomography;
    double xScale = sensorSize.width / 2.0;
    double yScale = sensorSize.height / 2.0;

    result.row(0) =
        xScale * (normalizedHomography.row(0) + normalizedHomography.row(2));

    result.row(1) =
        yScale * (normalizedHomography.row(1) + normalizedHomography.row(2));

    return result;
}


Eigen::Matrix<double, 3, 4> GetExtrinsics(
    const HomographyMatrix &pixelHomography,
    const ReprojectionParameters &parameters)
{
    auto intrinsicsInverse = ToIntrinsicsMatrix(parameters).inverse();

    Eigen::Vector3<double> first =
        intrinsicsInverse * pixelHomography.col(0);

    Eigen::Vector3<double> second =
        intrinsicsInverse * pixelHomography.col(1);

    double scale = 2.0 / (first.norm() + second.norm());

    Eigen::Matrix<double, 3, 3> rotationEstimate;
    rotationEstimate.col(0) = scale * first;
    rotationEstimate.col(1) = scale * second;
    rotationEstimate.col(2) =
        rotationEstimate.col(0).cross(rotationEstimate.col(1));

    Eigen::JacobiSVD<Eigen::Matrix<double, 3, 3>> svd(
        rotationEstimate,
        Eigen::ComputeFullU | Eigen::ComputeFullV);

    Eigen::Matrix<double, 3, 3> rotation =
        svd.matrixU() * svd.matrixV().transpose();

    if (rotation.determinant() < 0.0)
    {
        rotation.col(2) *= -1.0;
    }

    Eigen::Matrix<double, 3, 4> result;
    result.block<3, 3>(0, 0) = rotation;
    result.col(3) = scale * intrinsicsInverse * pixelHomography.col(2);

    return result;
}


Eigen::Vector3<double> GetRotationVector(
    const Eigen::Matrix<double, 3, 3> &rotation)
{
    Eigen::AngleAxis<double> angleAxis(rotation);

    return angleAxis.axis() * angleAxis.angle();
}


Eigen::Matrix<double, 3, 3> GetRotation(
    const Eigen::Vector3<double> &rotationVector)
{
    double angle = rotationVector.norm();

    if (angle < 1e-12)
    {
        return Eigen::Matrix<double, 3, 3>::Identity();
    }

    return Eigen::AngleAxis<double>(angle, rotationVector / angle)
        .toRotationMatrix();
}


ParameterVector GetInitialParameters(
    const std::vector<HomographyMatrix> &pixelHomographies,
    const IntrinsicsMatrix &intrinsics)
{
    // The first nine parameters are shared camera parameters. Each board adds
    // one rotation-vector and translation pair.
    ParameterVector result(9 + 6 * static_cast<Eigen::Index>(
        pixelHomographies.size()));

    result.head(9) = ToVector(ToReprojectionParameters(intrinsics));
    auto cameraParameters = ToReprojectionParameters(result);

    for (size_t i = 0; i < pixelHomographies.size(); ++i)
    {
        // Zhang's closed-form K gives a good first estimate for each board
        // pose, then the nonlinear pass lets those poses move with K and D.
        auto extrinsics = GetExtrinsics(pixelHomographies[i], cameraParameters);
        Eigen::Index offset = 9 + 6 * static_cast<Eigen::Index>(i);

        result.segment<3>(offset) =
            GetRotationVector(extrinsics.block<3, 3>(0, 0));

        result.segment<3>(offset + 3) = extrinsics.col(3);
    }

    return result;
}


Eigen::Vector<double, Eigen::Dynamic> GetReprojectionResiduals(
    const std::vector<NamedVertices> &namedVertices,
    const World &world,
    const ParameterVector &parameters)
{
    using Index = Eigen::Index;

    ReprojectionParameters cameraParameters =
        ToReprojectionParameters(parameters);

    Index pointCount{};

    for (const auto &vertices: namedVertices)
    {
        pointCount += static_cast<Index>(vertices.size());
    }

    Eigen::Vector<double, Eigen::Dynamic> result(2 * pointCount);
    Index row{};

    for (size_t i = 0; i < namedVertices.size(); ++i)
    {
        Index offset = 9 + 6 * static_cast<Index>(i);
        auto rotation = GetRotation(parameters.segment<3>(offset));
        auto translation = parameters.segment<3>(offset + 3);
        const auto &vertices = namedVertices[i];

        for (const auto &vertex: vertices)
        {
            auto worldPoint = world(vertex.logical);

            Eigen::Vector3<double> planarWorld(
                worldPoint.x,
                worldPoint.y,
                0.0);

            Eigen::Vector3<double> camera =
                rotation * planarWorld + translation;

            // Residuals are measured in pixels after perspective projection
            // and lens distortion.
            camera.array() /= camera(2);

            auto distorted =
                distortion::DistortPoint(
                    cameraParameters.distortion,
                    tau::Point2d<double>(camera.template head<2>()));

            double predictedX =
                cameraParameters.fx * distorted.x + cameraParameters.cx;

            double predictedY =
                cameraParameters.fy * distorted.y + cameraParameters.cy;

            result(row++) = predictedX - vertex.pixel.x;
            result(row++) = predictedY - vertex.pixel.y;
        }
    }

    return result;
}


double GetStep(double value, size_t index)
{
    if (index < 4)
    {
        return std::max(1e-3, std::abs(value) * 1e-6);
    }

    if (index < 9)
    {
        return std::max(1e-8, std::abs(value) * 1e-4);
    }

    return std::max(1e-8, std::abs(value) * 1e-4);
}


} // end anonymous namespace


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

    intrinsics.array() /= intrinsics(2, 2);

    const auto &n = this->normalize_;

    intrinsics(0, 0) = n.Unscale(intrinsics(0, 0), true);
    intrinsics(1, 1) = n.Unscale(intrinsics(1, 1), false);
    intrinsics(0, 2) = n.ToPixel(intrinsics(0, 2), true);
    intrinsics(1, 2) = n.ToPixel(intrinsics(1, 2), false);

    return intrinsics;
}


CalibrationResult<double> Homography::RefineIntrinsics(
    const IntrinsicsMatrix &intrinsics,
    const std::vector<NamedVertices> &namedVertices)
{
    using Index = Eigen::Index;

    Index pointCount{};

    for (const auto &verices: namedVertices)
    {
        pointCount += static_cast<Index>(verices.size());
    }

    Index parameterCount =
        9 + 6 * static_cast<Index>(namedVertices.size());

    if (2 * pointCount <= parameterCount)
    {
        throw RayError("Underdetermined reprojection verices");
    }

    std::vector<HomographyMatrix> pixelHomographies;
    pixelHomographies.reserve(namedVertices.size());

    for (const auto &verices: namedVertices)
    {
        pixelHomographies.push_back(
            ToPixelHomography(
                this->GetHomographyMatrix(verices),
                this->sensorSize_));
    }

    ParameterVector parameters =
        GetInitialParameters(pixelHomographies, intrinsics);

    double damping = 1e-3;

    auto residuals = GetReprojectionResiduals(
        namedVertices,
        this->world_,
        parameters);

    double error = residuals.squaredNorm();

    // Levenberg-Marquardt style refinement over K, distortion, and every board
    // pose. Skew is excluded from the parameter vector, so it remains zero.
    for (size_t iteration = 0; iteration < 60; ++iteration)
    {
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> jacobian(
            residuals.size(),
            parameters.size());

        // Numerical derivatives keep the optimizer local to this translation
        // unit without adding a dependency on a larger optimization library.
        for (Index parameterIndex = 0; parameterIndex < parameters.size();
            ++parameterIndex)
        {
            double step =
                GetStep(
                    parameters(parameterIndex),
                    static_cast<size_t>(parameterIndex));

            ParameterVector trial = parameters;
            trial(parameterIndex) += step;

            auto trialResiduals = GetReprojectionResiduals(
                namedVertices,
                this->world_,
                trial);

            jacobian.col(parameterIndex) = (trialResiduals - residuals) / step;
        }

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> normal =
            jacobian.transpose() * jacobian;

        Eigen::Vector<double, Eigen::Dynamic> gradient =
            jacobian.transpose() * residuals;

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> damped = normal;
        damped.diagonal().array() +=
            damping * normal.diagonal().cwiseAbs().array().max(1.0);

        Eigen::Vector<double, Eigen::Dynamic> update =
            damped.colPivHouseholderQr().solve(-gradient);

        if (!update.allFinite())
        {
            break;
        }

        ParameterVector trial = parameters + update;
        trial(1) = std::max(trial(1), 1.0);
        trial(0) = std::max(trial(0), 1.0);

        auto trialResiduals = GetReprojectionResiduals(
            namedVertices,
            this->world_,
            trial);

        double trialError = trialResiduals.squaredNorm();

        // Accept only downhill steps. Rejected steps increase damping, accepted
        // steps relax it so the solve moves back toward Gauss-Newton.
        if (trialError < error)
        {
            parameters = trial;
            residuals = trialResiduals;

            if (std::abs(error - trialError) < 1e-10)
            {
                error = trialError;
                break;
            }

            error = trialError;
            damping = std::max(damping * 0.5, 1e-12);
        }
        else
        {
            damping = std::min(damping * 4.0, 1e12);
        }
    }

    auto reprojectionParameters = ToReprojectionParameters(parameters);
    auto intrinsicsMatrix = ToIntrinsicsMatrix(reprojectionParameters);

    return {
        Intrinsics<double>::FromArray_pixels(
            this->settings_.pixelSize_microns,
            intrinsicsMatrix),
        reprojectionParameters.distortion,
        GetRmsResidual_pixels(residuals)};
}


} // end namespace ray
