#include <ray/homography.h>
#include <ray/error.h>


namespace ray
{


namespace
{


using Eigen::Index;


constexpr Index cameraParameterCount = 9;
constexpr Index poseParameterCount = 6;


using Eigen::VectorXd;
using Eigen::MatrixXd;


struct ReprojectionParameters
{
    IntrinsicsMatrix intrinsics;
    distortion::BrownConrady<double> distortion;
};


ReprojectionParameters ToReprojectionParameters(
    const IntrinsicsMatrix &intrinsics)
{
    return {
        intrinsics,
        {}};
}


IntrinsicsMatrix ToIntrinsicsMatrix(const ReprojectionParameters &parameters)
{
    assert(result(0, 1) == 0.0);

    return parameters.intrinsics;
}


Eigen::Matrix<double, 3, 4> GetExtrinsics(
    const HomographyMatrix &homography,
    const IntrinsicsMatrix &intrinsics)
{
    using RotationMatrix = Eigen::Matrix<double, 3, 3>;
    using RotationColumns = Eigen::Matrix<double, 3, 2>;

    auto intrinsicsInverse = intrinsics.inverse();

    Eigen::Vector3<double> first =
        intrinsicsInverse * homography.col(0);

    Eigen::Vector3<double> second =
        intrinsicsInverse * homography.col(1);

    double scale = 2.0 / (first.norm() + second.norm());

    RotationMatrix rotationEstimate;
    rotationEstimate.col(0) = scale * first;
    rotationEstimate.col(1) = scale * second;

    rotationEstimate.col(2) =
        rotationEstimate.col(0).cross(rotationEstimate.col(1));

    Eigen::JacobiSVD<RotationMatrix> svd(
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
    result.col(3) = scale * intrinsicsInverse * homography.col(2);

    return result;
}


VectorXd ToVector(const ReprojectionParameters &parameters)
{
    VectorXd result(cameraParameterCount);

    const auto &intrinsics = parameters.intrinsics;
    const auto &distortion = parameters.distortion;

    result <<
        // fx
        parameters.intrinsics(0, 0),

        // fy
        parameters.intrinsics(1, 1),

        // cx
        parameters.intrinsics(0, 2),

        // cy
        parameters.intrinsics(1, 2),

        distortion.k1,
        distortion.k2,
        distortion.p1,
        distortion.p2,
        distortion.k3;

    return result;
}


ReprojectionParameters ToReprojectionParameters(
    const VectorXd &parameters)
{
    IntrinsicsMatrix intrinsics = IntrinsicsMatrix::Identity();
    intrinsics(0, 0) = parameters(0);
    intrinsics(1, 1) = parameters(1);
    intrinsics(0, 2) = parameters(2);
    intrinsics(1, 2) = parameters(3);

    return {
        intrinsics,
        {
            parameters(4),
            parameters(5),
            parameters(6),
            parameters(7),
            parameters(8)}};
}


// Return the angle axis representation of the rotation matrix.
Eigen::Vector3<double> GetRotationVector(
    const Eigen::Matrix<double, 3, 3> &rotation)
{
    Eigen::AngleAxis<double> angleAxis(rotation);

    return angleAxis.axis() * angleAxis.angle();
}


// Return the rotation matrix defined by the angle axis representation.
Eigen::Matrix<double, 3, 3> GetRotationMatrix(
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


VectorXd GetInitialParameters(
    const IntrinsicsMatrix &intrinsics,
    const std::vector<HomographyMatrix> &homographies)
{
    assert(homographies.size() < std::numeric_limits<Index>::max());
    auto homographyCount = static_cast<Index>(homographies.size());

    // The first nine parameters are shared camera parameters. Each board adds
    // one 3-value rotation-vector and one 3-value translation.
    auto parameterCount =
        cameraParameterCount + (poseParameterCount * homographyCount);

    VectorXd result(parameterCount);

    // Initial the intrinsics/distortion portion of the parameter vector.
    result.head(cameraParameterCount) =
        ToVector(ToReprojectionParameters(intrinsics));

    // Initialize the rotation/translation vectors associated with each board
    // view.
    for (size_t i = 0; i < homographies.size(); ++i)
    {
        // Zhang's closed-form K gives a good first estimate for each board
        // pose, then the nonlinear pass lets those poses move with K and D.
        auto extrinsics = GetExtrinsics(homographies[i], intrinsics);

        Index offset = cameraParameterCount
            + (poseParameterCount * static_cast<Index>(i));

        // The full rotation is represented by an angle-axis vector.
        result.segment<3>(offset) =
            GetRotationVector(extrinsics.block<3, 3>(0, 0));

        // Translation lives in the 4th column
        result.segment<3>(offset + 3) = extrinsics.col(3);
    }

    return result;
}


double GetStep(double value, size_t index)
{
    if (index < 4)
    {
        // fx, fy, cx, cy
        return std::max(1e-3, std::abs(value) * 1e-6);
    }

    if (index < cameraParameterCount)
    {
        // Distortion parameters.
        return std::max(1e-8, std::abs(value) * 1e-4);
    }

    size_t poseIndex =
        (index - cameraParameterCount) % poseParameterCount;

    if (poseIndex < 3)
    {
        // Rotation vector, radians.
        return std::max(1e-6, std::abs(value) * 1e-5);
    }

    // Translation, meters.
    return std::max(1e-6, std::abs(value) * 1e-6);
}


VectorXd GetReprojectionResiduals(
    const std::vector<PlanarVertices> &namedVertices,
    const World &world,
    const VectorXd &parameters,
    double unscale)
{
    ReprojectionParameters cameraParameters =
        ToReprojectionParameters(parameters);

    Index pointCount{};

    for (const auto &vertices: namedVertices)
    {
        pointCount += static_cast<Index>(vertices.size());
    }

    VectorXd result(2 * pointCount);
    Index row{};

    for (size_t i = 0; i < namedVertices.size(); ++i)
    {
        Index offset = cameraParameterCount
            + (poseParameterCount * static_cast<Index>(i));

        auto rotation = GetRotationMatrix(parameters.segment<3>(offset));
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

            auto distorted = cameraParameters.distortion.Apply(
                tau::Point2d<double>(camera.template head<2>()));

            Eigen::Vector3d predicted =
                cameraParameters.intrinsics * distorted.GetHomogeneous();

            auto residual =
                predicted.head<2>().array() - vertex.pixel.ToEigen().array();

            result(row++) = residual(0);
            result(row++) = residual(1);
        }
    }

    // The quantity I want to minimize is geometric error measured in pixels.
    return result * unscale;
}


double GetRmsResidual_pixels(
    const VectorXd &residuals)
{
    if (residuals.size() == 0)
    {
        return 0.0;
    }

    const auto pointCount = static_cast<double>(residuals.size()) / 2.0;

    return std::sqrt(residuals.squaredNorm() / pointCount);
}


} // end anonymous namespace


CalibrationResult<double> Homography::RefineIntrinsics(
    const IntrinsicsMatrix &intrinsics,
    const std::vector<PlanarVertices> &namedVertices)
{
    std::vector<HomographyMatrix> homographies;
    std::vector<PlanarVertices> validVertices;

    homographies.reserve(namedVertices.size());
    validVertices.reserve(namedVertices.size());

    for (const auto &vertices: namedVertices)
    {
        if (vertices.size() < 4)
        {
            continue;
        }

        try
        {
            homographies.push_back(this->GetHomographyMatrix(vertices));
        }
        catch (RayError &)
        {
            continue;
        }

        validVertices.push_back(vertices);
    }

    if (homographies.size() < 2)
    {
        throw RayError("Insufficient homographies for optimization.");
    }

    if (homographies.size() != validVertices.size())
    {
        throw std::logic_error("Unexpected validVertices size");
    }

    Index pointCount{};

    for (const auto &vertices: validVertices)
    {
        pointCount += static_cast<Index>(vertices.size());
    }

    // There is one camera, and there are validVertices.size() views of the
    // chess board.
    Index parameterCount = cameraParameterCount
        + (poseParameterCount * static_cast<Index>(validVertices.size()));

    if (2 * pointCount <= parameterCount)
    {
        throw RayError("Underdetermined reprojection vertices");
    }

    VectorXd parameters =
        GetInitialParameters(intrinsics, homographies);

    double damping = 1e-3;

    auto residuals = GetReprojectionResiduals(
        validVertices,
        this->world_,
        parameters,
        this->normalize_.GetUnscale());

    double error = residuals.squaredNorm();

    // Levenberg-Marquardt style refinement over K, distortion, and every board
    // pose. Skew is excluded from the parameter vector, so it remains zero.
    for (size_t iteration = 0; iteration < 60; ++iteration)
    {
        MatrixXd jacobian(
            residuals.size(),
            parameters.size());

        // Numerical derivatives keep the optimizer local to this translation
        // unit without adding a dependency on a larger optimization library.
        for (
            Index parameterIndex = 0;
            parameterIndex < parameters.size();
            ++parameterIndex)
        {
            double step =
                GetStep(
                    parameters(parameterIndex),
                    static_cast<size_t>(parameterIndex));

            VectorXd trial = parameters;
            trial(parameterIndex) += step;

            auto trialResiduals = GetReprojectionResiduals(
                validVertices,
                this->world_,
                trial,
                this->normalize_.GetUnscale());

            jacobian.col(parameterIndex) = (trialResiduals - residuals) / step;
        }

        MatrixXd normal = jacobian.transpose() * jacobian;

        VectorXd gradient =
            jacobian.transpose() * residuals;

        MatrixXd damped = normal;

        damped.diagonal().array() +=
            damping * normal.diagonal().cwiseAbs().array().max(1.0);

        VectorXd update =
            damped.colPivHouseholderQr().solve(-gradient);

        if (!update.allFinite())
        {
            break;
        }

        VectorXd trial = parameters + update;

        auto trialResiduals = GetReprojectionResiduals(
            validVertices,
            this->world_,
            trial,
            this->normalize_.GetUnscale());

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

    return {
        Intrinsics<double>::FromArray_pixels(
            this->settings_.pixelSize_microns,
            this->normalize_.ToPixels(reprojectionParameters.intrinsics)),

        reprojectionParameters.distortion,
        GetRmsResidual_pixels(residuals)};
}


} // end namespace ray
