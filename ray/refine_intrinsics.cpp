#include <ray/homography.h>
#include <ray/error.h>


namespace ray
{


namespace
{


constexpr Eigen::Index cameraParameterCount = 9;
constexpr Eigen::Index poseParameterCount = 6;


#if 0
struct ReprojectionResidual
{
private:
    tau::Point2d<double> world_;
    tau::Point2d<double> observed_pixels_;

public:
    ReprojectionResidual(
        const tau::Point2d<double> &world,
        const tau::Point2d<double> &observed_pixels)
        :
        world_(world),
        observed_pixels_(observed_pixels)
    {

    }

    template<typename T>
    bool operator()(
        const T *camera,
        const T *pose,
        T *residuals) const
    {



        distortion::BrownConrady<T> distortion;

        distortion.k1 = parameters[0];
        distortion.k2 = parameters[1];
        distortion.p1 = parameters[2];
        distortion.p2 = parameters[3];
        distortion.k3 = parameters[4];

        tau::Point2d<T> idealAsT = this->ideal.template Cast<T>();

        tau::Point2d<T> predicted =
            distortion::DistortPoint(distortion, idealAsT);

        residuals[0] = predicted.x - T(measured.x);
        residuals[1] = predicted.y - T(measured.y);

        return true;
    }
};
#endif


using ParameterVector = Eigen::Vector<double, Eigen::Dynamic>;


struct ReprojectionParameters
{
    double fx;
    double fy;
    double cx;
    double cy;
    distortion::BrownConrady<double> distortion;
};


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


Eigen::Matrix<double, 3, 4> GetExtrinsics(
    const HomographyMatrix &homography,
    const ReprojectionParameters &parameters)
{
    using RotationMatrix = Eigen::Matrix<double, 3, 3>;
    using RotationColumns = Eigen::Matrix<double, 3, 2>;

    auto intrinsicsInverse = ToIntrinsicsMatrix(parameters).inverse();

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


ParameterVector ToVector(const ReprojectionParameters &parameters)
{
    ParameterVector result(cameraParameterCount);

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


ParameterVector GetInitialParameters(
    const std::vector<HomographyMatrix> &homographies,
    const IntrinsicsMatrix &intrinsics)
{
    assert(homographies.size() < std::numeric_limits<Eigen::Index>::max());
    auto homographyCount = static_cast<Eigen::Index>(homographies.size());

    // The first nine parameters are shared camera parameters. Each board adds
    // one 3-value rotation-vector and one 3-value translation.
    auto parameterCount =
        cameraParameterCount + (poseParameterCount * homographyCount);

    ParameterVector result(parameterCount);

    result.head(cameraParameterCount) =
        ToVector(ToReprojectionParameters(intrinsics));

    auto cameraParameters = ToReprojectionParameters(result);

    for (size_t i = 0; i < homographies.size(); ++i)
    {
        // Zhang's closed-form K gives a good first estimate for each board
        // pose, then the nonlinear pass lets those poses move with K and D.
        auto extrinsics = GetExtrinsics(homographies[i], cameraParameters);

        Eigen::Index offset = cameraParameterCount
            + (poseParameterCount * static_cast<Eigen::Index>(i));

        result.segment<3>(offset) =
            GetRotationVector(extrinsics.block<3, 3>(0, 0));

        result.segment<3>(offset + 3) = extrinsics.col(3);
    }

    return result;
}


double GetStep(double value, size_t index)
{
    if (index < 4)
    {
        return std::max(1e-3, std::abs(value) * 1e-6);
    }

    if (index < cameraParameterCount)
    {
        return std::max(1e-8, std::abs(value) * 1e-4);
    }

    return std::max(1e-8, std::abs(value) * 1e-4);
}


Eigen::Vector<double, Eigen::Dynamic> GetReprojectionResiduals(
    const std::vector<PlanarVertices> &namedVertices,
    const World &world,
    const ParameterVector &parameters,
    double unscale)
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

    // The quantity I want to minimize is geometric error measured in pixels.
    return result * unscale;
}


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


} // end anonymous namespace


CalibrationResult<double> Homography::RefineIntrinsics(
    const IntrinsicsMatrix &intrinsics,
    const std::vector<PlanarVertices> &namedVertices)
{
    using Index = Eigen::Index;

    Index pointCount{};

    for (const auto &vertices: namedVertices)
    {
        pointCount += static_cast<Index>(vertices.size());
    }

    // There is one camera, and there are namedVertices.size() views of the
    // chess board.
    Index parameterCount = cameraParameterCount
        + (poseParameterCount * static_cast<Index>(namedVertices.size()));

    if (2 * pointCount <= parameterCount)
    {
        throw RayError("Underdetermined reprojection vertices");
    }

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

    ParameterVector parameters =
        GetInitialParameters(homographies, intrinsics);

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
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> jacobian(
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

            ParameterVector trial = parameters;
            trial(parameterIndex) += step;

            auto trialResiduals = GetReprojectionResiduals(
                validVertices,
                this->world_,
                trial,
                this->normalize_.GetUnscale());

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
    auto intrinsicsMatrix = ToIntrinsicsMatrix(reprojectionParameters);

    return {
        Intrinsics<double>::FromArray_pixels(
            this->settings_.pixelSize_microns,
            this->normalize_.ToPixels(intrinsicsMatrix)),
        reprojectionParameters.distortion,
        this->normalize_.Unscale(GetRmsResidual_pixels(residuals))};
}


} // end namespace ray
