#include <ray/homography.h>
#include <ray/error.h>
#include <ceres/ceres.h>


namespace ray
{


namespace
{


using Eigen::Index;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::Matrix3d;
using Eigen::Matrix4d;


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
    const IntrinsicsMatrix &intrinsics,
    distortion::Direction direction)
{
    return {
        intrinsics,
        distortion::BrownConradyBase<double>{direction, 0, 0, 0, 0, 0}};
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


template<typename T>
Eigen::Matrix<T, 3, 3> GetIntrinsicsMatrix(
    const T *parameters)
{
    using Matrix = Eigen::Matrix<T, 3, 3>;
    Matrix result = Matrix::Identity();
    result(0, 0) = parameters[0];
    result(1, 1) = parameters[1];
    result(0, 2) = parameters[2];
    result(1, 2) = parameters[3];

    return result;
}


ReprojectionParameters ToReprojectionParameters(
    const VectorXd &parameters,
    distortion::Direction direction)
{
    assert(parameters.size() >= 9);

    IntrinsicsMatrix intrinsics = GetIntrinsicsMatrix(parameters.data());

    return {
        intrinsics,
        {direction, parameters.data() + 4}};
}


// Return the angle axis representation of the rotation matrix.
Eigen::Vector3<double> GetRotationVector(
    const Eigen::Matrix<double, 3, 3> &rotation)
{
    Eigen::AngleAxis<double> angleAxis(rotation);

    return angleAxis.axis() * angleAxis.angle();
}


// Return the rotation matrix defined by the angle axis representation.
template<typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 3> GetRotationMatrix(
    const Eigen::DenseBase<Derived> &rotationVector)
{
    static_assert(
        Derived::ColsAtCompileTime == 1
            || Derived::ColsAtCompileTime == Eigen::Dynamic,
        "rotationVector must be a column vector");

    using T = typename Derived::Scalar;

    if (rotationVector.size() != 3)
    {
        throw RayError("Rotation vector must have 3 elements");
    }

    T angle = rotationVector.derived().norm();

    if (angle < T(1e-12))
    {
        return Eigen::Matrix<T, 3, 3>::Identity();
    }

    Eigen::Vector<T, 3> axis = rotationVector.derived() / angle;

    return Eigen::AngleAxis<T>(angle, axis).toRotationMatrix();
}


VectorXd GetInitialParameters(
    const IntrinsicsMatrix &intrinsics,
    const std::vector<HomographyMatrix> &homographies,
    distortion::Direction direction)
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
        ToVector(ToReprojectionParameters(intrinsics, direction));

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


class ForwardReprojectionResidual
{
public:
    ForwardReprojectionResidual(
        const tau::Point2d<double> &observed,
        const tau::Point3d<double> &world)
        :
        observed_(observed),
        world_(world.GetHomogeneous())
    {

    }

    template<typename T>
    static Eigen::Matrix<T, 4, 4> GetExtrinsics(const T *pose)
    {
        Eigen::Matrix<T, 4, 4> result = Eigen::Matrix<T, 4, 4>::Identity();

        // Take the first values of pose as the rotation vector.
        result.template block<3, 3>(0, 0) =
            GetRotationMatrix(Eigen::Map<const Eigen::Vector<T, 3>>(pose));

        result.template block<3, 1>(0, 3) =
            Eigen::Map<const Eigen::Vector<T, 3>>(pose + 3);

        return result;
    }

    // Let Ceres Solver use its own type for auto differentiation.
    template<typename T>
    bool operator()(
        const T *camera,
        const T *pose,
        T *residuals) const
    {
        auto intrinsics = GetIntrinsicsMatrix(camera);

        auto brownConrady =
            distortion::BrownConrady<T>(
                distortion::Direction::forward,
                camera + 4);

        Eigen::Vector<T, 4> trueCameraPoint =
            GetExtrinsics(pose) * this->world_.template cast<T>();

        Eigen::Vector<T, 3> normalizedTrueCameraPoint =
            trueCameraPoint.template head<3>() / trueCameraPoint(2);

        auto distortedPoint = brownConrady.Apply(normalizedTrueCameraPoint);

        Eigen::Vector<T, 3> predicted_pixels = intrinsics * distortedPoint;

        residuals[0] = predicted_pixels(0) - T(this->observed_.x);
        residuals[1] = predicted_pixels(1) - T(this->observed_.y);

        return true;
    }

private:
    // Homogeneous points
    tau::Point2d<double> observed_;
    Vector4d world_;
};


class InverseReprojectionResidual
{
public:
    InverseReprojectionResidual(
        const tau::Point2d<double> &observed,
        const tau::Point3d<double> &world)
        :
        observed_(observed),
        world_(world.GetHomogeneous())
    {

    }

    template<typename T>
    static Eigen::Matrix<T, 4, 4> GetExtrinsics(const T *pose)
    {
        Eigen::Matrix<T, 4, 4> result = Eigen::Matrix<T, 4, 4>::Identity();

        // Take the first values of pose as the rotation vector.
        result.template block<3, 3>(0, 0) =
            GetRotationMatrix(Eigen::Map<const Eigen::Vector<T, 3>>(pose));

        result.template block<3, 1>(0, 3) =
            Eigen::Map<const Eigen::Vector<T, 3>>(pose + 3);

        return result;
    }

    // Let Ceres Solver use its own type for auto differentiation.
    template<typename T>
    bool operator()(
        const T *camera,
        const T *pose,
        T *residuals) const
    {
        auto intrinsics = GetIntrinsicsMatrix(camera);

        auto intrinsicsAsPixels =
            ray::IntrinsicsAsPixels<T>::FromArray(intrinsics);

        auto brownConrady =
            distortion::BrownConrady<T>(
                distortion::Direction::inverse,
                camera + 4);

        Eigen::Vector<T, 4> trueCameraPoint =
            GetExtrinsics(pose) * this->world_.template cast<T>();

        Eigen::Vector<T, 3> normalizedTrueCameraPoint =
            trueCameraPoint.template head<3>() / trueCameraPoint(2);

        // Normalize the observed point.
        auto normalizedObservation =
            intrinsicsAsPixels.ToNormalizedPixel(this->observed_);

        auto correctedPoint =
            intrinsicsAsPixels.ToSensorPixel(
                brownConrady.Apply(normalizedObservation));

        Eigen::Vector<T, 3> truePoint = intrinsics * normalizedTrueCameraPoint;

        residuals[0] = correctedPoint.x - truePoint(0);
        residuals[1] = correctedPoint.y - truePoint(1);

        return true;
    }

private:
    // Homogeneous points
    tau::Point2d<double> observed_;
    Vector4d world_;
};


VectorXd GetForwardReprojectionResiduals(
    const std::vector<PlanarVertices> &namedVertices,
    const LogicalToMeters &logicalToMeters,
    const VectorXd &parameters,
    double unscale)
{
    ReprojectionParameters cameraParameters =
        ToReprojectionParameters(parameters, distortion::Direction::forward);

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
            auto planarWorld = logicalToMeters(vertex.logical).ToEigen();

            Eigen::Vector3<double> camera =
                rotation * planarWorld + translation;

            camera(0) /= camera(2);
            camera(1) /= camera(2);

            auto distorted = cameraParameters.distortion.Apply(
                tau::Point2d<double>(camera.template head<2>()));

            // Residuals are measured in pixels after perspective projection
            // and lens distortion.
            Vector3d predicted =
                cameraParameters.intrinsics * distorted.GetHomogeneous();

            auto residual =
                predicted.template head<2>().array()
                - vertex.pixel.ToEigen().array();

            result(row++) = residual(0);
            result(row++) = residual(1);
        }
    }

    // The quantity I want to minimize is geometric error measured in pixels.
    return result * unscale;
}


VectorXd GetInverseReprojectionResiduals(
    const std::vector<PlanarVertices> &namedVertices,
    const LogicalToMeters &logicalToMeters,
    const VectorXd &parameters,
    double unscale)
{
    ReprojectionParameters cameraParameters =
        ToReprojectionParameters(parameters, distortion::Direction::inverse);

    auto intrinsicsAsPixels =
        ray::IntrinsicsAsPixels<double>::FromArray(cameraParameters.intrinsics);

    auto intrinsicsMatrix = intrinsicsAsPixels.GetArray();

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
            // Normalize the observed point.
            auto normalizedObservation =
                intrinsicsAsPixels.ToNormalizedPixel(vertex.pixel);

            auto correctedPoint =
                intrinsicsAsPixels.ToSensorPixel(
                    cameraParameters.distortion.Apply(normalizedObservation));

            auto planarWorld = logicalToMeters(vertex.logical).ToEigen();

            Eigen::Vector3<double> trueCameraPoint =
                rotation * planarWorld + translation;

            trueCameraPoint(0) /= trueCameraPoint(2);
            trueCameraPoint(1) /= trueCameraPoint(2);
            trueCameraPoint(2) = 1;

            Eigen::Vector<double, 3> truePoint =
                intrinsicsMatrix * trueCameraPoint;

            result(row++) = correctedPoint.x - truePoint(0);
            result(row++) = correctedPoint.y - truePoint(1);
        }
    }

    // The quantity I want to minimize is geometric error measured in pixels.
    return result * unscale;
}


template<typename ResidualFunction>
using CostFunction =
    ceres::AutoDiffCostFunction
    <
        ResidualFunction,

        // Two residuals
        2,

        // 9 parameters in block 0
        cameraParameterCount,

        // 6 parameters in block 1
        poseParameterCount
    >;



using ForwardReprojectionCostFunction =
    CostFunction<ForwardReprojectionResidual>;

using InverseReprojectionCostFunction =
    CostFunction<InverseReprojectionResidual>;


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


template<typename ResidualFunction>
void BuildProblem(
    ceres::Problem &problem,
    const std::vector<PlanarVertices> &validVertices,
    VectorXd &parameters,
    std::shared_ptr<ceres::ParameterBlockOrdering> ordering,
    const LogicalToMeters &logicalToMeters)
{
    using ProblemCostFunction = CostFunction<ResidualFunction>;
    double *cameraParameters = parameters.data();

    for (size_t i = 0; i < validVertices.size(); ++i)
    {
        // pose parameters come after camera parameters.
        double *poseParameters =
            cameraParameters + cameraParameterCount + (poseParameterCount * i);

        problem.AddParameterBlock(
            poseParameters,
            poseParameterCount);

        ordering->AddElementToGroup(poseParameters, 0);

        for (const auto &vertex: validVertices[i])
        {
            auto worldPoint = logicalToMeters(vertex.logical);

            auto residual =
                std::make_unique<ResidualFunction>(
                    vertex.pixel,
                    worldPoint);

            auto costFunction =
                std::make_unique<ProblemCostFunction>(
                    residual.release());

            problem.AddResidualBlock(
                costFunction.release(),

                // We are not using a loss function.
                nullptr,

                cameraParameters,
                poseParameters);
        }
    }
}


} // end anonymous namespace


CalibrationResult<double> Homography::RefineIntrinsics(
    const IntrinsicsMatrix &intrinsics,
    const std::vector<PlanarVertices> &namedVertices,
    distortion::Direction direction)
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
        GetInitialParameters(intrinsics, homographies, direction);

    ceres::Problem problem;

    double *cameraParameters = parameters.data();

    problem.AddParameterBlock(
        cameraParameters,
        cameraParameterCount);

    auto ordering =
        std::make_shared<ceres::ParameterBlockOrdering>();

    ordering->AddElementToGroup(cameraParameters, 1);

    if (direction == distortion::Direction::forward)
    {
        BuildProblem<ForwardReprojectionResidual>(
            problem,
            validVertices,
            parameters,
            ordering,
            this->logicalToMeters_);
    }
    else if (direction == distortion::Direction::inverse)
    {
        BuildProblem<InverseReprojectionResidual>(
            problem,
            validVertices,
            parameters,
            ordering,
            this->logicalToMeters_);
    }
    else
    {
        throw std::logic_error("Unknown distortion direction");
    }

    ceres::Solver::Options options;

    options.max_num_iterations = 50;
    options.function_tolerance = 1e-10;
    options.gradient_tolerance = 1e-12;
    options.parameter_tolerance = 1e-12;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.linear_solver_ordering = ordering;

#ifdef NDEBUG
    options.logging_type = ceres::SILENT;
    options.minimizer_progress_to_stdout = false;
#endif

    ceres::Solver::Summary summary;

    ceres::Solve(options, &problem, &summary);

    if (!summary.IsSolutionUsable())
    {
        throw RayError(
            fmt::format(
                "Ceres Solver failed to refine intrinsics: {}",
                summary.message));
    }
    else
    {
        // std::cout << summary.FullReport() << std::endl;
    }

    VectorXd residuals;

    if (direction == distortion::Direction::forward)
    {
        residuals =
            GetForwardReprojectionResiduals(
                validVertices,
                this->logicalToMeters_,
                parameters,
                this->normalize_.GetUnscale());
    }
    else
    {
        assert(direction == distortion::Direction::inverse);

        residuals =
            GetInverseReprojectionResiduals(
                validVertices,
                this->logicalToMeters_,
                parameters,
                this->normalize_.GetUnscale());
    }

    auto reprojectionParameters =
        ToReprojectionParameters(parameters, direction);

    return {
        Intrinsics<double>::FromArray_pixels(
            this->settings_.pixelSize_microns,
            this->normalize_.ToPixels(reprojectionParameters.intrinsics)),

        reprojectionParameters.distortion,
        GetRmsResidual_pixels(residuals)};
}


} // end namespace ray
