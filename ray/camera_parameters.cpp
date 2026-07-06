#include <ray/camera_parameters.h>


namespace ray
{


void RqDecompose::MakePositiveDiagonal(Matrix &R, Matrix &Q, double tolerance)
{
    auto smallest = std::min(R.rows(), R.cols());
    tolerance = std::abs(tolerance);

    for (int i = 0; i < smallest; ++i)
    {
        const double value = R(i, i);

        if (value < -tolerance)
        {
            R.col(i) *= -1.0;
            Q.row(i) *= -1.0;
        }
    }
}


RqDecompose::RqDecompose(const Matrix &input)
    :
    r(),
    q()
{
    using namespace Eigen;

    /*

    Make our matrix P for flipping rows and columns.

    [0 0 1]
    [0 1 0]
    [1 0 0]

    */

    Matrix P = Matrix::Identity().rowwise().reverse();
    Matrix reversedRows = P * input;

    Eigen::HouseholderQR<Matrix> qr(reversedRows.transpose());

    this->q = P * qr.householderQ().transpose();

    Matrix upperTriangular =
        qr.matrixQR().template triangularView<Eigen::Upper>();

    this->r = P * upperTriangular.transpose() * P;

    this->r = (this->r.array().abs() < 1e-12).select(0.0, this->r);
    this->q = (this->q.array().abs() < 1e-12).select(0.0, this->q);

    MakePositiveDiagonal(this->r, this->q, 0.0);
}


CameraParameters CameraParameters::FromMatrix(
    double pixelSize_um,
    const Matrix &input)
{
    CameraParameters result{};

    // Use the SVD to extract translation.
    Eigen::JacobiSVD<Matrix> svd(
        input,
        Eigen::ComputeFullU | Eigen::ComputeFullV);

    Eigen::Vector4d translation = svd.matrixV().col(3);
    translation.array() /= translation(3);

    // Use RQ Decomposition to extract intrinsics/extrinsics.
    Eigen::Matrix<double, 3, 3> rotationAndIntrinsics = input.block<3, 3>(0, 0);

    // auto decompose = GivensRqDecompose::Decompose(rotationAndIntrinsics);
    auto decompose = RqDecompose(rotationAndIntrinsics);

    Eigen::Matrix3d recompose = decompose.r * decompose.q;

    if (!recompose.isApprox(rotationAndIntrinsics))
    {
        throw std::runtime_error("decompose failed");
    }

    Eigen::Matrix3d K = decompose.r;
    K.array() /= K(2, 2);

    result.intrinsics =
        Intrinsics<double>::FromArray_pixels(pixelSize_um, K);

    Eigen::Matrix3d backToWorld =
        decompose.q.inverse()
        * tau::CameraBodyInImage<double>();

    Orthonormalize(backToWorld);

    result.pose.point_m = tau::Point3d<double>(translation.head<3>());

    result.pose.rotation =
        tau::RotationAngles<double>(backToWorld, tau::AxisOrder{2, 1, 0});

    return result;
}


} // end namespace ray
