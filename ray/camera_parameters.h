#pragma once

#include <tau/eigen_shim.h>
#include <tau/rotation.h>
#include <ray/pose.h>
#include <ray/intrinsics.h>


namespace ray
{


struct RqDecompose
{
    using Matrix = Eigen::Matrix3d;

    static void MakePositiveDiagonal(Matrix &R, Matrix &Q, double tolerance);
    RqDecompose(const Matrix &input);

    Matrix r;
    Matrix q;
};


template<typename Derived>
void Orthonormalize(Eigen::MatrixBase<Derived> &R)
{
    Eigen::JacobiSVD<Derived> svd(
        R,
        Eigen::ComputeFullU | Eigen::ComputeFullV);

    Derived U = svd.matrixU();
    Derived Vt = svd.matrixV().transpose();

    R = U * Vt;

    if (R.determinant() < 0.0)
    {
        // Fix improper rotation
        Eigen::Matrix3d V = svd.matrixV();
        V.col(2) *= -1.0;
        R = U * V.transpose();
    }
}


struct CameraParameters
{
    using Matrix = Eigen::Matrix<double, 3, 4>;

    Intrinsics<double> intrinsics;
    Pose<double> pose;

    static constexpr auto fields = std::make_tuple(
        fields::Field(&CameraParameters::intrinsics, "intrinsics"),
        fields::Field(&CameraParameters::pose, "pose"));

    static CameraParameters FromMatrix(
        double pixelSize_um,
        const Matrix &matrix);
};


} // end namespace ray
