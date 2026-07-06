#pragma once

#include <fields/fields.h>

#include <tau/eigen_shim.h>
#include <tau/stack.h>
#include <tau/vector2d.h>
#include <tau/line3d.h>

#include <ray/intrinsics.h>
#include <ray/pose.h>


namespace ray
{


// Project vector onto normal
template<typename T>
Eigen::Vector<T, 3> Project(
    const Eigen::Vector<T, 3> &normal,
    const Eigen::Vector<T, 3> &vector)
{
    return normal.transpose().dot(vector) * normal;
}


// Rejection of vector onto normal
// This is the projection of the vector onto the plane defined by the normal.
template<typename T>
Eigen::Vector<T, 3> Reject(
    const Eigen::Vector<T, 3> &normal,
    const Eigen::Vector<T, 3> &vector)
{
    return vector - Project(normal, vector);
}


template<typename T>
class Projection
{
public:
    Projection()
        :
        intrinsics_{},
        pose_{},
        cameraPosition_pixels_{},
        worldToImage_{},
        imageToWorld_{}
    {

    }

    Projection(
        const Intrinsics<T> &intrinsics,
        const Pose<T> &pose)
        :
        intrinsics_(intrinsics),
        pose_(pose),
        cameraPosition_pixels_(this->GetCameraPosition_pixels())
    {
        Extrinsic<T> extrinsics_m = this->pose_.GetExtrinsic_m();

        Eigen::Matrix<T, 3, 4> topThreeRows =
            extrinsics_m(Eigen::seq(0, 2), Eigen::all);

        this->worldToImage_ =
            tau::VerticalStack(
                (this->intrinsics_.GetArray_pixels() * topThreeRows).eval(),
                tau::RowVector<4, T>(T(0), T(0), T(0), T(1)));

        auto intrinsicsInverse = this->intrinsics_.GetInverse_pixels();

        this->imageToWorld_ =
            (this->pose_.GetImageInWorld() * intrinsicsInverse);
    }

    Extrinsic<T> GetExtrinsic_m() const
    {
        return this->pose_.GetExtrinsic_m();
    }

    tau::Vector3<T> WorldToImage(const tau::Point3d<T> &world) const
    {
        tau::Vector3<T> scaled =
            (this->worldToImage_ * world.GetHomogeneous())(Eigen::seq(0, 2));

        return scaled.array() / scaled(2);
    }

    tau::Point2d<T> WorldToPixel(const tau::Point3d<T> &world) const
    {
        auto homogeneous = this->WorldToImage(world);

        return tau::Point2d<T>(homogeneous.template head<2>());
    }

    tau::Vector3<T> VectorWorldToImage(const Eigen::Vector<T, 4> &world) const
    {
        tau::Vector3<T> scaled =
            (this->worldToImage_ * world)(Eigen::seq(0, 2));

        return scaled.array() / scaled(2);
    }

    // The camera only knows the direction to the world point.
    // Use GetLine to get a line that passes through the world point.
    tau::Vector3<T> ImageToWorld(const tau::Vector3<T> &camera) const
    {
        return (this->imageToWorld_ * camera);
    }

    tau::Vector3<T> PixelToWorld(const tau::Point2d<T> &pixel) const
    {
        return this->ImageToWorld(pixel.GetHomogeneous()).normalized();
    }

    tau::Point3d<T> GetCameraPosition_pixels() const
    {
        return this->pose_.GetPosition_pixels(this->intrinsics_);
    }

    tau::Line3d<T> GetLine(const tau::Point2d<T> &pixel) const
    {
        return tau::Line3d<T>(
            this->cameraPosition_pixels_,
            this->PixelToWorld(pixel));
    }

    tau::Line3d<T> GetLine_m(const tau::Point2d<T> &pixel) const
    {
        return tau::Line3d<T>(
            this->pose_.GetPosition_m(),
            this->ImageToWorld(pixel.GetHomogeneous()).normalized());
    }

    Eigen::Matrix<T, 4, 4> GetWorldToImage() const
    {
        return this->worldToImage_;
    }

    Eigen::Matrix<T, 3, 3> GetImageToWorld() const
    {
        return this->imageToWorld_;
    }

    template<typename>
    friend class Projection;

    template<typename Result, typename, typename, typename Source>
    friend Result tau::CastFields(const Source &);

    template<typename U, typename Style = tau::Round>
    Projection<U> Cast() const
    {
        return tau::CastFields<Projection<U>, U, Style>(*this);
    }

    const Intrinsics<T> & GetIntrinsics() const
    {
        return this->intrinsics_;
    }

    const Pose<T> & GetPose() const
    {
        return this->pose_;
    }

private:
    Intrinsics<T> intrinsics_;
    Pose<T> pose_;
    tau::Point3d<T> cameraPosition_pixels_;
    Eigen::Matrix<T, 4, 4> worldToImage_;
    Eigen::Matrix<T, 3, 3> imageToWorld_;

public:
    static constexpr auto fields = std::make_tuple(
        fields::Field(&Projection::intrinsics_, "intrinsics"),
        fields::Field(&Projection::pose_, "pose"),
        fields::Field(&Projection::worldToImage_, "worldToImage"),
        fields::Field(&Projection::imageToWorld_, "imageToWorld"));

};


} // end namespace ray
