#pragma once


#include <tau/size.h>
#include <tau/vector2d.h>


namespace ray
{


class NormalizePixel
{
public:
    NormalizePixel(const tau::Size<double> &sensorSize)
        :
        scale_(
            std::sqrt(2.0)
            / std::max(sensorSize.width, sensorSize.height)),

        unscale_(1.0 / this->scale_),

        transform_(),
        inverse_()
    {
        this->transform_ <<
            this->scale_, 0.0, -this->scale_ * sensorSize.width / 2.0,
            0.0, this->scale_, -this->scale_ * sensorSize.height / 2.0,
            0.0, 0.0, 1.0;

        this->inverse_ = this->transform_.inverse();
    }

    tau::Point2d<double> operator()(const tau::Point2d<double> &pixel) const
    {
        return this->ToNormalized(pixel);
    }

    tau::Point2d<double> ToNormalized(const tau::Point2d<double> &pixel) const
    {
        Eigen::Vector3d normalized =
            this->transform_ * pixel.GetHomogeneous();

        return tau::Point2d<double>(normalized.head<2>());
    }

    tau::Point2d<double> ToPixels(const tau::Point2d<double> &normalized) const
    {
        Eigen::Vector3d unscaled = this->inverse_ * normalized.GetHomogeneous();

        return tau::Point2d<double>(unscaled.head<2>());
    }

    // Scale/Unscale do not shift by 1.
    double Unscale(double normalized) const
    {
        return normalized * this->unscale_;
    }

    double GetUnscale() const
    {
        return this->unscale_;
    }

    double GetScale() const
    {
        return this->scale_;
    }

    double Scale(double pixel) const
    {
        return pixel * this->scale_;
    }

    Eigen::Matrix3d ToNormalized(const Eigen::Matrix3d &intrinsics) const
    {
        return this->transform_ * intrinsics;
    }

    Eigen::Matrix3d ToPixels(const Eigen::Matrix3d &intrinsics) const
    {
        return this->inverse_ * intrinsics;
    }

private:
    double scale_;
    double unscale_;
    Eigen::Matrix3d transform_;
    Eigen::Matrix3d inverse_;
};


} // end namespace ray
