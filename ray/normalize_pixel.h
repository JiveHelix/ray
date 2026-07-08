#pragma once


#include <tau/size.h>
#include <tau/vector2d.h>


namespace tau
{


class NormalizePixel
{
public:
    NormalizePixel(const tau::Size<double> &sensorSize)
        :
        sensorSize_(sensorSize)
    {

    }

    tau::Point2d<double> operator()(const tau::Point2d<double> &pixel) const
    {
        return this->ToNormalized(pixel);
    }

    tau::Point2d<double> ToNormalized(const tau::Point2d<double> &pixel) const
    {
        // Scale from 0 to 2.
        tau::Point2d<double> result = pixel * 2 / this->sensorSize_;

        // Shift center point to 0.
        return result - 1;
    }

    tau::Point2d<double> ToPixel(const tau::Point2d<double> &normalized) const
    {
        tau::Point2d<double> unshifted = normalized;

        // Shift center point back to 1, 1
        unshifted += 1;

        // Scale back to sensor size.
        return unshifted * this->sensorSize_ / 2.0;
    }

    double ToPixel(double normalized, bool isX) const
    {
        double unshifted = normalized;

        // Shift center point back to 1, 1
        unshifted += 1;

        // Scale back to sensor size.
        if (isX)
        {
            return unshifted * this->sensorSize_.width / 2.0;
        }
        else
        {
            return unshifted * this->sensorSize_.height / 2.0;
        }
    }

    double Unscale(double normalized, bool isX) const
    {
        if (isX)
        {
            return normalized * this->sensorSize_.width / 2.0;
        }
        else
        {
            return normalized * this->sensorSize_.height / 2.0;
        }
    }

private:
    tau::Size<double> sensorSize_;
};


} // end namespace tau
