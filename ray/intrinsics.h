#pragma once

#include <fields/fields.h>
#include <fields/compare.h>
#include <fields/formatter.h>
#include <pex/group.h>
#include <pex/identity.h>
#include <nlohmann/json.hpp>

#include <tau/eigen_shim.h>
#include <tau/vector3d.h>
#include <tau/vector2d.h>


namespace ray
{


template<typename T>
class PixelConvert
{
    T pixelSize_um_;

public:
    static constexpr auto metersPerMicron = static_cast<T>(1e-6);

    PixelConvert()
        :
        pixelSize_um_(10)
    {

    }

    T GetPixelSize_um() const
    {
        return this->pixelSize_um_;
    }

    PixelConvert(T pixelSize_um)
        :
        pixelSize_um_(pixelSize_um)
    {
        if (pixelSize_um == static_cast<T>(0))
        {
            throw std::runtime_error("Invalid pixel size");
        }
    }

    template <typename Value>
    std::enable_if_t<std::is_floating_point_v<Value>, Value>
    PixelsToMeters(const Value &pixels) const
    {
        return pixels * (this->pixelSize_um_ * metersPerMicron);
    }

    template <typename Value>
    Value PixelsToMeters(const Eigen::MatrixBase<Value> &pixels) const
    {
        return pixels.array() * (this->pixelSize_um_ * metersPerMicron);
    }

    template <typename Value>
    tau::Point3d<Value> PixelsToMeters(
        const tau::Point3d<Value> &pixels) const
    {
        return pixels * (this->pixelSize_um_ * metersPerMicron);
    }

    template <typename Value>
    std::enable_if_t<std::is_floating_point_v<Value>, Value>
    MetersToPixels(const Value &meters) const
    {
        return meters / (this->pixelSize_um_ * metersPerMicron);
    }

    template<typename Value>
    Value MetersToPixels(const Eigen::MatrixBase<Value> &meters) const
    {
        return meters.array() / (this->pixelSize_um_ * metersPerMicron);
    }

    template<typename Value>
    tau::Point3d<Value>
    MetersToPixels(const tau::Point3d<Value> &meters) const
    {
        return meters / (this->pixelSize_um_ * metersPerMicron);
    }

    template<typename U>
    PixelConvert<U> Cast() const
    {
        if constexpr (std::is_same_v<U, T>)
        {
            return *this;
        }

        return PixelConvert<U>(static_cast<U>(this->pixelSize_um_));
    }
};


template<typename T>
struct IntrinsicsFields
{
    static constexpr auto fields = std::make_tuple(
        fields::Field(&T::pixelSize_um, "pixelSize_um"),
        fields::Field(&T::focalLengthX_mm, "focalLengthX_mm"),
        fields::Field(&T::focalLengthY_mm, "focalLengthY_mm"),
        fields::Field(&T::principalX_pixels, "principalX_pixels"),
        fields::Field(&T::principalY_pixels, "principalY_pixels"),
        fields::Field(&T::skew_pixels, "skew_pixels", "skew"));
};


template<typename Float>
struct IntrinsicsTemplate
{
    template<template<typename> typename T>
    struct Template
    {
        T<Float> pixelSize_um;
        T<Float> focalLengthX_mm;
        T<Float> focalLengthY_mm;
        T<Float> principalX_pixels;
        T<Float> principalY_pixels;
        T<Float> skew_pixels;

        static constexpr auto fields =
            IntrinsicsFields<Template>::fields;

        static constexpr auto fieldsTypeName = "Intrinsics";
    };
};


template<typename T>
struct IntrinsicsAsPixelsFields
{
    static constexpr auto fields = std::make_tuple(
        fields::Field(&T::focalLengthX, "focalLengthX"),
        fields::Field(&T::focalLengthY, "focalLengthY"),
        fields::Field(&T::principalX, "principalX"),
        fields::Field(&T::principalY, "principalY"),
        fields::Field(&T::skew, "skew"));
};


template<typename Float>
struct IntrinsicsAsPixelsTemplate
{
    template<template<typename> typename T>
    struct Template
    {
        T<Float> focalLengthX;
        T<Float> focalLengthY;
        T<Float> principalX;
        T<Float> principalY;
        T<Float> skew;

        static constexpr auto fields =
            IntrinsicsAsPixelsFields<Template>::fields;

        static constexpr auto fieldsTypeName = "IntrinsicsAsPixels";
    };
};


template<typename Float>
struct IntrinsicsAsPixelsCustom
{
    template<typename Base>
    struct Plain: public Base
    {
        tau::Point2d<Float>
        ToNormalizedPixel(const tau::Point2d<Float> &pixel) const
        {
            return {
                (pixel.x - this->principalX) / this->focalLengthX,
                (pixel.y - this->principalY) / this->focalLengthY};
        }

        tau::Point2d<Float>
        ToSensorPixel(const tau::Point2d<Float> &pixel) const
        {
            return {
                pixel.x * this->focalLengthX + this->principalX,
                pixel.y * this->focalLengthY + this->principalY};
        }

        template<typename U, typename Style = tau::Round>
        auto Cast() const
        {
            using Result = IntrinsicsAsPixelsCustom<U>::template Plain
                <
                    typename IntrinsicsAsPixelsTemplate<U>
                        ::template Template<pex::Identity>
                >;

            return tau::CastFields<Result, U, Style>(*this);
        }

        using Matrix = Eigen::Matrix<Float, 3, 3>;

        static Plain FromArray(
            const Matrix &array_pixels)
        {
            Plain result{};
            result.focalLengthX = array_pixels(0, 0);
            result.focalLengthY = array_pixels(1, 1);
            result.skew = array_pixels(0, 1);
            result.principalX = array_pixels(0, 2);
            result.principalY = array_pixels(1, 2);

            return result;
        }

        Matrix GetArray() const
        {
            Matrix m{
                {this->focalLengthX, this->skew, this->principalX},
                {0.0, this->focalLengthY, this->principalY},
                {0.0, 0.0, 1.0}};

            return m;
        }
    };
};


template<typename T>
using IntrinsicsAsPixelsGroup =
    pex::Group
    <
        IntrinsicsAsPixelsFields,
        IntrinsicsAsPixelsTemplate<T>::template Template,
        IntrinsicsAsPixelsCustom<T>
    >;

template<typename T>
using IntrinsicsAsPixels = IntrinsicsAsPixelsGroup<T>::Plain;

DECLARE_OUTPUT_STREAM_OPERATOR(IntrinsicsAsPixels<float>)
DECLARE_OUTPUT_STREAM_OPERATOR(IntrinsicsAsPixels<double>)
DECLARE_EQUALITY_OPERATORS(IntrinsicsAsPixels<float>)
DECLARE_EQUALITY_OPERATORS(IntrinsicsAsPixels<double>)



template<typename T>
struct Intrinsics:
    public IntrinsicsTemplate<T>::template Template<pex::Identity>
{
    using Base =
        typename IntrinsicsTemplate<T>::template Template<pex::Identity>;

    static constexpr auto millimetersPerMeter = static_cast<T>(1e3);

    using Matrix = Eigen::Matrix<T, 3, 3>;

    Intrinsics()
        :
        Base(
            {
                static_cast<T>(10),
                static_cast<T>(25),
                static_cast<T>(25),
                static_cast<T>(1920.0 / 2.0),
                static_cast<T>(1080.0 / 2.0),
                static_cast<T>(0)})
    {

    }

    Intrinsics(const Base &base)
        :
        Base(base)
    {

    }

    Intrinsics(T pixelSize_um_)
        :
        Base{}
    {
        this->pixelSize_um = pixelSize_um_;
    }

    template<typename Value>
    auto MetersToPixels(const Value &meters) const
    {
        return PixelConvert(this->pixelSize_um).MetersToPixels(meters);
    }

    template<typename Value>
    auto PixelsToMeters(const Value &pixels) const
    {
        return PixelConvert(this->pixelSize_um).PixelsToMeters(pixels);
    }

    static Intrinsics FromArray_meters(
        T pixelSize_um_,
        const Matrix &array_meters)
    {
        Intrinsics result{};
        result.pixelSize_um = pixelSize_um_;
        auto pixelConvert = PixelConvert(pixelSize_um_);
        T focalLengthX_m = array_meters(0, 0);
        T focalLengthY_m = array_meters(1, 1);
        result.focalLengthX_mm = focalLengthX_m * millimetersPerMeter;
        result.focalLengthY_mm = focalLengthY_m * millimetersPerMeter;
        result.skew_pixels = pixelConvert.MetersToPixels(array_meters(0, 1));

        result.principalX_pixels =
            pixelConvert.MetersToPixels(array_meters(0, 2));

        result.principalY_pixels =
            pixelConvert.MetersToPixels(array_meters(1, 2));

        return result;
    }

    static Intrinsics FromArray_pixels(
        T pixelSize_um_,
        const Matrix &array_pixels)
    {
        Intrinsics result{};
        result.pixelSize_um = pixelSize_um_;
        auto pixelConvert = PixelConvert(pixelSize_um_);
        T focalLengthX_m = pixelConvert.PixelsToMeters(array_pixels(0, 0));
        T focalLengthY_m = pixelConvert.PixelsToMeters(array_pixels(1, 1));
        result.focalLengthX_mm = focalLengthX_m * millimetersPerMeter;
        result.focalLengthY_mm = focalLengthY_m * millimetersPerMeter;
        result.skew_pixels = array_pixels(0, 1);
        result.principalX_pixels = array_pixels(0, 2);
        result.principalY_pixels = array_pixels(1, 2);

        return result;
    }

    IntrinsicsAsPixels<T> GetAsPixels() const
    {
        auto pixelConvert = PixelConvert(this->pixelSize_um);

        auto focalLengthX_pixels = pixelConvert.MetersToPixels(
            this->focalLengthX_mm / millimetersPerMeter);

        auto focalLengthY_pixels = pixelConvert.MetersToPixels(
            this->focalLengthY_mm / millimetersPerMeter);

        return {
            focalLengthX_pixels,
            focalLengthY_pixels,
            this->principalX_pixels,
            this->principalY_pixels,
            this->skew_pixels};
    }

    std::string SerializeAsPixels() const
    {
        auto unstructured =
            fields::Unstructure<nlohmann::json>(this->GetAsPixels());

        return unstructured.dump(4);
    }

    Matrix GetArray_pixels() const
    {
        auto pixelConvert = PixelConvert(this->pixelSize_um);

        auto focalLengthX_pixels = pixelConvert.MetersToPixels(
            this->focalLengthX_mm / millimetersPerMeter);

        auto focalLengthY_pixels = pixelConvert.MetersToPixels(
            this->focalLengthY_mm / millimetersPerMeter);

        Matrix m{
            {focalLengthX_pixels, this->skew_pixels, this->principalX_pixels},
            {0.0, focalLengthY_pixels, this->principalY_pixels},
            {0.0, 0.0, 1.0}};

        return m;
    }

    Matrix GetArray_m() const
    {
        Matrix result = PixelConvert(this->pixelSize_um)
            .PixelsToMeters(this->GetArray_pixels());

        // Keep the bottom right value.
        result(2, 2) = 1.0;

        return result;
    }

    Matrix GetInverse_pixels() const
    {
        auto pixelConvert = PixelConvert(this->pixelSize_um);

        auto focalLengthX_pixels = pixelConvert.MetersToPixels(
            this->focalLengthX_mm / millimetersPerMeter);

        auto focalLengthY_pixels = pixelConvert.MetersToPixels(
            this->focalLengthY_mm / millimetersPerMeter);

        auto inversePrincipalX =
            (this->principalY_pixels * this->skew_pixels
             - this->principalX_pixels * focalLengthY_pixels);

        auto inversePrincipalY = -this->principalY_pixels * focalLengthY_pixels;

        auto focalProduct = focalLengthX_pixels * focalLengthY_pixels;

        Matrix m{
            {focalLengthY_pixels, -this->skew_pixels, inversePrincipalX},
            {0.0, focalLengthX_pixels, inversePrincipalY},
            {0.0, 0.0, focalProduct}};

        return m / focalProduct;
    }

    template<typename U, typename Style = tau::Round>
    Intrinsics<U> Cast() const
    {
        return tau::CastFields<Intrinsics<U>, U, Style>(*this);
    }
};


DECLARE_OUTPUT_STREAM_OPERATOR(Intrinsics<float>)
DECLARE_OUTPUT_STREAM_OPERATOR(Intrinsics<double>)
DECLARE_EQUALITY_OPERATORS(Intrinsics<float>)
DECLARE_EQUALITY_OPERATORS(Intrinsics<double>)


template<typename T>
using IntrinsicsGroup =
    pex::Group
    <
        IntrinsicsFields,
        IntrinsicsTemplate<T>::template Template,
        pex::PlainT<Intrinsics<T>>
    >;

template<typename T>
using IntrinsicsModel = typename IntrinsicsGroup<T>::Model;

template<typename T>
using IntrinsicsControl = typename IntrinsicsGroup<T>::DefaultControl;


} // end namespace ray



template<>
struct fmt::formatter<ray::Intrinsics<double>>
    : fields::Formatter<ray::Intrinsics<double>> {};

template<>
struct fmt::formatter<ray::Intrinsics<float>>
    : fields::Formatter<ray::Intrinsics<float>> {};
