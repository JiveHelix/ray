#include <catch2/catch.hpp>
#include <jive/equal.h>

#include <tau/literals.h>
#include <ray/intrinsics.h>
#include <ray/pose.h>
#include <ray/projection.h>
#include <ray/homography.h>


using namespace tau::literals;
using Distortion = ray::distortion::BrownConrady<double>;


ray::NamedVertices CreateNamedVertices(
    double x_m,
    double squareSize_mm,
    const ray::Intrinsics<double> &intrinsics,
    const ray::Pose<double> &pose,
    const Distortion &distortion = {0, 0, 0, 0, 0})
{
    ray::Projection projection(intrinsics, pose);
    auto intrinsicsArray = intrinsics.GetArray_pixels();
    auto intrinsicsInverse = intrinsics.GetInverse_pixels();

    ray::NamedVertices vertices;
    ray::NamedVertex current{};

    double squareSize_m = squareSize_mm * 1e-3;

    double startingY = 0.1;
    double startingZ = 0.075;

    for (size_t i = 0; i < 8; ++i)
    {
        current.logical.x = i;

        for (size_t j = 0; j < 6; ++j)
        {
            current.logical.y = j;

            tau::Vector3d<double> world(
                x_m,
                startingY - static_cast<double>(i) * squareSize_m,
                startingZ - static_cast<double>(j) * squareSize_m);

            tau::Vector3<double> sensor = projection.WorldToImage(world);

            Eigen::Vector3<double> camera =
                intrinsicsInverse
                * Eigen::Vector3<double>(sensor(0), sensor(1), 1);

            camera.array() /= camera(2);

            auto point = tau::Point2d<double>(camera.template head<2>());

            auto distorted =
                ray::distortion::DistortPoint(
                    distortion,
                    point);

            Eigen::Vector3<double> projected =
                intrinsicsArray
                * Eigen::Vector3<double>(distorted.x, distorted.y, 1);

            current.pixel.x = projected(0);
            current.pixel.y = projected(1);

            vertices.push_back(current);
        }
    }

    return vertices;
}


using Solutions = std::vector<ray::NamedVertices>;


class SolutionCreator
{
public:
    SolutionCreator(
        double squareSize_mm,
        const ray::Intrinsics<double> &intrinsics,
        const Distortion &distortion =
            {0, 0, 0, 0, 0})
        :
        squareSize_mm_(squareSize_mm),
        intrinsics_(intrinsics),
        distortion_(distortion)
    {

    }

    ray::NamedVertices CreateSolution(
        double x_deg,
        double y_deg,
        double z_deg,
        double virtualZ_m)
    {
        // Create the pose of the camera in the world.
        // The camera is rotated relative to the space where the virtual
        // vertices will be placed.
        // Compute the translation of the camera such that the chess board
        // remains centered in the projected view.
        ray::Pose<double> pose(
            {x_deg, y_deg, z_deg},
            0_d,
            0_d,
            0_d);

        // Positive rotation about y makes the camera look down.
        // Raise the camera to keep the vertices in view.
        pose.point_m.z =
            virtualZ_m * std::tan(tau::ToRadians(pose.rotation.pitch_deg));

        // Positive rotation about z makes the camera look left.
        // Translate to the right (-y) to compensate.
        pose.point_m.y =
            -std::tan(tau::ToRadians(pose.rotation.yaw_deg)) * virtualZ_m;

        ray::NamedVertices solution =
            CreateNamedVertices(
                virtualZ_m,
                this->squareSize_mm_,
                this->intrinsics_,
                pose,
                this->distortion_);

        return solution;
    }

    double squareSize_mm_;
    ray::Intrinsics<double> intrinsics_;
    Distortion distortion_;
};


Solutions CreateDegenerateSolutions(
    double squareSize_mm,
    const ray::Intrinsics<double> &intrinsics)
{
    Solutions solutions;

    SolutionCreator creator(squareSize_mm, intrinsics);

    solutions.push_back(
        creator.CreateSolution(0, 0, 0, 2));

    solutions.push_back(
        creator.CreateSolution(0, 0, 15, 1.9));

    solutions.push_back(
        creator.CreateSolution(0, 0, -17, 2.1));

    solutions.push_back(
        creator.CreateSolution(0, 0, -10, 2.05));

    solutions.push_back(
        creator.CreateSolution(0, 0, 11, 1.95));

    return solutions;
}


Solutions CreateSolutions(
    double squareSize_mm,
    const ray::Intrinsics<double> &intrinsics)
{
    Solutions solutions;

    SolutionCreator creator(squareSize_mm, intrinsics);

    solutions.push_back(
        creator.CreateSolution(0, 0, 0, 2));

    solutions.push_back(
        creator.CreateSolution(8, -6, 15, 1.9));

    solutions.push_back(
        creator.CreateSolution(-7, 9, -17, 2.1));

    solutions.push_back(
        creator.CreateSolution(5, 11, -10, 2.05));

    solutions.push_back(
        creator.CreateSolution(-6, -8, 11, 1.95));

    return solutions;
}


TEST_CASE("HomographyMatrix round trip", "[homography]")
{
    ray::Intrinsics<double> intrinsics{{
        10_d,
        25_d,
        25_d,
        1920.0_d / 2.0_d,
        1080.0_d / 2.0_d,
        0_d}};

    auto homographySettings = ray::HomographySettings{};

    auto solution =
        SolutionCreator(
            homographySettings.squareSize_mm,
            intrinsics).CreateSolution(0, 0, 0, 2);

    auto homography = ray::Homography(homographySettings);

    ray::HomographyMatrix homographyMatrix
        = homography.GetHomographyMatrix(solution);

    auto normalize = tau::NormalizePixel(homographySettings.sensorSize_pixels);
    auto world = ray::World(homographySettings.squareSize_mm);

    for (auto &vertex: solution)
    {
        auto worldPoint = world(vertex.logical);
        auto pixel = normalize(vertex.pixel);

        tau::Vector3<double> pixelH(pixel.x, pixel.y, 1);
        tau::Vector3<double> worldH(worldPoint.x, worldPoint.y, 1);
        tau::Vector3<double> projected = homographyMatrix * worldH;
        projected.array() /= projected(2);

        if (!projected.isApprox(pixelH))
        {
            std::cout << "projected:\n" << projected << "\n!=\n" << pixelH << std::endl;
        }

        REQUIRE(projected.isApprox(pixelH));
    }
}


TEST_CASE("Test intrinsics solver for degenate case", "[homography]")
{
    ray::Intrinsics<double> intrinsics{{
        10_d,
        25_d,
        25_d,
        1920.0_d / 2.0_d,
        1080.0_d / 2.0_d,
        0_d}};

    auto homographySettings = ray::HomographySettings{};

    auto solutions =
        CreateDegenerateSolutions(homographySettings.squareSize_mm, intrinsics);

    auto homography = ray::Homography(homographySettings);

    REQUIRE_THROWS(homography.EstimateIntrinsics(solutions));
}


TEST_CASE("Solve for intrinsics", "[homography]")
{
    ray::Intrinsics<double> expectedIntrinsics{{
        10_d,
        25_d,
        25_d,
        1920.0_d / 2.0_d,
        1080.0_d / 2.0_d,
        0_d}};

    auto homographySettings = ray::HomographySettings{};

    auto solutions =
        CreateSolutions(homographySettings.squareSize_mm, expectedIntrinsics);

    auto homography = ray::Homography(homographySettings);

    ray::IntrinsicsMatrix result =
        homography.EstimateIntrinsics(solutions);

    std::cout << "Invented:\n" << expectedIntrinsics << std::endl;
    std::cout << "\nComputed:\n" << result << std::endl;

    auto intrinsics = ray::Intrinsics<double>::FromArray_pixels(10_d, result);
    std::cout << intrinsics << std::endl;

    REQUIRE(intrinsics.focalLengthX_mm == Approx(25_d));
}


TEST_CASE("Solve for zero distortion", "[homography]")
{
    ray::Intrinsics<double> intrinsics{{
        10_d,
        25_d,
        25_d,
        1920.0_d / 2.0_d,
        1080.0_d / 2.0_d,
        0_d}};

    auto homographySettings = ray::HomographySettings{};

    Solutions solutions;
    SolutionCreator creator(homographySettings.squareSize_mm, intrinsics);

    solutions.push_back(creator.CreateSolution(0, 0, 0, 2));
    solutions.push_back(creator.CreateSolution(8, -6, 15, 1.9));
    solutions.push_back(creator.CreateSolution(-7, 9, -17, 2.1));
    solutions.push_back(creator.CreateSolution(5, 11, -10, 2.05));
    solutions.push_back(creator.CreateSolution(-6, -8, 11, 1.95));

    auto homography = ray::Homography(homographySettings);

    auto minimized =
        homography.RefineIntrinsics(
            intrinsics.GetArray_pixels(),
            solutions);

    Distortion distortion = minimized.lensCalibration.distortion;

    REQUIRE(distortion.k1 == Approx(0.0).margin(1e-7));
    REQUIRE(distortion.k2 == Approx(0.0).margin(1e-7));
    REQUIRE(distortion.p1 == Approx(0.0).margin(1e-7));
    REQUIRE(distortion.p2 == Approx(0.0).margin(1e-7));
    REQUIRE(distortion.k3 == Approx(0.0).margin(1e-7));
    REQUIRE(minimized.rmsReprojectionError_pixels == Approx(0.0).margin(1e-7));

    auto intrinsicsMatrix =
        minimized.lensCalibration.intrinsics.GetArray_pixels();

    REQUIRE(
        intrinsicsMatrix(0, 1)
            == Approx(0.0).margin(1e-12));

    REQUIRE(
        intrinsicsMatrix(0, 0)
            == Approx(intrinsics.GetArray_pixels()(0, 0)));

    REQUIRE(
        intrinsicsMatrix(1, 1)
            == Approx(intrinsics.GetArray_pixels()(1, 1)));
}


TEST_CASE("Jointly solve intrinsics and distortion", "[homography]")
{
    ray::Intrinsics<double> expectedIntrinsics{{
        10_d,
        25_d,
        25_d,
        1920.0_d / 2.0_d,
        1080.0_d / 2.0_d,
        0_d}};

    Distortion expectedDistortion{-0.05, 0.01, 0.001, -0.0005, 0.0};

    auto homographySettings = ray::HomographySettings{};

    Solutions solutions;

    SolutionCreator creator(
        homographySettings.squareSize_mm,
        expectedIntrinsics,
        expectedDistortion);

    solutions.push_back(creator.CreateSolution(0, 0, 0, 2));
    solutions.push_back(creator.CreateSolution(8, -6, 15, 1.9));
    solutions.push_back(creator.CreateSolution(-7, 9, -17, 2.1));
    solutions.push_back(creator.CreateSolution(5, 11, -10, 2.05));
    solutions.push_back(creator.CreateSolution(-6, -8, 11, 1.95));

    auto homography = ray::Homography(homographySettings);

    auto initialIntrinsics = homography.EstimateIntrinsics(solutions);

    auto minimized =
        homography.RefineIntrinsics(initialIntrinsics, solutions);

    auto intrinsicsMatrix =
        minimized.lensCalibration.intrinsics.GetArray_pixels();

    REQUIRE(intrinsicsMatrix(0, 1) == Approx(0.0).margin(1e-12));

    REQUIRE(
        intrinsicsMatrix(0, 0)
            == Approx(expectedIntrinsics.GetArray_pixels()(0, 0))
                .epsilon(0.02));

    REQUIRE(
        intrinsicsMatrix(1, 1)
            == Approx(expectedIntrinsics.GetArray_pixels()(1, 1))
                .epsilon(0.02));

    REQUIRE(
        minimized.lensCalibration.distortion.k1
            == Approx(expectedDistortion.k1).margin(0.02));

    REQUIRE(minimized.rmsReprojectionError_pixels < 1e-6);
}
