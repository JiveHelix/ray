#include <catch2/catch.hpp>
#include <jive/equal.h>

#include <tau/literals.h>
#include <ray/intrinsics.h>
#include <ray/pose.h>
#include <ray/projection.h>
#include <ray/homography.h>


using namespace tau::literals;
using Distortion = ray::distortion::BrownConrady<double>;


// chessBoardOffset_m
//    x, y, and z in world coordinates
//      positive x extends away from the camera,
//      positive y extends to the left,
//      positive z extends up

ray::PlanarVertices CreatePlanarVertices(
    const tau::Point3d<double> &chessBoardOffset_m,
    const tau::Size<size_t> &vertexCount,
    const ray::HomographySettings &homographySettings,
    const ray::Intrinsics<double> &intrinsics,
    const ray::Pose<double> &pose,
    const Distortion &distortion = {0, 0, 0, 0, 0})
{
    ray::Projection projection(intrinsics, pose);
    auto intrinsicsArray = intrinsics.GetArray_pixels();
    auto intrinsicsInverse = intrinsics.GetInverse_pixels();

    ray::PlanarVertices vertices;
    ray::NamedVertex current{};

    static constexpr auto metersPerMillimeter = 1e-3;

    double squareSize_m =
        homographySettings.squareSize_mm * metersPerMillimeter;

    // Subtract one for the last fence post.
    auto boardSize = (vertexCount.template Cast<double>() - 1.0) * squareSize_m;

    double startingY = chessBoardOffset_m.y + (boardSize.width / 2.0);
    double startingZ = chessBoardOffset_m.z + (boardSize.height/ 2.0);

    for (size_t i = 0; i < vertexCount.width; ++i)
    {
        current.logical.x = i;

        for (size_t j = 0; j < vertexCount.height; ++j)
        {
            current.logical.y = j;

            tau::Vector3d<double> world(
                chessBoardOffset_m.x,
                startingY - static_cast<double>(i) * squareSize_m,
                startingZ - static_cast<double>(j) * squareSize_m);

            tau::Vector3<double> sensor = projection.WorldToImage(world);

            Eigen::Vector3<double> camera =
                intrinsicsInverse
                * Eigen::Vector3<double>(sensor(0), sensor(1), 1);

            camera.array() /= camera(2);

            auto point = tau::Point2d<double>(camera.template head<2>());

            auto distorted = distortion.Apply(point);

            Eigen::Vector3<double> projected =
                intrinsicsArray
                * Eigen::Vector3<double>(distorted.x, distorted.y, 1);

            current.pixel.x = projected(0);
            current.pixel.y = projected(1);

            // Check visibility and cull points that project outside of the
            // image sensor.
            auto maxX = homographySettings.sensorSize_pixels.width - 1;
            auto maxY = homographySettings.sensorSize_pixels.height - 1;

            if (
                current.pixel.x >= 0.0
                && current.pixel.y >= 0.0
                && current.pixel.x <= maxX
                && current.pixel.y <= maxY)
            {
                vertices.push_back(current);
            }
        }
    }

    return vertices;
}


using Solutions = std::vector<ray::PlanarVertices>;


class SolutionCreator
{
public:
    SolutionCreator(
        const ray::HomographySettings &homographySettings,
        const tau::Size<size_t> &vertexCount,
        const ray::Intrinsics<double> &intrinsics,
        const Distortion &distortion =
            {0, 0, 0, 0, 0})
        :
        homographySettings_(homographySettings),
        vertexCount_(vertexCount),
        intrinsics_(intrinsics),
        distortion_(distortion)
    {

    }

    ray::PlanarVertices CreateSolution(
        double x_deg,
        double y_deg,
        double z_deg,
        const tau::Point3d<double> &chessBoardOffset_m)
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
            chessBoardOffset_m.x
                * std::tan(tau::ToRadians(pose.rotation.pitch_deg));

        // Positive rotation about z makes the camera look left.
        // Translate to the right (-y) to compensate.
        pose.point_m.y =
            std::tan(tau::ToRadians(pose.rotation.yaw_deg))
                * chessBoardOffset_m.x;

        return CreatePlanarVertices(
            chessBoardOffset_m,
            this->vertexCount_,
            this->homographySettings_,
            this->intrinsics_,
            pose,
            this->distortion_);
    }

    ray::HomographySettings homographySettings_;
    tau::Size<size_t> vertexCount_;
    ray::Intrinsics<double> intrinsics_;
    Distortion distortion_;
};


Solutions CreateDegenerateSolutions(
    const ray::HomographySettings &homographySettings,
    const ray::Intrinsics<double> &intrinsics)
{
    Solutions solutions;

    SolutionCreator creator(homographySettings, {8, 6}, intrinsics);

    solutions.push_back(
        creator.CreateSolution(0, 0, 0, {2, 0, 0}));

    solutions.push_back(
        creator.CreateSolution(0, 0, 15, {1.9, 0, 0}));

    solutions.push_back(
        creator.CreateSolution(0, 0, -17, {2.1, 0, 0}));

    solutions.push_back(
        creator.CreateSolution(0, 0, -10, {2.05, 0, 0}));

    solutions.push_back(
        creator.CreateSolution(0, 0, 11, {1.95, 0, 0}));

    return solutions;
}


Solutions CreateSolutions(
    const ray::HomographySettings &homographySettings,
    const ray::Intrinsics<double> &intrinsics)
{
    Solutions solutions;

    SolutionCreator creator(homographySettings, {8, 6}, intrinsics);

    solutions.push_back(
        creator.CreateSolution(0, 0, 0, {2, 0, 0}));

    solutions.push_back(
        creator.CreateSolution(8, -6, 15, {1.9, 0, 0}));

    solutions.push_back(
        creator.CreateSolution(-7, 9, -17, {2.1, 0, 0}));

    solutions.push_back(
        creator.CreateSolution(5, 11, -10, {2.05, 0, 0}));

    solutions.push_back(
        creator.CreateSolution(-6, -8, 11, {1.95, 0, 0}));

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
            homographySettings,
            {8, 6},
            intrinsics).CreateSolution(0, 0, 0, {2, 0, 0});

    auto homography = ray::Homography(homographySettings);
    auto normalize = ray::NormalizePixel(homographySettings.sensorSize_pixels);

    ray::HomographyMatrix homographyMatrix =
        homography.GetHomographyMatrix(GetNormalized(normalize, solution));

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
            std::cout << "projected:\n" << projected << "\n!=\n"
                << pixelH << std::endl;
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
        CreateDegenerateSolutions(homographySettings, intrinsics);

    auto homography = ray::Homography(homographySettings);

    REQUIRE_THROWS(
        homography.EstimateIntrinsics(
            GetNormalized(homography.GetNormalizePixel(), solutions)));
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
        CreateSolutions(homographySettings, expectedIntrinsics);

    auto homography = ray::Homography(homographySettings);

    ray::IntrinsicsMatrix result =
        homography.GetNormalizePixel().ToPixels(
            homography.EstimateIntrinsics(
                GetNormalized(homography.GetNormalizePixel(), solutions)));

    auto intrinsics = ray::Intrinsics<double>::FromArray_pixels(10_d, result);

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
    SolutionCreator creator(homographySettings, {8, 6}, intrinsics);

    solutions.push_back(creator.CreateSolution(0, 0, 0, {2, 0, 0}));
    solutions.push_back(creator.CreateSolution(8, -6, 15, {1.9, 0, 0}));
    solutions.push_back(creator.CreateSolution(-7, 9, -17, {2.1, 0, 0}));
    solutions.push_back(creator.CreateSolution(5, 11, -10, {2.05, 0, 0}));
    solutions.push_back(creator.CreateSolution(-6, -8, 11, {1.95, 0, 0}));

    auto homography = ray::Homography(homographySettings);

    const auto &normalize = homography.GetNormalizePixel();

    auto minimized =
        homography.RefineIntrinsics(
            normalize.ToNormalized(intrinsics.GetArray_pixels()),
            GetNormalized(normalize, solutions));

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
        intrinsicsMatrix(0, 1) == Approx(0.0).margin(1e-12));

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

    Distortion expectedDistortion{-0.05, 0.01, 0.001, -0.0005, 0.002};

    auto homographySettings = ray::HomographySettings{};

    Solutions solutions;

    SolutionCreator creator(
        homographySettings,
        {8, 6},
        expectedIntrinsics,
        expectedDistortion);

    solutions.push_back(creator.CreateSolution(0, 0, 0, {2, 0, 0}));
    solutions.push_back(creator.CreateSolution(8, -6, 15, {1.9, 0.2, -0.1}));
    solutions.push_back(creator.CreateSolution(-7, 9, -17, {2.1, -0.3, 0}));
    solutions.push_back(creator.CreateSolution(5, 11, -10, {2.05, -0.5, 0}));
    solutions.push_back(creator.CreateSolution(-6, -8, 11, {1.95, 0, 0.2}));
    solutions.push_back(creator.CreateSolution(-5, -7, 12, {3.0, 0.4, 0}));
    solutions.push_back(creator.CreateSolution(7, 10, -14, {1.5, 0.1, 0}));

    // Find min/max pixel coordinates of synthetic vertices
    double minX = 10000.;
    double maxX = 0.;

    double minY = 10000.;
    double maxY = 0.;

    for (const auto &planarVertices: solutions)
    {
        for (const auto &vertex: planarVertices)
        {
            minX = std::min(vertex.pixel.x, minX);
            minY = std::min(vertex.pixel.y, minY);

            maxX = std::max(vertex.pixel.x, maxX);
            maxY = std::max(vertex.pixel.y, maxY);
        }
    }

    REQUIRE(minX >= 0.0);
    REQUIRE(minY >= 0.0);
    REQUIRE(maxX <= homographySettings.sensorSize_pixels.width - 1);
    REQUIRE(maxY <= homographySettings.sensorSize_pixels.height - 1);

    auto homography = ray::Homography(homographySettings);

    auto normalizedSolutions =
        GetNormalized(homography.GetNormalizePixel(), solutions);

    auto initialIntrinsics = homography.EstimateIntrinsics(normalizedSolutions);

    auto minimized =
        homography.RefineIntrinsics(initialIntrinsics, normalizedSolutions);

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

    std::cout << fields::DescribeColorized(minimized, 1) << std::endl;

    auto distortionMargin = 1e-4;

    REQUIRE(
        minimized.lensCalibration.distortion.k1
            == Approx(expectedDistortion.k1).margin(distortionMargin));

    REQUIRE(
        minimized.lensCalibration.distortion.k2
            == Approx(expectedDistortion.k2).margin(distortionMargin));

    REQUIRE(
        minimized.lensCalibration.distortion.k3
            == Approx(expectedDistortion.k3).margin(distortionMargin));

    REQUIRE(
        minimized.lensCalibration.distortion.p1
            == Approx(expectedDistortion.p1).margin(distortionMargin));

    REQUIRE(
        minimized.lensCalibration.distortion.p2
            == Approx(expectedDistortion.p2).margin(distortionMargin));

    REQUIRE(minimized.rmsReprojectionError_pixels < 1e-5);
}
