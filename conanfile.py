from conan import ConanFile


class RayConan(ConanFile):
    name = "ray"
    version = "1.0.0"

    python_requires = "boiler/0.2"
    python_requires_extend = "boiler.LibraryConanFile"

    license = "MIT"
    author = "Jive Helix (jivehelix@gmail.com)"
    url = "https://github.com/JiveHelix/ray"
    description = "Camera calibration and projection geometry in C++."

    topics = (
        "Camera Calibration",
        "Distortion")

    def build_requirements(self):
        self.test_requires("catch2/2.13.9")

    def requirements(self):
        self.requires("tau/[~1.15]", transitive_headers=True)
        self.requires("nlohmann_json/[~3]", transitive_headers=True)

        self.requires(
            "ceres-solver/[~2.2]",
            transitive_headers=True,
            options={
                "use_glog": False,
                "miniglog_max_log_level": -1})
