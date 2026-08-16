from setuptools import setup

from mlx import extension


if __name__ == "__main__":
    setup(
        name="mtplx_native_mlp",
        version="0.0.0",
        description="Native MLX VerifyCore MLP kernel probes for MTPLX.",
        ext_modules=[extension.CMakeExtension("mtplx_native_mlp._ext")],
        cmdclass={"build_ext": extension.CMakeBuild},
        packages=["mtplx_native_mlp"],
        package_data={"mtplx_native_mlp": ["*.so", "*.dylib", "*.metallib"]},
        zip_safe=False,
        python_requires=">=3.11",
    )
