from skbuild import setup
from setuptools import find_packages

setup(
    name="veloxml",
    version="0.0.1",
    description="High-Performance Machine Learning Library for Python (Powered by C++)",
    author="Yuji Chinen",
    author_email="veloxml1113@gmail.com",
    license="MIT",
    packages=find_packages(where=".", include=["veloxml", "veloxml.*"]),
    package_dir={".": "veloxml"},  
    cmake_install_dir=".",  
    install_requires=[
        "numpy",
        "pybind11",
    ],
    cmake_args=[
        "-DCMAKE_C_COMPILER=/usr/local/bin/gcc",
        "-DCMAKE_CXX_COMPILER=/usr/local/bin/g++",
        "-DBUILD_TESTS=OFF",
        "-DBUILD_GMOCK=OFF",
        "-DINSTALL_GTEST=OFF",
    ],
    include_package_data=True,  # .so を含める
)
