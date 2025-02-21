from skbuild import setup
from setuptools import find_packages

import os, sys

import os
import sys
import subprocess

# def find_executable(name):
#     """指定したコマンドのフルパスを取得する"""
#     try:
#         return subprocess.check_output(["which", name], text=True).strip()
#     except subprocess.CalledProcessError:
#         return None

if sys.platform == "darwin":  # macOS
    TBB_PREFIX = os.popen("brew --prefix tbb").read().strip()
    TBB_DIR = os.path.join(TBB_PREFIX, "lib/cmake/tbb")
    CC = os.environ.get("CMAKE_C_COMPILER", "/opt/homebrew/bin/gcc-13")
    CXX = os.environ.get("CMAKE_CXX_COMPILER", "/opt/homebrew/bin/g++-13")

elif sys.platform.startswith("linux"):
    TBB_DIR = os.environ.get("TBB_DIR", "/usr/lib/x86_64-linux-gnu/cmake/TBB")
    CC = "gcc"
    CXX = "g++"
elif sys.platform == "win32":
    VCPKG_ROOT = os.environ.get("VCPKG_ROOT", "C:/vcpkg")
    TBB_DIR = os.path.join(VCPKG_ROOT, "installed", "x64-windows", "share", "tbb")
    # WindowsでGCCを確実に使うためのパス設定
    CC = os.environ.get("CMAKE_C_COMPILER", "C:/mingw64/bin/gcc.exe")
    CXX = os.environ.get("CMAKE_CXX_COMPILER", "C:/mingw64/bin/g++.exe")
else:
    raise RuntimeError("Unsupported OS")

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
        f"-DCMAKE_C_COMPILER={CC}",
        f"-DCMAKE_CXX_COMPILER={CXX}",
        "-DBUILD_TESTS=OFF",
        "-DBUILD_GMOCK=OFF",
        "-DINSTALL_GTEST=OFF",
        f"-DTBB_DIR={TBB_DIR}",
        f"-DCMAKE_PREFIX_PATH={TBB_DIR}",
    ],
    include_package_data=True,  # .so を含める
)
