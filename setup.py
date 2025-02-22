from skbuild import setup
from setuptools import find_packages

import os, sys

import os
import sys
import platform

# def find_executable(name):
#     """指定したコマンドのフルパスを取得する"""
#     try:
#         return subprocess.check_output(["which", name], text=True).strip()
#     except subprocess.CalledProcessError:
#         return None

if sys.platform == "darwin":  # macOS
    TBB_PREFIX = os.popen("brew --prefix tbb").read().strip()
    TBB_DIR = os.path.join(TBB_PREFIX, "lib/cmake/tbb")
    
    # 実行中のマシンのアーキテクチャを判別 (arm64: Apple Silicon, x86_64: Intel)
    machine = platform.machine()
    if machine == "arm64":
        default_cc = "/opt/homebrew/bin/gcc-13"
        default_cxx = "/opt/homebrew/bin/g++-13"
        PKG_CONFIG_PATH = "/usr/local/opt/openblas/lib/pkgconfig"

    else:
        default_cc = "gcc-13"
        default_cxx = "g++-13"
        PKG_CONFIG_PATH = "/usr/local/opt/openblas/lib/pkgconfig"

    
    # 環境変数に値が設定されていなければデフォルト値を利用
    CC = os.environ.get("CMAKE_C_COMPILER", default_cc)
    CXX = os.environ.get("CMAKE_CXX_COMPILER", default_cxx)

    # 確認用の出力
    print("TBB_DIR:", TBB_DIR)
    print("C Compiler:", CC)
    print("C++ Compiler:", CXX)

elif sys.platform.startswith("linux"):
    TBB_DIR = os.environ.get("TBB_DIR", "/usr/lib/x86_64-linux-gnu/cmake/TBB")
    CC =  os.popen("which gcc-14").read().strip()
    CXX = os.popen("which g++-14").read().strip()
    PKG_CONFIG_PATH = ""
elif sys.platform == "win32":
    VCPKG_ROOT = os.environ.get("VCPKG_ROOT", "C:/vcpkg")
    TBB_DIR = os.path.join(VCPKG_ROOT, "installed", "x64-windows", "share", "tbb")
    # WindowsでGCCを確実に使うためのパス設定
    CC = os.environ.get("CMAKE_C_COMPILER", "C:/mingw64/bin/gcc.exe")
    CXX = os.environ.get("CMAKE_CXX_COMPILER", "C:/mingw64/bin/g++.exe")
    PKG_CONFIG_PATH = ""
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
        f"-DPKG_CONFIG_PATH={PKG_CONFIG_PATH}"
    ],
    include_package_data=True,  # .so を含める
)
