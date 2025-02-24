from skbuild import setup
from setuptools import find_packages

import os, sys

import os
import sys
import platform
import sysconfig

def list_files_limited_depth(base_dir, max_depth=2, current_depth=0):
    """ 指定したディレクトリを max_depth まで再帰的に探索 """
    if current_depth >= max_depth:
        return

    with os.scandir(base_dir) as entries:
        for entry in entries:
            print("  " * current_depth + f"- {entry.name}")
            if entry.is_dir():
                list_files_limited_depth(entry.path, max_depth, current_depth + 1)

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
        default_cc = "/usr/local/bin/gcc-13"
        default_cxx = "/usr/local/bin/g++-13"
        PKG_CONFIG_PATH = "/usr/local/opt/openblas/lib/pkgconfig"

    
    # 環境変数に値が設定されていなければデフォルト値を利用
    CC = os.environ.get("CMAKE_C_COMPILER", default_cc)
    CXX = os.environ.get("CMAKE_CXX_COMPILER", default_cxx)
    PKG_CONFIG_EXEC_PATH = ""
    
    # 確認用の出力
    print("TBB_DIR:", TBB_DIR)
    print("C Compiler:", CC)
    print("C++ Compiler:", CXX)

elif sys.platform.startswith("linux"):
    machine = platform.machine()
    if machine == "aarch64" or machine == "arm64":
        TBB_DIR = "/usr/local/tbb"
    else:
        site_packages_path = sysconfig.get_paths()["purelib"]
        TBB_DIR = f"{site_packages_path}/"
    
    CC =  os.environ.get("CMAKE_C_COMPILER", "/opt/rh/gcc-toolset-14/root/usr/bin/gcc")
    CXX = os.environ.get("CMAKE_CXX_COMPILER", "/opt/rh/gcc-toolset-14/root/usr/bin/g++")
    
    PKG_CONFIG_PATH = ""
    PKG_CONFIG_EXEC_PATH = ""
    Python3_LIBRARIES = os.popen("python3 -c 'import sysconfig; print(sysconfig.get_config_var(\"LIBDIR\"))'").read().strip()
elif sys.platform == "win32":
    TBB_DIR = f"/mingw64/usr/lib/tbb"
    
    # WindowsでGCCを確実に使うためのパス設定
    # CC = "C:/msys64/mingw64/bin/gcc.exe"
    # CXX = "C:/mingw64/mingw64/bin/g++.exe"
    CC = "/mingw64/bin/gcc.exe"
    CXX = "/mingw64/bin/g++.exe"
    PKG_CONFIG_PATH = "/mingw64/openblas/lib/pkgconfig"
    PKG_CONFIG_EXEC_PATH = "/mingw64/bin/pkg-config"
else:
    raise RuntimeError("Unsupported OS")

setup(
    name="veloxml",
    version="0.0.1",
    description="High-Performance Machine Learning Library for Python (Powered by C++)",
    author="Yuji Chinen",
    author_email="veloxml1113@gmail.com",
    packages=find_packages(where=".", include=["veloxml", "veloxml.*"]),
    package_dir={".": "veloxml"},  
    cmake_install_dir=".",  
    cmake_args=[
        f"-DCMAKE_C_COMPILER={CC}",
        f"-DCMAKE_CXX_COMPILER={CXX}",
        "-DBUILD_TESTS=OFF",
        "-DBUILD_GMOCK=OFF",
        "-DINSTALL_GTEST=OFF",
        f"-DBLAS_INCLUDE_DIRS:PATH=/usr/include/openblas",   
        f"-DTBB_DIR:PATH={TBB_DIR}",
        f"-DCMAKE_PREFIX_PATH:PATH={TBB_DIR}",
        f"-DPKG_CONFIG_PATH:PATH={PKG_CONFIG_PATH}",
        f"-DPKG_CONFIG_EXECUTABLE:PATH={PKG_CONFIG_EXEC_PATH}"
    ],
    include_package_data=True,  # .so を含める
)
