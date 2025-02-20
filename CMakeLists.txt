cmake_minimum_required(VERSION 3.15)
project(VeloxML LANGUAGES CXX)

# ------------------------------------------------------------------------------
# C++ 標準の設定
# ------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frtti")

# テストビルドのON/OFF切替
option(BUILD_TESTS "Build unit tests" ON)

# SIMD 命令を有効化（Mac では Clang 用のオプションを適用）
if (CMAKE_SYSTEM_PROCESSOR MATCHES "arm" OR CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
    message(STATUS "Using ARM NEON (AArch64) settings.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -ffast-math -ftree-vectorize")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(STATUS "Using GCC with AVX2 settings.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mavx2 -mfma -ffast-math")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(STATUS "Using Clang with AVX2 settings.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mavx2 -mfma -ffast-math")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    message(STATUS "Using MSVC with AVX2 settings.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2 /fp:fast")
endif()

# ------------------------------------------------------------------------------
# OpenMP 設定
# ------------------------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang")
    message(STATUS "Using AppleClang")
    # message(STATUS "Using Clang: enabling llvm profiling and vectorizer verbose options")
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-instr-generate -fcoverage-mapping -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize -fxray-instrument")
    find_package(OpenMP)
    if(NOT OpenMP_CXX_FOUND)
        message(STATUS "OpenMP が自動検出できませんでした。AppleClang 用に手動設定を適用します。")
        # AppleClangの場合、-Xpreprocessor と -fopenmp を個別の引数として指定
        set(OpenMP_CXX_FLAGS "-Xpreprocessor;-fopenmp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        # Homebrewでインストールされた libomp のパスを指定（バージョンに合わせて変更）
        set(OpenMP_omp_LIBRARY "/opt/homebrew/Cellar/libomp/19.1.7/lib/libomp.dylib")
        # ここで omp.h があるディレクトリを指定
        set(OpenMP_INCLUDE_DIR "/opt/homebrew/Cellar/libomp/19.1.7/include")
        set(OpenMP_CXX_FOUND TRUE)
    endif()
    if(NOT TARGET OpenMP::OpenMP_CXX)
        add_library(OpenMP::OpenMP_CXX INTERFACE IMPORTED)
        set_target_properties(OpenMP::OpenMP_CXX PROPERTIES
            INTERFACE_COMPILE_OPTIONS "${OpenMP_CXX_FLAGS}"
            INTERFACE_LINK_LIBRARIES "${OpenMP_omp_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${OpenMP_INCLUDE_DIR}"
        )
    endif()
    message(STATUS "Using OpenMP compile options: ${OpenMP_CXX_FLAGS}")
    message(STATUS "Using OpenMP library: ${OpenMP_omp_LIBRARY}")
    message(STATUS "Using OpenMP include dir: ${OpenMP_INCLUDE_DIR}")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    message(STATUS "Using GNU GCC")
    find_package(OpenMP REQUIRED)
    add_compile_options(-fopenmp)
    add_link_options(-fopenmp)
    # gprof用フラグを追加（C++ コンパイラ）
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(STATUS "Using Clang")
    find_package(OpenMP)
    if(OpenMP_FOUND)
        add_compile_options(-fopenmp)
        add_link_options(-fopenmp)
    endif()
endif()
# ------------------------------------------------------------------------------
# TBB 設定
# ------------------------------------------------------------------------------
find_package(TBB CONFIG REQUIRED)

if(NOT TARGET TBB::tbb)
    message(FATAL_ERROR "TBB target not found!")
else()
    message(STATUS "TBB Found! ${}")
endif()

# TBB のヘッダーパスを追加
include_directories(SYSTEM "/opt/homebrew/include")


# ------------------------------------------------------------------------------
# AddressSanitizer 設定（Clang, GCC 両方に対応）
# ------------------------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(STATUS "Enabling AddressSanitizer")
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
endif()

# Python3 設定
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
if(Python3_FOUND)
    message(STATUS "Found Python3: ${Python3_EXECUTABLE}")
    message(STATUS "Python Include Dir: ${Python3_INCLUDE_DIRS}")
    message(STATUS "Python Library: ${Python3_LIBRARIES}")
else()
    message(FATAL_ERROR "Python3 not found! Please install Python3 development headers.")
endif()

# OpenBLAS 設定（Mac 用）
set(OPENBLAS_INCLUDE_DIR "/opt/homebrew/Cellar/openblas/0.3.29/include")
set(OPENBLAS_LIB "/opt/homebrew/Cellar/openblas/0.3.29/lib/libopenblas.dylib")
if (EXISTS ${OPENBLAS_LIB})
    message(STATUS "Found OpenBLAS: ${OPENBLAS_LIB}")
else()
    message(FATAL_ERROR "OpenBLAS not found! Install it using 'brew install openblas'")
endif()
execute_process(COMMAND which python OUTPUT_VARIABLE PYTHON_EXECUTABLE OUTPUT_STRIP_TRAILING_WHITESPACE)
# ------------------------------------------------------------------------------
# 外部コンテンツ: pybind11 の取得
# ------------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.6
)
FetchContent_MakeAvailable(pybind11)

# ------------------------------------------------------------------------------
# ソースファイルの収集
# ------------------------------------------------------------------------------

file(GLOB BUILD_SOURCES "src/*.cpp" "src/*/*.cpp" "scr/*/binding/*.cpp")

add_library(VeloxML SHARED ${BUILD_SOURCES})
target_include_directories(VeloxML PUBLIC
    ${Python3_INCLUDE_DIRS}
    include
    ${OPENBLAS_INCLUDE_DIR}
    tests/testutils
)
set_property(TARGET VeloxML PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(VeloxML PUBLIC
    OpenMP::OpenMP_CXX
    TBB::tbb
    ${OPENBLAS_LIB}
    pybind11::module
    ${Python3_LIBRARIES}
    "-framework CoreFoundation"
)
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -undefined dynamic_lookup")

# ------------------------------------------------------------------------------
# Python バインドのビルド設定
# ------------------------------------------------------------------------------
set(PYBIND11_FINDPYTHON ON)
# pybind11 を使って C++ モジュールをビルド
pybind11_add_module(c_veloxml_core ${BUILD_SOURCES} "${CMAKE_SOURCE_DIR}/src/bindings.cpp")

# .so ファイルの出力ディレクトリを `_skbuild` 内の適切なディレクトリにする
set_target_properties(c_veloxml_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"  # ここを修正
)

# Include パス & ライブラリを設定
target_include_directories(c_veloxml_core PRIVATE include ${OPENBLAS_INCLUDE_DIR})
target_link_libraries(c_veloxml_core PRIVATE ${OPENBLAS_LIB} TBB::tbb)
target_link_libraries(c_veloxml_core PRIVATE ${LAPACK_LIBRARIES})

# Google Test を明示的にインストールしないようにする
if(TARGET gtest OR TARGET gtest_main)
    message(STATUS "Excluding Google Test from installation")
    set_target_properties(gtest PROPERTIES EXCLUDE_FROM_ALL TRUE EXCLUDE_FROM_DEFAULT_BUILD TRUE)
    set_target_properties(gtest_main PROPERTIES EXCLUDE_FROM_ALL TRUE EXCLUDE_FROM_DEFAULT_BUILD TRUE)
endif()

install(TARGETS c_veloxml_core LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/veloxml/core")

# ビルド
add_executable(MyExecutable src/main.cpp)
target_link_libraries(MyExecutable PRIVATE VeloxML)
# target_link_libraries(${VELOXML_MODULE_NAME} PUBLIC pybind11::headers)

if(BUILD_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
    )
    FetchContent_MakeAvailable(googletest)
    
    enable_testing()
    add_library(TestUtils STATIC tests/testutils/test_data_utils.cpp)
    target_include_directories(TestUtils PRIVATE ${CMAKE_SOURCE_DIR}/src, ${OPENBLAS_INCLUDE_DIR})
    
    file(GLOB TEST_SOURCES
        "tests/*.cpp"
        "tests/*/*.cpp"
    )
    add_executable(VeloxMLTest ${TEST_SOURCES})
    target_link_libraries(VeloxMLTest PRIVATE
        VeloxML
        TestUtils
        gtest
        gtest_main
    )
    add_test(NAME VeloxMLTest COMMAND VeloxMLTest)
endif()
