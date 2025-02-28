cmake_minimum_required(VERSION 3.15)
project(VeloxML LANGUAGES CXX)

# ------------------------------------------------------------------------------
# C++ 標準の設定
# ------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(-DWIN32_LEAN_AND_MEAN)
    add_definitions(-D_CRT_NONSTDC_NO_WARNINGS)
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frtti")
endif()


# ------------------------------------------------------------------------------
# テストビルドのON/OFF切替
# ------------------------------------------------------------------------------
option(BUILD_TESTS "Build unit tests" OFF)

# ------------------------------------------------------------------------------
# SIMD 命令を有効化（Mac では Clang 用のオプションを適用）
# ------------------------------------------------------------------------------
if (CMAKE_SYSTEM_PROCESSOR MATCHES "arm" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch" OR CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
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
    
    find_package(OpenMP)
    if(NOT OpenMP_CXX_FOUND)
        message(STATUS "OpenMP が自動検出できませんでした。AppleClang 用に手動設定を適用します。")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor;-fopenmp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
        set(OpenMP_INCLUDE_DIR "/opt/homebrew/opt/libomp/include")
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

    include_directories(${OpenMP_INCLUDE_DIR})

elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    message(STATUS "Using GNU GCC")
    
    find_package(OpenMP)
    
    if(NOT OpenMP_CXX_FOUND)
        message(WARNING "OpenMP が自動検出できませんでした。GCC 用に手動設定を適用します。")
        set(OpenMP_CXX_FLAGS "-fopenmp")

        # Linux 環境
        if(UNIX AND NOT APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch")
            set(OpenMP_CXX_LIB_NAMES "gomp")
            set(OpenMP_omp_LIBRARY "/opt/rh/gcc-toolset-14/root/usr/lib/gcc/aarch64-redhat-linux/14/libgomp.so")
            set(OpenMP_INCLUDE_DIR "/opt/rh/gcc-toolset-14/root/usr/lib/gcc/aarch64-redhat-linux/14/include")

        elseif(UNIX AND NOT APPLE)
            set(OpenMP_CXX_LIB_NAMES "gomp")
            set(OpenMP_omp_LIBRARY "/opt/rh/gcc-toolset-14/root/usr/lib/gcc/x86_64-redhat-linux/14/libgomp.so")
            set(OpenMP_INCLUDE_DIR "/opt/rh/gcc-toolset-14/root/usr/lib/gcc/x86_64-redhat-linux/14/include")
            
        # macOS 環境 (Homebrew の libomp)
        elseif(APPLE)
            set(OpenMP_CXX_LIB_NAMES "omp")
            set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
            set(OpenMP_INCLUDE_DIR "/opt/homebrew/opt/libomp/include")
        endif()
        
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

elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(STATUS "Using Clang")
    find_package(OpenMP)
    if(OpenMP_FOUND)
        add_compile_options(-fopenmp)
        add_link_options(-fopenmp)
    endif()

elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    message(STATUS "Using MSVC (Microsoft Visual Studio)")

    find_package(OpenMP)
    if(NOT OpenMP_CXX_FOUND)
        message(STATUS "OpenMP が自動検出できませんでした。MSVC 用に手動設定を適用します。")

        # MSVC では /openmp フラグを追加
        set(OpenMP_CXX_FLAGS "/openmp")
        set(OpenMP_CXX_LIB_NAMES "vcomp")
        set(OpenMP_omp_LIBRARY "vcomp.lib")  # MSVC の OpenMP ライブラリ
        set(OpenMP_CXX_FOUND TRUE)
    endif()

    if(NOT TARGET OpenMP::OpenMP_CXX)
        add_library(OpenMP::OpenMP_CXX INTERFACE IMPORTED)
        set_target_properties(OpenMP::OpenMP_CXX PROPERTIES
            INTERFACE_COMPILE_OPTIONS "${OpenMP_CXX_FLAGS}"
            INTERFACE_LINK_LIBRARIES "${OpenMP_omp_LIBRARY}"
        )
    endif()

    message(STATUS "Using OpenMP compile options: ${OpenMP_CXX_FLAGS}")
    message(STATUS "Using OpenMP library: ${OpenMP_omp_LIBRARY}")

endif()


# ------------------------------------------------------------------------------
# TBB 設定
# ------------------------------------------------------------------------------

# OS によって TBB の検索方法を変更
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")  # macOS
    # macOS: Homebrew の TBB を使用
    execute_process(COMMAND brew --prefix tbb OUTPUT_VARIABLE TBB_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(TBB_DIR "${TBB_PREFIX}/lib/cmake/tbb")
    find_package(TBB CONFIG REQUIRED)

elseif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")  # Ubuntu
    
    message(STATUS "TBB_DIR: ${TBB_DIR}")

    find_package(TBB REQUIRED)

elseif(${CMAKE_SYSTEM_NAME} MATCHES "Windows")  # Windows
    set(PKG_CONFIG_EXECUTABLE "D:/a/_temp/msys64/mingw64/bin/pkg-config.exe")
    set(PKG_CONFIG_PATH "D:/a/_temp/msys64/mingw64/lib/pkgconfig")

    # TBB を `pkg-config` で検出
    find_package(PkgConfig REQUIRED)

    set(TBB_DIR "D:/a/_temp/msys64/mingw64/lib/cmake/TBB")

    pkg_check_modules(TBB REQUIRED tbb)

    find_package(TBB REQUIRED)

else()
    message(FATAL_ERROR "Unsupported OS for TBB!")
endif()

# TBB が見つかったか確認
if(NOT TBB_FOUND)
    message(FATAL_ERROR "TBB not found!")
else()
    message(STATUS "TBB Found!")
endif()

# TBB のヘッダーパスを追加
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    include_directories(SYSTEM "${TBB_PREFIX}/include")
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    include_directories(SYSTEM "${TBB_DIR}/include")
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    include_directories(SYSTEM "$ENV{VCPKG_ROOT}/installed/x64-windows/include")
endif()


# # ------------------------------------------------------------------------------
# # AddressSanitizer 設定（Clang, GCC 両方に対応）
# # ------------------------------------------------------------------------------
# if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#     message(STATUS "Enabling AddressSanitizer")
#     add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
#     add_link_options(-fsanitize=address)
# endif()

# ------------------------------------------------------------------------------
# OpenBLAS + LAPACK 設定 (macOS, Linux, Windows)
# ------------------------------------------------------------------------------

if (APPLE)
    execute_process(COMMAND brew --prefix openblas OUTPUT_VARIABLE OPENBLAS_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)

    set(OPENBLAS_INCLUDE_DIR "${OPENBLAS_PREFIX}/include")
    set(OPENBLAS_LIB "${OPENBLAS_PREFIX}/lib/libopenblas.dylib")

    if (EXISTS ${OPENBLAS_LIB})
        message(STATUS "Using OpenBLAS: ${OPENBLAS_LIB}")
        set(BLAS_LIBRARIES ${OPENBLAS_LIB})
        set(BLAS_INCLUDE_DIRS ${OPENBLAS_INCLUDE_DIR})
        # OpenBLAS には LAPACK 機能も含まれるため、LAPACK_LIBRARIES にも同じパスを設定
        set(LAPACK_LIBRARIES ${OPENBLAS_LIB})
    else()
        message(FATAL_ERROR "OpenBLAS not found! Install it using 'brew install openblas'")
    endif()

elseif (UNIX AND NOT APPLE)  # Linux

    find_package(OpenBLAS)

    if (NOT OpenBLAS_FOUND)
        message(WARNING "OpenBLAS が自動検出できませんでした。手動設定を適用します。")

        # find コマンドで OpenBLAS のライブラリを検索
        execute_process(
            COMMAND find /usr /usr/local -type f -name "libopenblas-*.so"
            OUTPUT_VARIABLE OPENBLAS_LIB_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        # 結果が見つかったかチェック
        if(OPENBLAS_LIB_PATH)
            message(STATUS "Found OpenBLAS Library: ${OPENBLAS_LIB_PATH}")

            # 最初の結果のみを取得 (複数ヒットする場合は最初の行を使用)
            string(REGEX MATCH "([^\n]+)" OPENBLAS_LIB_PATH ${OPENBLAS_LIB_PATH})

            message(STATUS "Using OpenBLAS Library: ${OPENBLAS_LIB_PATH}")

            # 変数としてセット
            set(OPENBLAS_LIB ${OPENBLAS_LIB_PATH} CACHE STRING "Path to OpenBLAS library" FORCE)
            set(OpenBLAS_LIBRA-NOTFOUND FALSE)
        endif()
        set(OPENBLAS_INCLUDE_DIR "/usr/include/openblas")

        message(STATUS "Finding LAPACK library...")
        # find コマンドで OpenBLAS のライブラリを検索
        execute_process(
            COMMAND find /usr /usr/lib64 -type f -name "liblapack.*"
            OUTPUT_VARIABLE LAPACK_LIB_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        # 結果が見つかったかチェック
        if(LAPACK_LIB_PATH)
            message(STATUS "Found LAPACK Library: ${LAPACK_LIB_PATH}")

            # 最初の結果のみを取得 (複数ヒットする場合は最初の行を使用)
            string(REGEX MATCH "([^\n]+)" LAPACK_LIB_PATH ${LAPACK_LIB_PATH})

            message(STATUS "Using LAPACK Library: ${LAPACK_LIB_PATH}")

            # 変数としてセット
            set(LAPACK_LIBRARIES ${LAPACK_LIB_PATH} CACHE STRING "Path to LAPACK library" FORCE)
        endif()
        set(LAPACK_INCLUDE_DIRS "/usr/include/openblas")
    endif()

elseif (WIN32)  # Windows
    set(OPENBLAS_LIB "D:/a/_temp/msys64/mingw64/lib/libopenblas.a")
    set(OPENBLAS_INCLUDE_DIR "D:/a/_temp/msys64/mingw64/include/openblas")
    set(LAPACK_LIBRARIES "D:/a/_temp/msys64/mingw64/lib/liblapack.a")
    set(LAPACK_INCLUDE_DIRS "D:/a/_temp/msys64/mingw64/include/openblas")
endif()

# ------------------------------------------------------------------------------
# Apple Silicon (ARM64) の場合のみMetalを有効化
# ------------------------------------------------------------------------------
if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    message(STATUS "Building with Metal support on Apple Silicon (ARM64)")

    find_library(METAL_LIB Metal)
    find_library(METALKIT_LIB MetalKit)
    find_library(FOUNDATION_LIB Foundation)

    # Metal シェーダーのソースファイル
    file(GLOB KERNEL_FILES metal_kernel/*/*.metal)

    # AIR ファイルと METALLIB の出力先
    set(SHADER_AIR_FILES "")
    set(SHADER_LIB "${CMAKE_BINARY_DIR}/kernel/shaders.metallib")

    # Metal シェーダーをコンパイルするカスタムコマンド
    foreach(METAL_FILE ${KERNEL_FILES})
        get_filename_component(KERNEL_NAME ${METAL_FILE} NAME_WE)
        set(SHADER_AIR "${CMAKE_BINARY_DIR}/${KERNEL_NAME}.air")
        list(APPEND SHADER_AIR_FILES ${SHADER_AIR})

        add_custom_command(
            OUTPUT ${SHADER_AIR}
            COMMAND xcrun -sdk macosx metal -c ${METAL_FILE} -o ${SHADER_AIR}
            DEPENDS ${METAL_FILE}
            COMMENT "Compiling Metal shader: ${KERNEL_NAME}.metal"
        )
    endforeach()

    # すべての .air ファイルを 1 つの .metallib に統合
    add_custom_command(
        OUTPUT ${SHADER_LIB}
        COMMAND xcrun -sdk macosx metallib ${SHADER_AIR_FILES} -o ${SHADER_LIB}
        DEPENDS ${SHADER_AIR_FILES}
        COMMENT "Linking Metal shaders into ${SHADER_LIB}"
    )
    set(METAL_LIBRARIES ${METAL_LIB} ${METALKIT_LIB} ${FOUNDATION_LIB})

    # シェーダーのビルドターゲット
    add_custom_target(
        compile_shaders ALL
        DEPENDS ${SHADER_LIB}
    )
else()
    message(STATUS "Metal is only supported on Apple Silicon (ARM64). Skipping Metal build.")
endif()


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

# ソースファイルの指定
file(GLOB BUILD_SOURCES "src/*.cpp" "src/*/*.cpp" "scr/*/binding/*.cpp")

if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    file(GLOB CXX_SRC_FILES src/*/*.cpp)
    file(GLOB OBJCXX_SRC_FILES src/metal/*/*.mm)
    file(GLOB CXX_HEADERS include/*/*.hpp)
    file(GLOB OBJCXX_HEADERS include/metal/*/*.h)
    # .mm ファイルを Objective-C++ でコンパイル
    set_source_files_properties(${OBJCXX_SRC_FILES} ${OBJCXX_HEADERS} PROPERTIES COMPILE_FLAGS "-x objective-c++")
    set_source_files_properties(${OBJCXX_FILES} PROPERTIES COMPILE_FLAGS "-fobjc-arc")
    set(BUILD_SOURCES ${CXX_SRC_FILES} ${OBJCXX_SRC_FILES} ${CXX_HEADERS} ${OBJCXX_HEADERS})
endif()

add_library(VeloxML SHARED ${BUILD_SOURCES})
target_include_directories(VeloxML PUBLIC
    include
    ${OPENBLAS_INCLUDE_DIR}
    tests/testutils
)
set_property(TARGET VeloxML PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(VeloxML PUBLIC
    OpenMP::OpenMP_CXX
    TBB::tbb
    ${OPENBLAS_LIB}
    ${LAPACK_LIBRARIES}
    pybind11::module
)

if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    target_link_libraries(VeloxML PRIVATE ${METAL_LIBRARIES})
    add_dependencies(VeloxML compile_shaders)
endif()

# ------------------------------------------------------------------------------
# Python バインドのビルド設定
# ------------------------------------------------------------------------------
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

if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    target_link_libraries(c_veloxml_core PRIVATE ${METAL_LIBRARIES})
    add_dependencies(c_veloxml_core compile_shaders)
endif()

# Google Test を明示的にインストールしないようにする
if(TARGET gtest OR TARGET gtest_main)
    message(STATUS "Excluding Google Test from installation")
    set_target_properties(gtest PROPERTIES EXCLUDE_FROM_ALL TRUE EXCLUDE_FROM_DEFAULT_BUILD TRUE)
    set_target_properties(gtest_main PROPERTIES EXCLUDE_FROM_ALL TRUE EXCLUDE_FROM_DEFAULT_BUILD TRUE)
endif()

install(TARGETS c_veloxml_core LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/veloxml/core")

# add_executable(VeloxMLMain src/main.cpp)
# target_link_libraries(VeloxMLMain VeloxML)

# テスト
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
