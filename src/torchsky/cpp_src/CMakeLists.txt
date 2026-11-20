cmake_minimum_required(VERSION 3.15)
project(torchsky_cpp)

if(DEFINED ENV{TORCHSKY_DISABLE_CPP} AND "$ENV{TORCHSKY_DISABLE_CPP}" STREQUAL "1")
    message(STATUS "TORCHSKY_DISABLE_CPP=1: skipping native extension build")
    return()
endif()

# Compiler optimization
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(TORCHSKY_COMPILE_OPTIONS "")
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    list(APPEND TORCHSKY_COMPILE_OPTIONS -O3)
endif()

option(TORCHSKY_ENABLE_IPO "Enable IPO/LTO for the native extension" ON)
if(TORCHSKY_ENABLE_IPO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT IPO_SUPPORTED OUTPUT IPO_ERROR)
    if(NOT IPO_SUPPORTED)
        message(STATUS "IPO/LTO not supported: ${IPO_ERROR}")
        set(TORCHSKY_ENABLE_IPO OFF)
    endif()
endif()




# Add environment prefix to search path (important for pixi/conda)
if(DEFINED ENV{BUILD_PREFIX} AND DEFINED ENV{PREFIX})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{PREFIX}")
    message(STATUS "Added environment prefix to CMAKE_PREFIX_PATH: $ENV{PREFIX}")
endif()

# Find required packages
# In pixi/conda packaging builds, ensure Python comes from the host prefix.
if(DEFINED ENV{BUILD_PREFIX} AND DEFINED ENV{PREFIX} AND EXISTS "$ENV{PREFIX}/bin/python")
    set(Python_ROOT_DIR "$ENV{PREFIX}" CACHE PATH "Python root directory" FORCE)
    set(Python_EXECUTABLE "$ENV{PREFIX}/bin/python" CACHE FILEPATH "Python executable" FORCE)
endif()
set(Python_FIND_STRATEGY LOCATION)
find_package(Python COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT} REQUIRED)

# Find nanobind
# Detect the installed nanobind package and import it into CMake
# For pixi builds, nanobind is in the build environment, not host environment
if(DEFINED ENV{BUILD_PREFIX})
    # In pixi build environment, use build prefix python for nanobind detection
    set(NANOBIND_PYTHON_EXECUTABLE "$ENV{BUILD_PREFIX}/bin/python")
    message(STATUS "Using build environment Python for nanobind: ${NANOBIND_PYTHON_EXECUTABLE}")
else()
    # Normal development environment
    set(NANOBIND_PYTHON_EXECUTABLE "${Python_EXECUTABLE}")
endif()

execute_process(
  COMMAND "${NANOBIND_PYTHON_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT
  ERROR_QUIET
)

if(nanobind_ROOT)
    message(STATUS "Found nanobind cmake directory: ${nanobind_ROOT}")
    list(APPEND CMAKE_PREFIX_PATH ${nanobind_ROOT})
else()
    # Fallback: try to find nanobind via Python import in build environment
    execute_process(
        COMMAND "${NANOBIND_PYTHON_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
        OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT
        ERROR_QUIET
    )
    if(nanobind_ROOT)
        message(STATUS "Found nanobind via Python import: ${nanobind_ROOT}")
        list(APPEND CMAKE_PREFIX_PATH ${nanobind_ROOT})
    endif()
endif()

execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('purelib'))"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(nanobind CONFIG REQUIRED)

message(DEBUG "torchsky: Python_EXECUTABLE=${Python_EXECUTABLE}")
message(DEBUG "torchsky: Python_VERSION=${Python_VERSION}")
message(DEBUG "torchsky: NANOBIND_PYTHON_EXECUTABLE=${NANOBIND_PYTHON_EXECUTABLE}")
message(DEBUG "torchsky: nanobind_DIR=${nanobind_DIR}")
message(DEBUG "torchsky: nanobind_ROOT=${nanobind_ROOT}")

set(SOURCES
    bindings.cpp
    cpu_isa.cpp
    crossmatch.cpp
    geom.cpp
    healpix.cpp
    ishioka_scalar.cpp
    sphere_ranges.cpp
    operators/drizzle_cpu.cpp
    operators/fixed_cpu.cpp
    operators/spectral_cpu.cpp
    operators/wcs_tpv_cpu.cpp
)

# Python module. The core does not include or link libtorch (tensors cross the
# boundary through DLPack), so one build serves every PyTorch version. On
# CPython >= 3.12 nanobind targets the stable ABI (one abi3 wheel per platform).
nanobind_add_module(_cpp_ext STABLE_ABI NB_STATIC ${SOURCES})
target_include_directories(
    _cpp_ext PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_compile_options(_cpp_ext PRIVATE ${TORCHSKY_COMPILE_OPTIONS})

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx2" TORCHSKY_HAS_AVX2)
    check_cxx_compiler_flag("-mfma" TORCHSKY_HAS_FMA)
    if(TORCHSKY_HAS_AVX2 AND TORCHSKY_HAS_FMA)
        add_library(torchsky_ishioka_avx2 OBJECT ishioka_avx2.cpp)
        target_include_directories(
            torchsky_ishioka_avx2 PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
        )
        target_compile_features(torchsky_ishioka_avx2 PRIVATE cxx_std_20)
        target_compile_options(torchsky_ishioka_avx2 PRIVATE -O3 -mavx2 -mfma)
        target_compile_definitions(torchsky_ishioka_avx2 PRIVATE TORCHSKY_HAVE_AVX2_KERNEL=1)
        target_compile_definitions(_cpp_ext PRIVATE TORCHSKY_HAVE_AVX2_KERNEL=1)
        target_sources(_cpp_ext PRIVATE $<TARGET_OBJECTS:torchsky_ishioka_avx2>)
        message(STATUS "Ishioka AVX2+FMA leaf enabled with runtime dispatch")
    endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    add_library(torchsky_ishioka_neon OBJECT ishioka_neon.cpp)
    target_include_directories(
        torchsky_ishioka_neon PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )
    target_compile_features(torchsky_ishioka_neon PRIVATE cxx_std_20)
    target_compile_options(torchsky_ishioka_neon PRIVATE -O3)
    target_compile_definitions(torchsky_ishioka_neon PRIVATE TORCHSKY_HAVE_NEON_KERNEL=1)
    target_compile_definitions(_cpp_ext PRIVATE TORCHSKY_HAVE_NEON_KERNEL=1)
    target_sources(_cpp_ext PRIVATE $<TARGET_OBJECTS:torchsky_ishioka_neon>)
    message(STATUS "Ishioka arm64 NEON leaf enabled with runtime dispatch")
endif()

if(TORCHSKY_ENABLE_IPO)
    set_property(TARGET _cpp_ext PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(
        _cpp_ext PRIVATE
        "-ffile-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}=torchsky-source"
        "-ffile-prefix-map=${CMAKE_CURRENT_BINARY_DIR}=torchsky-build"
    )
elseif(MSVC)
    target_compile_options(
        _cpp_ext PRIVATE
        "/pathmap:${CMAKE_CURRENT_SOURCE_DIR}=torchsky-source"
        "/pathmap:${CMAKE_CURRENT_BINARY_DIR}=torchsky-build"
    )
endif()
# Compiler flags
target_compile_definitions(_cpp_ext PRIVATE VERSION_INFO="")
target_compile_features(_cpp_ext PRIVATE cxx_std_20)

if(APPLE)
    target_link_libraries(_cpp_ext PRIVATE "-framework Accelerate")
endif()

# Install the compiled module
# For scikit-build-core, install to the package directory
install(TARGETS _cpp_ext DESTINATION torchsky)
