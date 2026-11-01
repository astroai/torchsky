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
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

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

# Find PyTorch
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.__version__)"
    OUTPUT_VARIABLE TORCHSKY_BUILD_TORCH_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REGEX MATCH "^[0-9]+\\.[0-9]+" TORCHSKY_BUILD_TORCH_ABI "${TORCHSKY_BUILD_TORCH_VERSION}")
if(NOT TORCHSKY_BUILD_TORCH_ABI)
    message(FATAL_ERROR "Could not parse PyTorch version ${TORCHSKY_BUILD_TORCH_VERSION}")
endif()
list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})

# CUDA-enabled PyTorch only: PyTorch's CMake package requires the full matching
# toolkit even though torchsky does not compile CUDA sources. Runtime libraries
# alone are insufficient; fail here before Caffe2 emits a less actionable error.
set(TORCHSKY_PYTORCH_CUDA_VERSION "")
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.version.cuda or '')"
    OUTPUT_VARIABLE TORCHSKY_PYTORCH_CUDA_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(TORCHSKY_PYTORCH_CUDA_VERSION)
    find_package(CUDAToolkit QUIET)
    if(NOT CUDAToolkit_FOUND OR NOT CUDAToolkit_NVCC_EXECUTABLE OR NOT TARGET CUDA::nvrtc)
        message(FATAL_ERROR
            "torchsky detected CUDA-enabled PyTorch ${TORCHSKY_BUILD_TORCH_VERSION} "
            "(CUDA ${TORCHSKY_PYTORCH_CUDA_VERSION}), but CMake could not find a full "
            "matching CUDA toolkit with CUDA::nvrtc. Runtime-only CUDA libraries are "
            "not enough for find_package(Torch). Install the toolkit and set CUDA_HOME "
            "or CUDAToolkit_ROOT, or build with CPU-only PyTorch."
        )
    endif()
    # conda-forge installs cuda.h under targets/<triple>/include, not
    # $PREFIX/include; Caffe2's config needs CUDA_INCLUDE_DIRS set before
    # find_package(Torch) or it fails with "cannot find the CUDA libraries".
    if(DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}/targets/x86_64-linux/include/cuda.h")
        set(CUDA_HOME "$ENV{CONDA_PREFIX}/targets/x86_64-linux")
        set(CUDAToolkit_ROOT "${CUDA_HOME}")
        set(CUDA_TOOLKIT_ROOT_DIR "${CUDA_HOME}")
    endif()
    if(CUDAToolkit_INCLUDE_DIRS AND NOT CUDA_INCLUDE_DIRS)
        set(CUDA_INCLUDE_DIRS "${CUDAToolkit_INCLUDE_DIRS}")
    endif()
endif()

find_package(Torch REQUIRED)

# Some distributions ship a Torch CMake target with include paths that may be missing.
# Filter out any non-existent include dirs to avoid CMake configure errors.
if(TARGET torch)
    get_target_property(TORCH_INCLUDE_DIRS torch INTERFACE_INCLUDE_DIRECTORIES)
    if(TORCH_INCLUDE_DIRS)
        set(TORCH_INCLUDE_DIRS_FILTERED "")
        foreach(DIR ${TORCH_INCLUDE_DIRS})
            if(EXISTS "${DIR}")
                list(APPEND TORCH_INCLUDE_DIRS_FILTERED "${DIR}")
            else()
                message(STATUS "Dropping missing Torch include dir: ${DIR}")
            endif()

            # ponytail: normalize conda's known split header layout; generalize
            # this adapter only if another TorchConfig layout appears.
            if(DIR MATCHES "^(.+)/torch/csrc/api/include/?$")
                set(_TORCHSKY_CONDA_INCLUDE_ROOT "${CMAKE_MATCH_1}")
                if(EXISTS "${_TORCHSKY_CONDA_INCLUDE_ROOT}/torch/csrc/autograd/autograd.h")
                    list(APPEND TORCH_INCLUDE_DIRS_FILTERED "${_TORCHSKY_CONDA_INCLUDE_ROOT}")
                    message(STATUS "Added conda PyTorch include root: ${_TORCHSKY_CONDA_INCLUDE_ROOT}")
                endif()
            endif()
        endforeach()
        list(REMOVE_DUPLICATES TORCH_INCLUDE_DIRS_FILTERED)
        if(TORCH_INCLUDE_DIRS_FILTERED)
            set_target_properties(torch PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${TORCH_INCLUDE_DIRS_FILTERED}")
        endif()
    endif()
endif()

# Source files
set(SOURCES
    bindings.cpp
    cpu_isa.cpp
    geom.cpp
    healpix.cpp
    ishioka_scalar.cpp
    operators/drizzle_cpu.cpp
    operators/fixed_cpu.cpp
    operators/fixed_schemas.cpp
    operators/spectral_cpu.cpp
    operators/spectral_schemas.cpp
    operators/wcs_cpu.cpp
    operators/wcs_schemas.cpp
    wcs.cpp
)

# Create python binding module
nanobind_add_module(_cpp_ext ${SOURCES})
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

# conda may install a copied header tree under site-packages/torch that is older
# than the matching libtorch package in the same solved environment.  Prefer the
# headers next to the loaded libtorch libraries; mixing the two corrupts the
# at::Tensor ABI even when both package versions read "2.13.0".
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import sys; print(sys.prefix)"
    OUTPUT_VARIABLE TORCHSKY_RUNTIME_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(EXISTS "${TORCHSKY_RUNTIME_PREFIX}/include/ATen/core/TensorBody.h")
    target_include_directories(
        _cpp_ext SYSTEM BEFORE PRIVATE
        "${TORCHSKY_RUNTIME_PREFIX}/include/torch/csrc/api/include"
        "${TORCHSKY_RUNTIME_PREFIX}/include"
    )
    message(STATUS "Prioritizing libtorch headers in ${TORCHSKY_RUNTIME_PREFIX}/include")
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(
        _cpp_ext PRIVATE
        "-ffile-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}=torchsky-source"
        "-ffile-prefix-map=${CMAKE_CURRENT_BINARY_DIR}=torchsky-build"
        "-ffile-prefix-map=${TORCHSKY_RUNTIME_PREFIX}=torch-runtime"
    )
elseif(MSVC)
    target_compile_options(
        _cpp_ext PRIVATE
        "/pathmap:${CMAKE_CURRENT_SOURCE_DIR}=torchsky-source"
        "/pathmap:${CMAKE_CURRENT_BINARY_DIR}=torchsky-build"
        "/pathmap:${TORCHSKY_RUNTIME_PREFIX}=torch-runtime"
    )
endif()
# Link libraries
if(TARGET torch)
    target_link_libraries(_cpp_ext PRIVATE torch)
    message(STATUS "Linking native extension with imported Torch target")
elseif(TORCH_LIBRARIES)
    target_link_libraries(_cpp_ext PRIVATE ${TORCH_LIBRARIES})
    if(TORCH_CXX_FLAGS)
        separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
        target_compile_options(_cpp_ext PRIVATE ${TORCH_CXX_FLAGS_LIST})
    endif()
    message(STATUS "Linking native extension with TorchConfig library fallback")
else()
    message(FATAL_ERROR "TorchConfig provided neither the torch target nor TORCH_LIBRARIES")
endif()

# Compiler flags
target_compile_definitions(_cpp_ext PRIVATE VERSION_INFO="")
target_compile_definitions(_cpp_ext PRIVATE TORCHSKY_TORCH_ABI="${TORCHSKY_BUILD_TORCH_ABI}")
target_compile_features(_cpp_ext PRIVATE cxx_std_20)

if(APPLE)
    target_link_libraries(_cpp_ext PRIVATE "-framework Accelerate")
endif()

# Install the compiled module
# For scikit-build-core, install to the package directory
install(TARGETS _cpp_ext DESTINATION torchsky)
