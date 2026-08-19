cmake_minimum_required(VERSION 3.15)
project(torchsky_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler optimization
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Global optimizations (Release mode)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    # Using only -O3 (avoid -funroll-loops as it may conflict with manual SIMD unrolling)
    add_compile_options(-O3)
endif()

# Ishioka k-step / k-reduce have hand AVX2+FMA paths; enable them on x86_64 Linux.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx2" TORCHSKY_HAS_AVX2)
    check_cxx_compiler_flag("-mfma" TORCHSKY_HAS_FMA)
    if(TORCHSKY_HAS_AVX2 AND TORCHSKY_HAS_FMA)
        add_compile_options(-mavx2 -mfma)
        message(STATUS "Ishioka k-step: enabling -mavx2 -mfma")
    endif()
endif()

# Enable IPO/LTO when available (portable opt-in)
include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_SUPPORTED OUTPUT IPO_ERROR)
if(IPO_SUPPORTED)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
else()
    message(STATUS "IPO/LTO not supported: ${IPO_ERROR}")
endif()




# Add environment prefix to search path (important for pixi/conda)
if(DEFINED ENV{PREFIX})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{PREFIX}")
    message(STATUS "Added environment prefix to CMAKE_PREFIX_PATH: $ENV{PREFIX}")
endif()

# Find required packages
# In pixi/conda packaging builds, ensure Python comes from the host prefix.
if(DEFINED ENV{PREFIX} AND EXISTS "$ENV{PREFIX}/bin/python")
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
if(NOT TORCHSKY_BUILD_TORCH_VERSION MATCHES "^2\\.10([.+]|$)")
    message(FATAL_ERROR
        "torchsky requires PyTorch 2.10.x at build time; found ${TORCHSKY_BUILD_TORCH_VERSION}. "
        "Install 'torch>=2.10,<2.11' before using --no-build-isolation."
    )
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





# Find DLPack headers (used for zero-copy tensor exchange)
find_path(DLPACK_INCLUDE_DIR dlpack/dlpack.h)
if(DLPACK_INCLUDE_DIR)
    message(STATUS "Found DLPack include: ${DLPACK_INCLUDE_DIR}")
else()
    message(WARNING "DLPack headers not found - DLPack-based zero-copy will fail to compile until headers are available")
endif()


# Source files
set(SOURCES
    bindings.cpp
)

# Create python binding module
nanobind_add_module(_cpp ${SOURCES})
set_source_files_properties(
    bindings.cpp
    PROPERTIES
        OBJECT_DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/healpix.cpp;${CMAKE_CURRENT_SOURCE_DIR}/geom.cpp;${CMAKE_CURRENT_SOURCE_DIR}/wcs.cpp"
)

# Link libraries
target_link_libraries(_cpp PRIVATE ${TORCH_LIBRARIES})

# Link torch_python for THPVariable_Wrap (bypasses DLPack overhead)
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch, os; print(os.path.dirname(torch.__file__))"
    OUTPUT_VARIABLE TORCH_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(DEBUG "torchsky: TORCH_ROOT=${TORCH_ROOT}")

# Avoid stale cached paths when switching environments/interpreters.
unset(TORCH_PYTHON_LIBRARY CACHE)
unset(TORCH_PYTHON_LIBRARY_FALLBACK CACHE)

find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_ROOT}/lib" NO_DEFAULT_PATH)
if(TORCH_PYTHON_LIBRARY)
    target_link_libraries(_cpp PRIVATE ${TORCH_PYTHON_LIBRARY})
    message(STATUS "Found libtorch_python: ${TORCH_PYTHON_LIBRARY}")
else()
    message(WARNING "libtorch_python not found in ${TORCH_ROOT}/lib - tensor returns will use slower DLPack path")
endif()

if(DLPACK_INCLUDE_DIR)
    target_include_directories(_cpp PRIVATE ${DLPACK_INCLUDE_DIR})
endif()


# Compiler flags
target_compile_definitions(_cpp PRIVATE VERSION_INFO="")
target_compile_definitions(_cpp PRIVATE TORCHSKY_TORCH_ABI="2.10")
target_compile_features(_cpp PRIVATE cxx_std_17)

if(APPLE)
    target_link_libraries(_cpp PRIVATE "-framework Accelerate")
endif()

# Install the compiled module
# For scikit-build-core, install to the package directory
install(TARGETS _cpp DESTINATION torchsky)
