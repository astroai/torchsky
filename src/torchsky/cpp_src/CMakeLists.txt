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

# Enable IPO/LTO when available (portable opt-in)
include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_SUPPORTED OUTPUT IPO_ERROR)
if(IPO_SUPPORTED)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
else()
    message(STATUS "IPO/LTO not supported: ${IPO_ERROR}")
endif()

# Enable SIMD optimizations globally (propagates to cfitsio)
# Note: For wheels, avoid -march=native. Using reasonable baselines.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    # Enable SSSE3 for Rice compression optimization (reasonable baseline for modern PyTorch)
    add_compile_definitions(CFITSIO_HAVE_SSSE3)
    add_compile_options(-mssse3)
    message(STATUS "Enabled x86_64 optimizations (SSSE3)")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    # ARM64 implies NEON, -O3 handles vectorization well.
    # On Apple Silicon, -mcpu=native ensures correct scheduling for M-series.
    # This assumes building on M1+ for M1+ target (standard for macOS wheels).
    if(APPLE)
       message(STATUS "Enabled Native Apple Silicon optimizations")
    else()
       message(STATUS "Enabled ARM64 optimizations")
    endif()
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
list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})

# CUDA-enabled PyTorch only: cuda.cmake expects CUDA::nvrtc before extension targets link. Conda
# toolkits and CMake 4.x often omit that imported target; CPU-only PyTorch skips this path.
set(TORCHSKY_PYTORCH_CUDA_VERSION "")
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.version.cuda or '')"
    OUTPUT_VARIABLE TORCHSKY_PYTORCH_CUDA_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# PyTorch's cuda.cmake (pulled in by find_package(Torch)) calls get_target_property on CUDA::nvrtc
# before extension code runs. INTERFACE_LINK_LIBRARIES is read by PyTorch for nvrtc shorthash.
if(TORCHSKY_PYTORCH_CUDA_VERSION AND NOT TARGET CUDA::nvrtc)
    find_package(CUDAToolkit QUIET COMPONENTS nvrtc cudart)
endif()
if(TORCHSKY_PYTORCH_CUDA_VERSION AND NOT TARGET CUDA::nvrtc)
    set(_TORCHSKY_CUDA_LIB_HINTS "")
    foreach(_root IN ITEMS "${CUDAToolkit_ROOT}" "${CUDAToolkit_LIBRARY_ROOT}" "$ENV{CUDA_HOME}" "$ENV{CUDA_PATH}")
        if(_root)
            if(_root MATCHES "/include/?$")
                get_filename_component(_root "${_root}" DIRECTORY)
            endif()
            list(APPEND _TORCHSKY_CUDA_LIB_HINTS "${_root}/lib" "${_root}/lib64")
            list(APPEND _TORCHSKY_CUDA_LIB_HINTS "${_root}/targets/x86_64-linux/lib")
            list(APPEND _TORCHSKY_CUDA_LIB_HINTS "${_root}/targets/sbsa/lib")
        endif()
    endforeach()
    list(APPEND _TORCHSKY_CUDA_LIB_HINTS
        "$ENV{CONDA_PREFIX}/targets/x86_64-linux/lib"
        "$ENV{PREFIX}/targets/x86_64-linux/lib"
        "$ENV{CONDA_PREFIX}/lib"
        "$ENV{PREFIX}/lib"
    )
    list(REMOVE_DUPLICATES _TORCHSKY_CUDA_LIB_HINTS)
    set(TORCHSKY_NVRTC_LIBRARY "")
    foreach(_hint IN LISTS _TORCHSKY_CUDA_LIB_HINTS)
        if(NOT EXISTS "${_hint}")
            continue()
        endif()
        file(GLOB _TORCHSKY_NVRTC_GLOB "${_hint}/libnvrtc.so.*")
        if(_TORCHSKY_NVRTC_GLOB)
            list(GET _TORCHSKY_NVRTC_GLOB 0 TORCHSKY_NVRTC_LIBRARY)
            break()
        endif()
    endforeach()
    if(NOT TORCHSKY_NVRTC_LIBRARY)
        find_library(TORCHSKY_NVRTC_LIBRARY NAMES nvrtc HINTS ${_TORCHSKY_CUDA_LIB_HINTS})
    endif()
    if(NOT TORCHSKY_NVRTC_LIBRARY)
        message(FATAL_ERROR
            "CUDA-enabled PyTorch needs libnvrtc (CUDA::nvrtc). "
            "Install the CUDA toolkit nvrtc library or set CUDA_HOME to the toolkit root.")
    endif()
    add_library(CUDA::nvrtc SHARED IMPORTED GLOBAL)
    set_target_properties(CUDA::nvrtc PROPERTIES
        IMPORTED_LOCATION "${TORCHSKY_NVRTC_LIBRARY}"
        INTERFACE_LINK_LIBRARIES "${TORCHSKY_NVRTC_LIBRARY}"
    )
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
        endforeach()
        if(TORCH_INCLUDE_DIRS_FILTERED)
            set_target_properties(torch PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${TORCH_INCLUDE_DIRS_FILTERED}")
        endif()
    endif()
endif()

# Find ZLIB (Required for compressed FITS support)
find_package(ZLIB REQUIRED)
set(_TORCHSKY_ZLIB_SELECTED "")
if(DEFINED ZLIB_LIBRARY_RELEASE AND ZLIB_LIBRARY_RELEASE)
    set(_TORCHSKY_ZLIB_SELECTED "${ZLIB_LIBRARY_RELEASE}")
elseif(DEFINED ZLIB_LIBRARY AND ZLIB_LIBRARY)
    set(_TORCHSKY_ZLIB_SELECTED "${ZLIB_LIBRARY}")
endif()

if(_TORCHSKY_ZLIB_SELECTED AND NOT EXISTS "${_TORCHSKY_ZLIB_SELECTED}")
    get_filename_component(_TORCHSKY_ZLIB_DIR "${_TORCHSKY_ZLIB_SELECTED}" DIRECTORY)
    set(_TORCHSKY_ZLIB_CANDIDATES
        "${_TORCHSKY_ZLIB_DIR}/libz.1.dylib"
        "${_TORCHSKY_ZLIB_DIR}/libzlib.dylib"
    )
    foreach(_zlib_candidate IN LISTS _TORCHSKY_ZLIB_CANDIDATES)
        if(EXISTS "${_zlib_candidate}")
            message(WARNING "ZLIB library path '${_TORCHSKY_ZLIB_SELECTED}' does not exist; using '${_zlib_candidate}'")
            if(NOT EXISTS "${_TORCHSKY_ZLIB_SELECTED}")
                file(CREATE_LINK "${_zlib_candidate}" "${_TORCHSKY_ZLIB_SELECTED}" SYMBOLIC RESULT _TORCHSKY_ZLIB_LINK_RESULT)
                if(_TORCHSKY_ZLIB_LINK_RESULT)
                    message(WARNING "Failed to create compatibility symlink '${_TORCHSKY_ZLIB_SELECTED}' -> '${_zlib_candidate}': ${_TORCHSKY_ZLIB_LINK_RESULT}")
                else()
                    message(STATUS "Created compatibility symlink '${_TORCHSKY_ZLIB_SELECTED}' -> '${_zlib_candidate}'")
                endif()
            endif()
            set(ZLIB_LIBRARY_RELEASE "${_zlib_candidate}" CACHE FILEPATH "Path to a library." FORCE)
            set(ZLIB_LIBRARY "${_zlib_candidate}" CACHE FILEPATH "Path to a library." FORCE)
            if(TARGET ZLIB::ZLIB)
                set_property(TARGET ZLIB::ZLIB PROPERTY IMPORTED_LOCATION "${_zlib_candidate}")
                set_property(TARGET ZLIB::ZLIB PROPERTY IMPORTED_LOCATION_RELEASE "${_zlib_candidate}")
            endif()
            break()
        endif()
    endforeach()
endif()

if(ZLIB_FOUND)
    message(STATUS "Found ZLIB: ${ZLIB_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "ZLIB not found - required for compressed FITS support")
endif()

get_filename_component(TORCHSKY_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(CFITSIO_SOURCE_DIR "${TORCHSKY_REPO_ROOT}/extern/cfitsio")
set(TORCHSKY_VENDOR_SCRIPT "${TORCHSKY_REPO_ROOT}/extern/vendor.sh")
option(TORCHSKY_AUTO_VENDOR_DEPS "Auto-run extern/vendor.sh when vendored deps are missing" ON)
set(TORCHSKY_VENDOR_RAN FALSE)

macro(torchsky_run_vendor_once)
    if(TORCHSKY_AUTO_VENDOR_DEPS AND NOT TORCHSKY_VENDOR_RAN)
        if(EXISTS "${TORCHSKY_VENDOR_SCRIPT}")
            message(STATUS "Vendored deps missing; running ${TORCHSKY_VENDOR_SCRIPT}")
            execute_process(
                COMMAND bash "${TORCHSKY_VENDOR_SCRIPT}"
                WORKING_DIRECTORY "${TORCHSKY_REPO_ROOT}"
                RESULT_VARIABLE TORCHSKY_VENDOR_RESULT
            )
            if(NOT TORCHSKY_VENDOR_RESULT EQUAL 0)
                message(FATAL_ERROR "Failed to run ${TORCHSKY_VENDOR_SCRIPT} (exit ${TORCHSKY_VENDOR_RESULT})")
            endif()
            set(TORCHSKY_VENDOR_RAN TRUE)
        endif()
    endif()
endmacro()

# Find CFITSIO
option(TORCHSKY_USE_VENDORED_CFITSIO "Prefer vendored CFITSIO over environment" ON)
set(TORCHSKY_NIOBUF "" CACHE STRING "Override CFITSIO NIOBUF (default 40)")
set(TORCHSKY_MINDIRECT "" CACHE STRING "Override CFITSIO MINDIRECT (default 8640)")

if(TORCHSKY_USE_VENDORED_CFITSIO)
    unset(CFITSIO_FOUND CACHE)
    unset(CFITSIO_INCLUDE_DIR CACHE)
    unset(CFITSIO_LIBRARY CACHE)
    unset(CFITSIO_INCLUDE_DIRS CACHE)
    unset(CFITSIO_LIBRARIES CACHE)
    unset(CFITSIO_LIBRARY_DIRS CACHE)
    set(CFITSIO_FOUND FALSE)
endif()

if(NOT TORCHSKY_USE_VENDORED_CFITSIO)
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(CFITSIO QUIET cfitsio)
    endif()

    if(NOT CFITSIO_FOUND)
        find_path(CFITSIO_INCLUDE_DIR fitsio.h)
        find_library(CFITSIO_LIBRARY cfitsio)
        if(CFITSIO_INCLUDE_DIR AND CFITSIO_LIBRARY)
            set(CFITSIO_FOUND TRUE)
        endif()
    endif()
endif()

if(CFITSIO_FOUND)
    message(STATUS "Found environment CFITSIO: ${CFITSIO_INCLUDE_DIRS} ${CFITSIO_LIBRARIES}")
else()
    # Vendor CFITSIO as fallback
    if(NOT EXISTS "${CFITSIO_SOURCE_DIR}/CMakeLists.txt")
        torchsky_run_vendor_once()
    endif()

    if(EXISTS "${CFITSIO_SOURCE_DIR}/CMakeLists.txt")
        message(STATUS "Building vendored CFITSIO from ${CFITSIO_SOURCE_DIR}")

        # CFITSIO hard-defines NIOBUF/MINDIRECT in headers. When we want to tune these
        # compile-time constants, patch in #ifndef guards once so command-line -D values
        # can actually take effect.
        if(TORCHSKY_NIOBUF OR TORCHSKY_MINDIRECT)
            set(_CFITSIO_FITSIO_H "${CFITSIO_SOURCE_DIR}/fitsio.h")
            if(EXISTS "${_CFITSIO_FITSIO_H}")
                file(READ "${_CFITSIO_FITSIO_H}" _fitsio_h_content)
                if(NOT _fitsio_h_content MATCHES "#ifndef NIOBUF")
                    string(REPLACE
                        "#define NIOBUF  40  /* number of IO buffers to create (default = 40) */"
                        "#ifndef NIOBUF\n#define NIOBUF  40  /* number of IO buffers to create (default = 40) */\n#endif"
                        _fitsio_h_content
                        "${_fitsio_h_content}"
                    )
                    file(WRITE "${_CFITSIO_FITSIO_H}" "${_fitsio_h_content}")
                endif()
            endif()

            set(_CFITSIO_FITSIO2_H "${CFITSIO_SOURCE_DIR}/fitsio2.h")
            if(EXISTS "${_CFITSIO_FITSIO2_H}")
                file(READ "${_CFITSIO_FITSIO2_H}" _fitsio2_h_content)
                if(NOT _fitsio2_h_content MATCHES "#ifndef MINDIRECT")
                    string(REPLACE
                        "#define MINDIRECT 8640   /* minimum size for direct reads and writes */"
                        "#ifndef MINDIRECT\n#define MINDIRECT 8640   /* minimum size for direct reads and writes */\n#endif"
                        _fitsio2_h_content
                        "${_fitsio2_h_content}"
                    )
                    file(WRITE "${_CFITSIO_FITSIO2_H}" "${_fitsio2_h_content}")
                endif()
            endif()
        endif()

        # CFITSIO Build Options
        set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static libraries" FORCE)
        set(TESTS OFF CACHE BOOL "Disable tests" FORCE)
        set(UTILS OFF CACHE BOOL "Disable utils" FORCE)
        set(USE_PTHREADS ON CACHE BOOL "Enable thread-safe build" FORCE)

        # Force position-independent code for static library to be linked into shared library
        set(CMAKE_POSITION_INDEPENDENT_CODE ON)

        add_subdirectory("${CFITSIO_SOURCE_DIR}" cfitsio_build)

        # CFITSIO target is named 'cfitsio'
        set(CFITSIO_LIBRARY cfitsio)
        set(CFITSIO_INCLUDE_DIR "${CFITSIO_SOURCE_DIR}")
        set(CFITSIO_FOUND TRUE)
        if(TORCHSKY_NIOBUF)
            # CFITSIO reads this compile-time macro directly from fitsio.h.
            target_compile_definitions(cfitsio PRIVATE NIOBUF=${TORCHSKY_NIOBUF})
            message(STATUS "Overriding CFITSIO NIOBUF=${TORCHSKY_NIOBUF}")
        endif()
        if(TORCHSKY_MINDIRECT)
            # CFITSIO direct I/O threshold (bytes), defined in fitsio2.h.
            target_compile_definitions(cfitsio PRIVATE MINDIRECT=${TORCHSKY_MINDIRECT})
            message(STATUS "Overriding CFITSIO MINDIRECT=${TORCHSKY_MINDIRECT}")
        endif()
    else()
        message(FATAL_ERROR "CFITSIO not found in environment and vendored version not found at ${CFITSIO_SOURCE_DIR}. Run ./extern/vendor.sh")
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
nanobind_add_module(cpp ${SOURCES})

# Link libraries
target_link_libraries(cpp PRIVATE ${TORCH_LIBRARIES})

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
    target_link_libraries(cpp PRIVATE ${TORCH_PYTHON_LIBRARY})
    message(STATUS "Found libtorch_python: ${TORCH_PYTHON_LIBRARY}")
else()
    message(WARNING "libtorch_python not found in ${TORCH_ROOT}/lib - tensor returns will use slower DLPack path")
endif()

if(TARGET cfitsio)
    target_link_libraries(cpp PRIVATE cfitsio)
    target_include_directories(cpp PRIVATE "${CFITSIO_INCLUDE_DIR}")
else()
    target_link_libraries(cpp PRIVATE ${CFITSIO_LIBRARIES} ${CFITSIO_LIBRARY})
    target_include_directories(cpp PRIVATE ${CFITSIO_INCLUDE_DIRS} ${CFITSIO_INCLUDE_DIR})
    if(CFITSIO_LIBRARY_DIRS)
        target_link_directories(cpp PRIVATE ${CFITSIO_LIBRARY_DIRS})
    endif()
endif()
target_link_libraries(cpp PRIVATE ZLIB::ZLIB)

if(DLPACK_INCLUDE_DIR)
    target_include_directories(cpp PRIVATE ${DLPACK_INCLUDE_DIR})
endif()
target_compile_definitions(cpp PRIVATE HAS_CFITSIO)


# Compiler flags
target_compile_definitions(cpp PRIVATE VERSION_INFO="")
target_compile_features(cpp PRIVATE cxx_std_17)


# Install the compiled module
# For scikit-build-core, install to the package directory
install(TARGETS cpp DESTINATION torchsky)
