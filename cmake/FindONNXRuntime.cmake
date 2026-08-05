set(_onnxruntime_hints
    "${ONNXRUNTIME_ROOT}"
    "$ENV{ONNXRUNTIME_ROOT}"
)

find_path(ONNXRuntime_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS ${_onnxruntime_hints}
    PATH_SUFFIXES include
)

find_library(ONNXRuntime_LIBRARY
    NAMES onnxruntime
    HINTS ${_onnxruntime_hints}
    PATH_SUFFIXES lib lib64
)

find_file(ONNXRuntime_DLL
    NAMES onnxruntime.dll
    HINTS ${_onnxruntime_hints}
    PATH_SUFFIXES lib bin
)

find_file(ONNXRuntime_PROVIDERS_SHARED_DLL
    NAMES onnxruntime_providers_shared.dll
    HINTS ${_onnxruntime_hints}
    PATH_SUFFIXES lib bin
)

find_file(ONNXRuntime_PROVIDERS_CUDA_DLL
    NAMES onnxruntime_providers_cuda.dll
    HINTS ${_onnxruntime_hints}
    PATH_SUFFIXES lib bin
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ONNXRuntime
    REQUIRED_VARS ONNXRuntime_INCLUDE_DIR ONNXRuntime_LIBRARY ONNXRuntime_DLL
    VERSION_VAR ONNXRuntime_VERSION
)

if(ONNXRuntime_FOUND AND NOT TARGET ONNXRuntime::ONNXRuntime)
    add_library(ONNXRuntime::ONNXRuntime SHARED IMPORTED)
    set_target_properties(ONNXRuntime::ONNXRuntime PROPERTIES
        IMPORTED_IMPLIB "${ONNXRuntime_LIBRARY}"
        IMPORTED_LOCATION "${ONNXRuntime_DLL}"
        INTERFACE_INCLUDE_DIRECTORIES "${ONNXRuntime_INCLUDE_DIR}"
    )
endif()

mark_as_advanced(
    ONNXRuntime_INCLUDE_DIR
    ONNXRuntime_LIBRARY
    ONNXRuntime_DLL
    ONNXRuntime_PROVIDERS_SHARED_DLL
    ONNXRuntime_PROVIDERS_CUDA_DLL
)
