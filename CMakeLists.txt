cmake_minimum_required(VERSION 3.28)

project(SimilarFinder VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

option(SIMILARFINDER_BUILD_TESTS "Build SimilarFinder C++ tests" ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(Qt6 6.8 REQUIRED COMPONENTS Core Concurrent Gui Sql Widgets)
find_package(OpenCV 4.11 REQUIRED COMPONENTS core imgcodecs imgproc)
find_package(Eigen3 CONFIG REQUIRED)
find_package(faiss CONFIG REQUIRED)
find_package(ONNXRuntime REQUIRED)

function(similarfinder_copy_runtime_dlls target)
    foreach(runtime_dll IN LISTS ARGN)
        if(EXISTS "${runtime_dll}")
            add_custom_command(TARGET ${target} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${runtime_dll}"
                    "$<TARGET_FILE_DIR:${target}>"
            )
        endif()
    endforeach()
endfunction()

add_library(similarfinder_core STATIC
    src/core/app_types.h
    src/core/catalog_store.cpp
    src/core/catalog_store.h
    src/core/dino_engine.cpp
    src/core/dino_engine.h
    src/core/faiss_index.cpp
    src/core/faiss_index.h
    src/core/feature_utils.cpp
    src/core/feature_utils.h
    src/core/image_preprocessor.cpp
    src/core/image_preprocessor.h
    src/core/similarity_service.cpp
    src/core/similarity_service.h
)

target_include_directories(similarfinder_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(similarfinder_core
    PUBLIC
        Qt6::Core
        Qt6::Gui
        Qt6::Sql
        Eigen3::Eigen
        ${OpenCV_LIBS}
        faiss
        ONNXRuntime::ONNXRuntime
)
target_compile_definitions(similarfinder_core
    PUBLIC
        NOMINMAX
        QT_NO_KEYWORDS
        SIMILARFINDER_VERSION="${PROJECT_VERSION}"
)

add_executable(SimilarFinder WIN32
    src/app/main.cpp
    src/app/main_window.cpp
    src/app/main_window.h
    src/app/result_delegate.cpp
    src/app/result_delegate.h
    src/app/result_model.cpp
    src/app/result_model.h
)
target_link_libraries(SimilarFinder
    PRIVATE
        similarfinder_core
        Qt6::Concurrent
        Qt6::Gui
        Qt6::Widgets
)
add_custom_command(TARGET SimilarFinder POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:SimilarFinder>/models"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/models/dinov2_vits14.onnx"
        "$<TARGET_FILE_DIR:SimilarFinder>/models/dinov2_vits14.onnx"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/models/dinov2_vits14_manifest.json"
        "$<TARGET_FILE_DIR:SimilarFinder>/models/dinov2_vits14_manifest.json"
)
similarfinder_copy_runtime_dlls(SimilarFinder
    "${ONNXRuntime_DLL}"
    "${ONNXRuntime_PROVIDERS_SHARED_DLL}"
    "${ONNXRuntime_PROVIDERS_CUDA_DLL}"
)

if(MSVC)
    foreach(target similarfinder_core SimilarFinder)
        target_compile_options(${target} PRIVATE /W4 /permissive- /utf-8 /EHsc /arch:AVX2)
    endforeach()
endif()

include(GNUInstallDirs)
install(TARGETS SimilarFinder RUNTIME DESTINATION .)
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/models/dinov2_vits14_manifest.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/models/dinov2_vits14.onnx"
    DESTINATION models
)

if(SIMILARFINDER_BUILD_TESTS)
    enable_testing()
    find_package(Qt6 6.8 REQUIRED COMPONENTS Test)
    add_executable(similarfinder_tests
        tests/cpp/test_core.cpp
    )
    target_link_libraries(similarfinder_tests PRIVATE similarfinder_core Qt6::Gui Qt6::Test)
    similarfinder_copy_runtime_dlls(similarfinder_tests "${ONNXRuntime_DLL}")
    if(MSVC)
        target_compile_options(similarfinder_tests PRIVATE /W4 /permissive- /utf-8 /EHsc /arch:AVX2)
    endif()
    add_test(NAME similarfinder_tests COMMAND similarfinder_tests)
endif()
