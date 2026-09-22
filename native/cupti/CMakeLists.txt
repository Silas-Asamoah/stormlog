cmake_minimum_required(VERSION 3.20)

project(stormlog_cupti_injection VERSION 0.1.0 LANGUAGES CXX)

option(
    STORMLOG_CUPTI_VALIDATE_ONLY
    "Validate the portable build definition without requiring a CUDA toolkit"
    OFF
)

if(STORMLOG_CUPTI_VALIDATE_ONLY)
    message(STATUS "Validated Stormlog CUPTI build definition (no native target built)")
    return()
endif()

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "The Stormlog CUPTI injection prototype currently supports Linux only")
endif()

find_package(CUDAToolkit 12.0 REQUIRED)

find_path(
    CUPTI_INCLUDE_DIR
    NAMES cupti.h
    HINTS
        "${CUDAToolkit_ROOT}/extras/CUPTI/include"
        "${CUDAToolkit_ROOT}/include"
        ${CUDAToolkit_INCLUDE_DIRS}
    REQUIRED
)
find_library(
    CUPTI_LIBRARY
    NAMES cupti
    HINTS
        "${CUDAToolkit_ROOT}/extras/CUPTI/lib64"
        "${CUDAToolkit_ROOT}/targets/${CMAKE_SYSTEM_PROCESSOR}-linux/lib"
        "${CUDAToolkit_LIBRARY_DIR}"
    REQUIRED
)

add_library(stormlog_cupti_injection SHARED stormlog_cupti_injection.cpp)
target_compile_features(stormlog_cupti_injection PRIVATE cxx_std_17)
target_include_directories(stormlog_cupti_injection PRIVATE "${CUPTI_INCLUDE_DIR}")
target_link_libraries(
    stormlog_cupti_injection
    PRIVATE
        "${CUPTI_LIBRARY}"
        dl
        pthread
)
target_compile_options(
    stormlog_cupti_injection
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -fstack-protector-strong
)
target_compile_definitions(
    stormlog_cupti_injection
    PRIVATE $<$<CONFIG:Release>:_FORTIFY_SOURCE=2>
)
target_link_options(
    stormlog_cupti_injection
    PRIVATE
        -Wl,-z,relro
        -Wl,-z,now
)
set_target_properties(
    stormlog_cupti_injection
    PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES
        OUTPUT_NAME stormlog_cupti_injection
)

install(
    TARGETS stormlog_cupti_injection
    LIBRARY DESTINATION lib/stormlog/native
)
