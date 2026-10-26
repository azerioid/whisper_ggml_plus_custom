cmake_minimum_required(VERSION 3.22.1)

project(whisper_flutter_library LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(WHISPER_VENDOR_DIR "${CMAKE_CURRENT_LIST_DIR}/src")
set(GGML_VENDOR_DIR "${WHISPER_VENDOR_DIR}/ggml")
set(WHISPER_ENGINE_DIR "${WHISPER_VENDOR_DIR}/src")

# arm64: ggml's CPU backend is built in several variants (armv8.0, 8.2 with
# dotprod, 8.2 with dotprod+fp16, 8.6 with i8mm) and main.cpp loads the best
# one the phone supports. A single build for the newest CPUs would SIGILL on
# Cortex-A53 phones; the old single armv8.0 build left 1.5-2x on the table.
# Runtime loading needs ggml as shared libraries.
if (ANDROID_ABI STREQUAL "arm64-v8a")
    set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
    set(GGML_BACKEND_DL ON CACHE BOOL "" FORCE)
    set(GGML_CPU_ALL_VARIANTS ON CACHE BOOL "" FORCE)
else()
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
endif()
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_CPU ON CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_ACCELERATE OFF CACHE BOOL "" FORCE)
set(WHISPER_ALL_WARNINGS OFF CACHE BOOL "" FORCE)
set(WHISPER_COREML OFF CACHE BOOL "" FORCE)
set(WHISPER_OPENVINO OFF CACHE BOOL "" FORCE)

add_subdirectory("${GGML_VENDOR_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/ggml")
add_subdirectory("${WHISPER_ENGINE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/whisper")

add_library(whisper_flutter SHARED main.cpp)

target_include_directories(whisper_flutter PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}"
    "${WHISPER_VENDOR_DIR}"
    "${WHISPER_VENDOR_DIR}/include"
)

target_compile_definitions(whisper_flutter PUBLIC DART_SHARED_LIB)

find_library(LOG_LIB log)
find_library(ATOMIC_LIB atomic)

target_link_libraries(whisper_flutter PRIVATE whisper ${LOG_LIB})

if (ATOMIC_LIB)
    target_link_libraries(whisper_flutter PRIVATE ${ATOMIC_LIB})
endif()

if (ANDROID)
    target_link_options(whisper_flutter PRIVATE
        $<$<OR:$<STREQUAL:${ANDROID_ABI},arm64-v8a>,$<STREQUAL:${ANDROID_ABI},x86_64>>:LINKER:-z,max-page-size=16384>
        $<$<OR:$<STREQUAL:${ANDROID_ABI},arm64-v8a>,$<STREQUAL:${ANDROID_ABI},x86_64>>:LINKER:-z,common-page-size=16384>
    )
endif()

# Not "whisper": with shared libs that name is whisper.cpp's own library.
set_target_properties(whisper_flutter PROPERTIES
    PUBLIC_HEADER src/whisper.h
    OUTPUT_NAME "whisper_flutter"
)
