cmake_minimum_required(VERSION 3.10)
project(paddle_ocr_flutter)

# ONNX Runtime
set(ORT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../OnnxRuntime")
include_directories(${ORT_DIR}/include)

# OpenCV
set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../OpenCV/sdk/native/jni")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -Ofast -Os -std=c++17")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -fvisibility-inlines-hidden -fdata-sections -ffunction-sections")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--gc-sections -Wl,-z,nocopyreloc")

add_library(paddle_ocr_jni SHARED
    native.cpp
    ocr_ppredictor.cpp
    ocr_db_post_process.cpp
    ocr_crnn_process.cpp
    ocr_cls_process.cpp
    ocr_clipper.cpp
    preprocess.cpp
)

find_library(log-lib log)

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION ${ORT_DIR}/libs/${ANDROID_ABI}/libonnxruntime.so
)

target_link_libraries(paddle_ocr_jni
    onnxruntime
    ${OpenCV_LIBS}
    jnigraphics
    ${log-lib}
)

# Copy ORT .so to output
add_custom_command(TARGET paddle_ocr_jni POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    ${ORT_DIR}/libs/${ANDROID_ABI}/libonnxruntime.so
    ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libonnxruntime.so
)
