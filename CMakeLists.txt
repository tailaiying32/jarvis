cmake_minimum_required(VERSION 4.1)
project(jarvis)

set(CMAKE_CXX_STANDARD 20)

find_package(whisper CONFIG REQUIRED)
find_package(llama CONFIG REQUIRED)

# miniaudio is header-only, just need to find the header
find_path(MINIAUDIO_INCLUDE_DIR miniaudio.h REQUIRED)

# sherpa-onnx from external build
set(SHERPA_ONNX_DIR ${CMAKE_SOURCE_DIR}/external/sherpa-onnx)
set(SHERPA_ONNX_BUILD_DIR ${SHERPA_ONNX_DIR}/build-shared)

add_executable(jarvis main.cpp
        src/audio/audio_capture.cpp
        src/audio/audio_capture.h
        src/audio/vad.h
        src/llm/text_inference.cpp
        src/llm/text_inference.h
        src/transcribe/transcribe.cpp
        src/transcribe/transcribe.h
        src/pipeline/assistant.cpp
        src/pipeline/assistant.h
        src/tts/tts.cpp
        src/tts/tts.h
)

target_include_directories(jarvis PRIVATE
        ${MINIAUDIO_INCLUDE_DIR}
        ${SHERPA_ONNX_DIR}  # for #include "sherpa-onnx/c-api/c-api.h"
)

target_link_directories(jarvis PRIVATE
        ${SHERPA_ONNX_BUILD_DIR}/lib/Release
)

target_link_libraries(jarvis PRIVATE
        whisper
        llama
        sherpa-onnx-c-api
)

# Copy sherpa-onnx DLLs to output directory
add_custom_command(TARGET jarvis POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${SHERPA_ONNX_BUILD_DIR}/bin/Release/sherpa-onnx-c-api.dll"
        "${SHERPA_ONNX_BUILD_DIR}/bin/Release/onnxruntime.dll"
#        "${SHERPA_ONNX_BUILD_DIR}/_deps/onnxruntime-src/lib/onnxruntime_providers_shared.dll"
        $<TARGET_FILE_DIR:jarvis>
)