cmake_minimum_required(VERSION 4.1)
project(jarvis)

set(CMAKE_CXX_STANDARD 20)

find_package(whisper CONFIG REQUIRED)
find_package(llama CONFIG REQUIRED)

# miniaudio is header-only, just need to find the header
find_path(MINIAUDIO_INCLUDE_DIR miniaudio.h REQUIRED)

add_executable(jarvis main.cpp
        src/audio/audio_capture.cpp
        src/audio/audio_capture.h
        src/audio/vad.h
)

target_include_directories(jarvis PRIVATE ${MINIAUDIO_INCLUDE_DIR})
target_link_libraries(jarvis PRIVATE whisper llama)
