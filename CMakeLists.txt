cmake_minimum_required(VERSION 4.2)
project(zllm)

set(CMAKE_CXX_STANDARD 23)

if (MSVC)
        add_compile_options(/utf-8)
endif()

add_library(jsoncpp STATIC IMPORTED
        include/utils.hpp)
set_target_properties(jsoncpp PROPERTIES
        IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/jsoncpp/lib/jsoncpp.lib"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/jsoncpp/include"
)

add_executable(zllm main.cpp
        src/gguf_parser.cpp)

target_include_directories(zllm
        PRIVATE
        include
)

target_link_libraries(zllm
        PRIVATE
        jsoncpp
)
