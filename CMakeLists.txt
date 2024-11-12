cmake_minimum_required(VERSION 3.29)
set(CMAKE_PROJECT_TOP_LEVEL_INCLUDES conan_provider.cmake)

# Required for Conan
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g --diag-suppress 20012")
set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_COMPILER_ID "gcc")

project(cusim CUDA)

set(CMAKE_CUDA_STANDARD 20)

find_package(opengl_system)
find_package(glfw3)
find_package(glad)
find_package(glm)
find_package(termcolor)
find_package(assimp)
find_package(whereami)
find_package(inipp)
find_package(stb)

add_executable(cusim main.cu
        src/path.cu
        src/rendering/shader.cu
        src/rendering/gl_wrapper.cu
        src/rendering/object.cu
        src/rendering/camera.cu
        src/simulation/stepper.cu
        src/loader.cu
        src/rendering/line.cu
        src/rendering/fps_counter.cu
        src/stb_impl.cu
)

set_target_properties(cusim PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(cusim PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(cusim
        opengl::opengl glfw glad::glad glm::glm termcolor::termcolor assimp::assimp
        whereami::whereami inipp::inipp stb::stb
)

file(CREATE_LINK "${CMAKE_SOURCE_DIR}/assets" "${CMAKE_BINARY_DIR}/assets" COPY_ON_ERROR SYMBOLIC)
file(CREATE_LINK "${CMAKE_SOURCE_DIR}/shaders" "${CMAKE_BINARY_DIR}/shaders" COPY_ON_ERROR SYMBOLIC)