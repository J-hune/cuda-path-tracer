cmake_minimum_required(VERSION 4.4)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
   set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "")
endif()

project(raytracer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 23)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# Pas de modules C++20 dans le projet : pas d'analyse des dépendances de modules
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

add_library(rt_options INTERFACE)
target_compile_options(rt_options INTERFACE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>"
  "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>"
  "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CONFIG:Debug>>>:-lineinfo>"
)

add_subdirectory(src/core)
add_subdirectory(src/gpu)
add_subdirectory(src/scene)
add_subdirectory(src/app)

option(RT_BUILD_TESTS "Compile les tests unitaires" ON)
if(RT_BUILD_TESTS)
   enable_testing()
   add_subdirectory(tests)
endif()
