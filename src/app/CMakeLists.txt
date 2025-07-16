add_executable(raytracer main.cpp image_io.cpp)
target_link_libraries(raytracer PRIVATE gpu scene stb rt_options)
