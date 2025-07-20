find_package(glfw3 3.4 REQUIRED)
find_package(OpenGL REQUIRED)

add_executable(raytracer main.cpp fly_camera.cpp image_io.cpp window.cpp)
target_link_libraries(raytracer PRIVATE gpu scene stb glfw OpenGL::GL rt_options)
