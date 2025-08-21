find_package(glfw3 3.4 REQUIRED)
find_package(OpenGL REQUIRED)

add_executable(raytracer main.cpp final_render.cpp fly_camera.cpp image_io.cpp manifest.cpp ui.cpp window.cpp)
target_link_libraries(raytracer PRIVATE gpu scene stb imgui glfw OpenGL::GL rt_options)
