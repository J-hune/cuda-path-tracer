find_package(OpenGL REQUIRED)

add_library(gpu STATIC renderer.cu denoiser.cu gl_pixel_buffer.cu gpu_timer.cu)
target_link_libraries(gpu PUBLIC core scene PRIVATE OpenGL::GL optix rt_options)
set_target_properties(gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
