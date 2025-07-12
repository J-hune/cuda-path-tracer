add_library(gpu STATIC render.cu)
target_link_libraries(gpu PUBLIC core scene PRIVATE rt_options)
set_target_properties(gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
