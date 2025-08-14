include(FetchContent)
FetchContent_Declare(
   Catch2
   GIT_REPOSITORY https://github.com/catchorg/Catch2.git
   GIT_TAG v3.16.0
   GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)

add_executable(rt_tests
   core/test_lights.cpp
   core/test_mis.cpp
   core/test_photon.cpp
   core/test_photon_map.cpp
   core/test_rng.cpp
   core/test_vec3.cpp
   core/test_ray.cpp
   core/test_aabb.cpp
   core/test_bsdf.cpp
   core/test_camera.cpp
   core/test_color.cpp
   core/test_dielectric.cpp
   core/test_frame.cpp
   core/test_half.cpp
   core/test_furnace.cpp
   core/test_ggx.cpp
   core/test_sampling.cpp
   core/test_texture.cpp
   core/test_tonemap.cpp
   core/test_scene_view.cpp
   core/test_shading_normal.cpp
   core/test_spectrum.cpp
   core/test_triangle.cpp
   scene/test_bvh.cpp
   scene/test_light_distribution.cpp
)
target_include_directories(rt_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(rt_tests PRIVATE core scene rt_options Catch2::Catch2WithMain)
catch_discover_tests(rt_tests)
