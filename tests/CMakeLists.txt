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
   core/test_vec3.cpp
)
target_link_libraries(rt_tests PRIVATE core rt_options Catch2::Catch2WithMain)
catch_discover_tests(rt_tests)
