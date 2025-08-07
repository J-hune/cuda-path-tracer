include(FetchContent)
FetchContent_Declare(
   fastgltf
   GIT_REPOSITORY https://github.com/spnda/fastgltf.git
   GIT_TAG v0.9.1
   GIT_SHALLOW TRUE
   SYSTEM
)
FetchContent_MakeAvailable(fastgltf)

# Chargement et représentation hôte de la scène (CPU uniquement)
add_library(scene STATIC gltf_loader.cpp bvh_builder.cpp photon_targets.cpp stb_image.cpp)
target_link_libraries(scene PUBLIC core PRIVATE fastgltf::fastgltf stb rt_options)
