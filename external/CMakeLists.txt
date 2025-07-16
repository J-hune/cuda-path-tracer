include(FetchContent)

# stb : décodage (stb_image) et écriture (stb_image_write) d'images, utilisé par scene et app
FetchContent_Declare(
   stb
   GIT_REPOSITORY https://github.com/nothings/stb.git
   GIT_TAG 2c980bb59875b0d32144a71867fbdebb2f77cd20
)
FetchContent_MakeAvailable(stb)

# stb n'a pas de CMakeLists : cible en-têtes seuls, SYSTEM pour ignorer ses avertissements
add_library(stb INTERFACE)
target_include_directories(stb SYSTEM INTERFACE ${stb_SOURCE_DIR})
