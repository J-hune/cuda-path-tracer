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

# OptiX : en-têtes seulement (licence SDK NVIDIA), les fonctions sont chargées depuis le pilote
FetchContent_Declare(
   optix
   GIT_REPOSITORY https://github.com/NVIDIA/optix-dev.git
   GIT_TAG v9.1.0
   GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(optix)

add_library(optix INTERFACE)
target_include_directories(optix SYSTEM INTERFACE ${optix_SOURCE_DIR}/include)
target_link_libraries(optix INTERFACE ${CMAKE_DL_LIBS})

# Dear ImGui : interface de la fenêtre, avec ses backends GLFW et OpenGL 3
FetchContent_Declare(
   imgui
   GIT_REPOSITORY https://github.com/ocornut/imgui.git
   GIT_TAG v1.92.9b
   GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(imgui)

find_package(glfw3 3.4 REQUIRED)
find_package(OpenGL REQUIRED)

add_library(
   imgui STATIC
   ${imgui_SOURCE_DIR}/imgui.cpp
   ${imgui_SOURCE_DIR}/imgui_draw.cpp
   ${imgui_SOURCE_DIR}/imgui_tables.cpp
   ${imgui_SOURCE_DIR}/imgui_widgets.cpp
   ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
   ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui SYSTEM PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
target_link_libraries(imgui PUBLIC glfw OpenGL::GL)
