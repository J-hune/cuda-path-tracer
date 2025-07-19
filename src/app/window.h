#pragma once

#include <cstdint>
#include <expected>
#include <string>
#include <vector>

struct GLFWwindow;

// Fenêtre GLFW avec contexte OpenGL, qui affiche une image RGBA de taille fixe.
// Une seule fenêtre par programme : elle initialise et termine GLFW.
class Window {
public:
   static std::expected<Window, std::string> create(
      int width,
      int height,
      const std::string& title
   );

   ~Window();
   Window(const Window&) = delete;
   Window& operator=(const Window&) = delete;
   Window(Window&& other) noexcept;
   Window& operator=(Window&& other) = delete;

   [[nodiscard]] bool should_close() const;
   static void poll_events();

   // Affiche une image RGBA8 de width × height pixels, ligne 0 en haut
   void show(const std::vector<std::uint8_t>& rgba);

private:
   Window(GLFWwindow* window, int width, int height);

   GLFWwindow* window_ = nullptr;
   unsigned int texture_ = 0;
   unsigned int framebuffer_ = 0;
   int width_ = 0;
   int height_ = 0;
};
