#pragma once

#include <expected>
#include <string>

struct GLFWwindow;

// Fenêtre GLFW avec contexte OpenGL, qui affiche une image RGBA8 de taille fixe.
// L'image est lue dans un pixel buffer OpenGL, que CUDA peut remplir directement.
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
   void set_title(const std::string& title);

   // Pixel buffer OpenGL de width × height pixels RGBA8, ligne 0 en haut
   [[nodiscard]] unsigned int pixel_buffer() const noexcept { return pixel_buffer_; }

   // Affiche le contenu actuel du pixel buffer
   void present();

private:
   Window(GLFWwindow* window, int width, int height);

   GLFWwindow* window_ = nullptr;
   unsigned int texture_ = 0;
   unsigned int framebuffer_ = 0;
   unsigned int pixel_buffer_ = 0;
   int width_ = 0;
   int height_ = 0;
};
