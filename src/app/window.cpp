#include "app/window.h"

// Fonctions OpenGL 3 déclarées directement (exportées par libGL sous Linux), sans chargeur
#define GL_GLEXT_PROTOTYPES
#define GLFW_INCLUDE_NONE
#include <GL/gl.h>
#include <GL/glext.h>
#include <GLFW/glfw3.h>
#include <expected>
#include <string>
#include <utility>

std::expected<Window, std::string> Window::create(int width, int height, const std::string& title) {
   if (glfwInit() == GLFW_FALSE) {
      return std::unexpected("Failed to initialize GLFW");
   }

   glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
   glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
   glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
   GLFWwindow* window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
   if (window == nullptr) {
      glfwTerminate();
      return std::unexpected("Failed to create the window");
   }
   glfwMakeContextCurrent(window);
   glfwSwapInterval(1);
   return Window(window, width, height);
}

Window::Window(GLFWwindow* window, int width, int height):
   window_(window),
   width_(width),
   height_(height) {
   // Texture affichée, attachée à un framebuffer pour être recopiée dans la fenêtre
   glGenTextures(1, &texture_);
   glBindTexture(GL_TEXTURE_2D, texture_);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

   glGenFramebuffers(1, &framebuffer_);
   glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer_);
   glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture_, 0);
   glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

   glGenBuffers(1, &pixel_buffer_);
   glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pixel_buffer_);
   glBufferData(
      GL_PIXEL_UNPACK_BUFFER,
      static_cast<GLsizeiptr>(width) * height * 4,
      nullptr,
      GL_STREAM_DRAW
   );
   glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

Window::Window(Window&& other) noexcept:
   window_(std::exchange(other.window_, nullptr)),
   texture_(std::exchange(other.texture_, 0)),
   framebuffer_(std::exchange(other.framebuffer_, 0)),
   pixel_buffer_(std::exchange(other.pixel_buffer_, 0)),
   width_(other.width_),
   height_(other.height_) {}

Window::~Window() {
   if (window_ == nullptr) {
      return;
   }
   glDeleteBuffers(1, &pixel_buffer_);
   glDeleteFramebuffers(1, &framebuffer_);
   glDeleteTextures(1, &texture_);
   glfwDestroyWindow(window_);
   glfwTerminate();
}

bool Window::should_close() const {
   return glfwWindowShouldClose(window_) == GLFW_TRUE;
}

void Window::poll_events() {
   glfwPollEvents();
}

void Window::present() {
   // Copie GPU -> GPU du pixel buffer vers la texture (le dernier argument est un décalage dans le
   // buffer lié, pas un pointeur CPU)
   glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pixel_buffer_);
   glBindTexture(GL_TEXTURE_2D, texture_);
   glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_, height_, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
   glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

   // La ligne 0 de l'image est en haut, celle d'OpenGL en bas : la copie retourne l'image
   int framebuffer_width = 0;
   int framebuffer_height = 0;
   glfwGetFramebufferSize(window_, &framebuffer_width, &framebuffer_height);
   glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer_);
   glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
   glBlitFramebuffer(
      0,
      0,
      width_,
      height_,
      0,
      framebuffer_height,
      framebuffer_width,
      0,
      GL_COLOR_BUFFER_BIT,
      GL_NEAREST
   );
   glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
   glfwSwapBuffers(window_);
}
