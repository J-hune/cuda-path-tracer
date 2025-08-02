#include "app/window.h"

// Fonctions OpenGL 3 déclarées directement (exportées par libGL sous Linux), sans chargeur
#define GL_GLEXT_PROTOTYPES
#define GLFW_INCLUDE_NONE
#include <GL/gl.h>
#include <GL/glext.h>
#include <GLFW/glfw3.h>
#include <expected>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <string>
#include <utility>

std::expected<Window, std::string> Window::create(int width, int height, const std::string& title) {
   if (glfwInit() == GLFW_FALSE) {
      return std::unexpected("Failed to initialize GLFW");
   }

   glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
   glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
   GLFWwindow* window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
   if (window == nullptr) {
      glfwTerminate();
      return std::unexpected("Failed to create the window");
   }
   glfwMakeContextCurrent(window);
   // Sans synchronisation verticale : le GPU accumule des échantillons sans attendre l'écran
   glfwSwapInterval(0);
   return Window(window, width, height);
}

namespace {

// (Ré)alloue la texture et le pixel buffer à la taille width × height
void allocate_image(GLuint texture, GLuint pixel_buffer, int width, int height) {
   glBindTexture(GL_TEXTURE_2D, texture);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

   glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pixel_buffer);
   glBufferData(
      GL_PIXEL_UNPACK_BUFFER,
      static_cast<GLsizeiptr>(width) * height * 4,
      nullptr,
      GL_STREAM_DRAW
   );
   glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

int glfw_key(Key key) {
   switch (key) {
      case Key::Forward:
         return GLFW_KEY_W;
      case Key::Backward:
         return GLFW_KEY_S;
      case Key::Left:
         return GLFW_KEY_A;
      case Key::Right:
         return GLFW_KEY_D;
      case Key::Up:
         return GLFW_KEY_SPACE;
      case Key::Down:
         return GLFW_KEY_LEFT_CONTROL;
      case Key::Fast:
         break;
   }
   return GLFW_KEY_LEFT_SHIFT;
}

} // namespace

Window::Window(GLFWwindow* window, int width, int height):
   window_(window),
   width_(width),
   height_(height) {
   // Texture affichée, attachée à un framebuffer pour être recopiée dans la fenêtre
   glGenTextures(1, &texture_);
   glBindTexture(GL_TEXTURE_2D, texture_);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   glGenBuffers(1, &pixel_buffer_);
   allocate_image(texture_, pixel_buffer_, width_, height_);

   glGenFramebuffers(1, &framebuffer_);
   glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer_);
   glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture_, 0);
   glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

   // ImGui reçoit les événements clavier et souris de GLFW, file d'attente comprise : aucun appui
   // n'est perdu, même avec peu d'images par seconde
   IMGUI_CHECKVERSION();
   ImGui::CreateContext();
   ImGui::GetIO().IniFilename = nullptr;
   ImGui::StyleColorsDark();
   ImGui_ImplGlfw_InitForOpenGL(window_, true);
   ImGui_ImplOpenGL3_Init("#version 330");
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
   ImGui_ImplOpenGL3_Shutdown();
   ImGui_ImplGlfw_Shutdown();
   ImGui::DestroyContext();
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

void Window::wait_events() {
   glfwWaitEvents();
}

Size Window::framebuffer_size() const {
   Size size;
   glfwGetFramebufferSize(window_, &size.width, &size.height);
   return size;
}

void Window::resize_image(Size size) {
   width_ = size.width;
   height_ = size.height;
   allocate_image(texture_, pixel_buffer_, width_, height_);
}

void Window::begin_frame() {
   ImGui_ImplOpenGL3_NewFrame();
   ImGui_ImplGlfw_NewFrame();
   ImGui::NewFrame();
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

   ImGui::Render();
   ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
   glfwSwapBuffers(window_);
}

bool Window::is_down(Key key) const {
   return glfwGetKey(window_, glfw_key(key)) == GLFW_PRESS;
}

bool Window::is_looking() const {
   // Un clic commencé sur l'interface ne fait pas tourner la caméra
   return glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS &&
          !ImGui::GetIO().WantCaptureMouse;
}

CursorPosition Window::cursor_position() const {
   CursorPosition position;
   glfwGetCursorPos(window_, &position.x, &position.y);
   return position;
}

void Window::close() {
   glfwSetWindowShouldClose(window_, GLFW_TRUE);
}

double Window::time() {
   return glfwGetTime();
}
