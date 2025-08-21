#pragma once

#include <cstdint>
#include <expected>
#include <optional>
#include <string>

struct GLFWwindow;

// Touches de déplacement, par position physique : Forward est la touche W d'un clavier QWERTY,
// donc Z sur un clavier AZERTY
enum class Key : std::uint8_t { Forward, Backward, Left, Right, Up, Down, Fast };

struct Size {
   int width = 0;
   int height = 0;
   bool operator==(const Size&) const = default;
};

struct CursorPosition {
   double x = 0.0;
   double y = 0.0;
};

// Fenêtre GLFW avec contexte OpenGL, qui affiche une image RGBA8, mise à l'échelle de la fenêtre,
// et l'interface ImGui par-dessus. L'image est lue dans un pixel buffer OpenGL, que CUDA peut
// remplir directement. Une seule fenêtre par programme : elle initialise et termine GLFW et ImGui.
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
   // Attend le prochain événement (fenêtre réduite : inutile de rendre)
   static void wait_events();

   // Pixel buffer OpenGL de width × height pixels RGBA8, ligne 0 en haut
   [[nodiscard]] unsigned int pixel_buffer() const noexcept { return pixel_buffer_; }

   // Commence l'image de l'interface : les appels ImGui se font entre begin_frame() et present()
   static void begin_frame();
   // Affiche le contenu actuel du pixel buffer, puis l'interface
   void present();

   // Taille actuelle de la zone de dessin, en pixels (0 × 0 si la fenêtre est réduite)
   [[nodiscard]] Size framebuffer_size() const;
   // Taille de l'image affichée (celle du pixel buffer)
   [[nodiscard]] Size image_size() const noexcept { return {width_, height_}; }
   // Change la taille de l'image affichée. Le pixel buffer ne doit plus être enregistré auprès de
   // CUDA : son contenu est réalloué.
   void resize_image(Size size);

   // Entrées clavier et souris, à jour après poll_events()
   [[nodiscard]] bool is_down(Key key) const;
   [[nodiscard]] bool is_looking() const; // clic droit maintenu hors de l'interface
   [[nodiscard]] CursorPosition cursor_position() const;
   // Position du curseur en pixels de l'image affichée, rien s'il est en dehors
   [[nodiscard]] std::optional<CursorPosition> cursor_in_image() const;
   void close();

   // Temps écoulé depuis l'initialisation, en secondes
   [[nodiscard]] static double time();

private:
   // Rectangle où l'image est affichée, en pixels du framebuffer depuis le coin supérieur gauche
   struct Viewport {
      int left = 0;
      int top = 0;
      int width = 0;
      int height = 0;
   };

   Window(GLFWwindow* window, int width, int height);
   [[nodiscard]] Viewport image_viewport() const;

   GLFWwindow* window_ = nullptr;
   unsigned int texture_ = 0;
   unsigned int framebuffer_ = 0;
   unsigned int pixel_buffer_ = 0;
   int width_ = 0;
   int height_ = 0;
};
