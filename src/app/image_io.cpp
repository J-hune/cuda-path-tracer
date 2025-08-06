#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "app/image_io.h"

#include "core/half.h"
#include "core/tonemap.h"
#include "core/vec3.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <stb_image_write.h>
#include <string>
#include <string_view>
#include <vector>

namespace {

std::uint8_t to_byte(float c) {
   return static_cast<std::uint8_t>(std::lround(std::clamp(c, 0.f, 1.f) * 255.f));
}

// Le format EXR est little-endian, comme les processeurs x86 et ARM visés
static_assert(std::endian::native == std::endian::little);

// Ajoute la représentation binaire d'une valeur à la fin du buffer
template <typename T>
void append(std::vector<std::byte>& buffer, T value) {
   const auto bytes = std::bit_cast<std::array<std::byte, sizeof(T)>>(value);
   buffer.insert(buffer.end(), bytes.begin(), bytes.end());
}

void append_string(std::vector<std::byte>& buffer, std::string_view text) {
   for (const char c : text) {
      buffer.push_back(static_cast<std::byte>(c));
   }
   buffer.push_back(std::byte{0});
}

// Attribut d'en-tête EXR : nom, type, taille de la valeur, puis la valeur
void append_attribute_header(
   std::vector<std::byte>& buffer,
   std::string_view name,
   std::string_view type,
   std::int32_t size
) {
   append_string(buffer, name);
   append_string(buffer, type);
   append(buffer, size);
}

// Demi-précision : écrêtée à la plus grande valeur finie (65504) plutôt que de devenir infinie
std::uint16_t to_half_bits(float value) {
   return rt::float_to_half_bits(std::clamp(value, -65504.0f, 65504.0f));
}

} // namespace

std::vector<std::uint8_t> to_rgba8(
   const std::vector<rt::Vec3>& image,
   rt::DisplaySettings display
) {
   std::vector<std::uint8_t> pixels;
   pixels.reserve(image.size() * 4);
   for (const rt::Vec3& pixel : image) {
      const rt::Vec3 c = rt::display_color(pixel, display);
      pixels.push_back(to_byte(c.x));
      pixels.push_back(to_byte(c.y));
      pixels.push_back(to_byte(c.z));
      pixels.push_back(255);
   }
   return pixels;
}

bool write_png(
   const std::string& path,
   const std::vector<rt::Vec3>& image,
   int width,
   int height,
   rt::DisplaySettings display
) {
   const std::vector<std::uint8_t> pixels = to_rgba8(image, display);
   return stbi_write_png(path.c_str(), width, height, 4, pixels.data(), width * 4) != 0;
}

// EXR « scanline » non compressé, trois canaux en demi-précision (format OpenEXR, version 2) :
// nombre magique, en-tête d'attributs, table des positions des lignes, puis les lignes de pixels.
bool write_exr(const std::string& path, const std::vector<rt::Vec3>& image, int width, int height) {
   std::vector<std::byte> file;
   append(file, std::int32_t{20000630}); // nombre magique d'OpenEXR
   append(file, std::int32_t{2});        // version 2, image simple par lignes

   // Canaux, dans l'ordre alphabétique imposé par le format
   append_attribute_header(file, "channels", "chlist", 3 * (2 + 16) + 1);
   for (const std::string_view channel : {"B", "G", "R"}) {
      append_string(file, channel);
      append(file, std::int32_t{1}); // type HALF
      append(file, std::int32_t{0}); // pLinear (1 octet) puis 3 octets réservés
      append(file, std::int32_t{1}); // sous-échantillonnage horizontal
      append(file, std::int32_t{1}); // sous-échantillonnage vertical
   }
   file.push_back(std::byte{0});

   append_attribute_header(file, "compression", "compression", 1);
   file.push_back(std::byte{0}); // NO_COMPRESSION
   for (const std::string_view window : {"dataWindow", "displayWindow"}) {
      append_attribute_header(file, window, "box2i", 16);
      append(file, std::int32_t{0});
      append(file, std::int32_t{0});
      append(file, std::int32_t{width - 1});
      append(file, std::int32_t{height - 1});
   }
   append_attribute_header(file, "lineOrder", "lineOrder", 1);
   file.push_back(std::byte{0}); // INCREASING_Y
   append_attribute_header(file, "pixelAspectRatio", "float", 4);
   append(file, 1.0f);
   append_attribute_header(file, "screenWindowCenter", "v2f", 8);
   append(file, 0.0f);
   append(file, 0.0f);
   append_attribute_header(file, "screenWindowWidth", "float", 4);
   append(file, 1.0f);
   file.push_back(std::byte{0}); // fin de l'en-tête

   // Table des positions : une entrée de 64 bits par ligne, chaque ligne étant un bloc
   const auto line_bytes = static_cast<std::int32_t>(
      static_cast<std::size_t>(width) * 3 * sizeof(std::uint16_t)
   );
   const std::uint64_t first_line = file.size() + (static_cast<std::uint64_t>(height) * 8);
   for (int y = 0; y < height; ++y) {
      append(file, first_line + (static_cast<std::uint64_t>(y) * (8 + line_bytes)));
   }

   // Chaque ligne : son numéro, sa taille, puis tous les B, tous les G, tous les R
   for (int y = 0; y < height; ++y) {
      append(file, std::int32_t{y});
      append(file, line_bytes);
      const auto row = static_cast<std::size_t>(y) * static_cast<std::size_t>(width);
      for (const auto component : {&rt::Vec3::z, &rt::Vec3::y, &rt::Vec3::x}) {
         for (int x = 0; x < width; ++x) {
            append(file, to_half_bits(image[row + static_cast<std::size_t>(x)].*component));
         }
      }
   }

   std::ofstream stream(path, std::ios::binary);
   stream.write(
      reinterpret_cast<const char*>(file.data()),
      static_cast<std::streamsize>(file.size())
   );
   return stream.good();
}
