#include "app/manifest.h"

#include <chrono>
#include <format>
#include <fstream>
#include <string>
#include <string_view>

namespace {

// Chaîne JSON : guillemets, barres obliques inverses et caractères de contrôle échappés
std::string json_string(std::string_view text) {
   std::string result = "\"";
   for (const char c : text) {
      switch (c) {
         case '"':
            result += "\\\"";
            break;
         case '\\':
            result += "\\\\";
            break;
         case '\n':
            result += "\\n";
            break;
         case '\t':
            result += "\\t";
            break;
         default:
            if (static_cast<unsigned char>(c) < 0x20) {
               result += std::format("\\u{:04x}", static_cast<unsigned char>(c));
            } else {
               result += c;
            }
      }
   }
   return result + "\"";
}

// Version CUDA encodée en entier (13030) -> « 13.3 »
std::string cuda_version(int encoded) {
   return std::format("{}.{}", encoded / 1000, (encoded % 1000) / 10);
}

} // namespace

std::string current_local_time() {
   const auto now = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
   const std::chrono::zoned_time local{std::chrono::current_zone(), now};
   return std::format("{:%Y-%m-%dT%H:%M:%S%z}", local);
}

bool write_manifest(const std::string& path, const RenderManifest& m) {
   const double seconds = m.gpu_milliseconds / 1000.0;
   const double mrays_per_second = m.gpu_milliseconds > 0.0
                                      ? static_cast<double>(m.rays) / (m.gpu_milliseconds * 1e3)
                                      : 0.0;
   constexpr double kGibibyte = 1024.0 * 1024.0 * 1024.0;

   std::ofstream file(path);
   file << "{\n";
   file << std::format("  \"date\": {},\n", json_string(m.date));
   file << "  \"machine\": {\n";
   file << std::format("    \"gpu\": {},\n", json_string(m.device.name));
   file << std::format(
      "    \"compute_capability\": \"{}.{}\",\n",
      m.device.compute_capability_major,
      m.device.compute_capability_minor
   );
   file << std::format(
      "    \"gpu_memory_gib\": {:.1f},\n",
      static_cast<double>(m.device.memory_bytes) / kGibibyte
   );
   file << std::format(
      "    \"driver_cuda_version\": \"{}\",\n",
      cuda_version(m.device.driver_version)
   );
   file << std::format(
      "    \"runtime_cuda_version\": \"{}\"\n",
      cuda_version(m.device.runtime_version)
   );
   file << "  },\n";
   file << "  \"render\": {\n";
   file << std::format("    \"scene\": {},\n", json_string(m.scene));
   file << std::format("    \"width\": {},\n", m.width);
   file << std::format("    \"height\": {},\n", m.height);
   file << std::format("    \"samples_per_pixel\": {},\n", m.samples_per_pixel);
   file << std::format("    \"gpu_time_seconds\": {:.3f},\n", seconds);
   file << std::format("    \"rays\": {},\n", m.rays);
   file << std::format("    \"mrays_per_second\": {:.1f}\n", mrays_per_second);
   file << "  },\n";
   file << "  \"settings\": {\n";
   file << std::format("    \"mis\": {},\n", m.mis);
   file << std::format("    \"max_bounces\": {},\n", m.max_bounces);
   file << std::format("    \"russian_roulette_start\": {},\n", m.russian_roulette_start);
   file << std::format("    \"bvh\": {},\n", json_string(m.bvh));
   file << std::format("    \"tone_mapping\": {},\n", json_string(m.tone_mapping));
   file << std::format("    \"exposure\": {}\n", m.exposure);
   file << "  }\n";
   file << "}\n";
   return file.good();
}
