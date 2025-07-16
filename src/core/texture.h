#pragma once

#include <cstdint>

namespace rt {

struct Texture {
   std::uint32_t offset;
   std::uint32_t width;
   std::uint32_t height;
};

} // namespace rt
