#include "gpu/hello.h"

#include <iostream>

int main() {
   unsigned int const result = hello();
   std::cout << result << '\n';
}
