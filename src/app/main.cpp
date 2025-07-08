#include "gpu/hello.h"

#include <iostream>

int main() {
   const unsigned int result = hello();
   std::cout << result << '\n';
}
