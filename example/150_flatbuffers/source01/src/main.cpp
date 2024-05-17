#include "image_generated.h"
#include <fstream>

int main(int argc, char **argv) {
  auto width{256};
  auto height{371};
  auto imageData{std::vector<uint8_t>(width * height, 128)};
  auto builder{flatbuffers::FlatBufferBuilder()};
  auto image{MyImage::CreateImageDirect(builder, width, height, &imageData)};
  builder.Finish(image);
  auto output{std::ofstream("image.bin", std::ios::binary)};
  output.write(reinterpret_cast<const char *>(builder.data()),
               builder.GetSize());
  output.close();
  return 0;
}
