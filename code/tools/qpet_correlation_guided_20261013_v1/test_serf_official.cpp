#include "compressor_32/serf_qt_compressor_32.h"
#include "compressor_32/serf_xor_compressor_32.h"
#include "decompressor_32/serf_qt_decompressor_32.h"
#include "decompressor_32/serf_xor_decompressor_32.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message)
{
  if (!condition)
    throw std::runtime_error(message);
}

std::vector<float> telemetry_like_signal()
{
  auto values = std::vector<float>(2048);
  for (std::size_t index = 0; index < values.size(); ++index) {
    const auto channel = index / 128;
    const auto time = index % 128;
    values[index] = static_cast<float>(
        0.17 * static_cast<double>(channel) +
        std::sin(0.071 * static_cast<double>(time)) +
        0.13 * std::cos(0.013 * static_cast<double>(index)));
  }
  return values;
}

void verify(const std::vector<float>& original, const std::vector<float>& decoded, float bound)
{
  require(decoded.size() == original.size(), "Serf decoded the wrong value count");
  for (std::size_t index = 0; index < original.size(); ++index) {
    require(std::isfinite(decoded[index]), "Serf decoded a non-finite value");
    require(
        std::abs(static_cast<double>(original[index]) - decoded[index]) <=
            static_cast<double>(bound) + 1e-7,
        "Serf violated its requested point-wise bound");
  }
}

void test_xor32()
{
  const auto original = telemetry_like_signal();
  constexpr float bound = 0.025F;
  auto compressor = SerfXORCompressor32(static_cast<int>(original.size()), bound);
  for (const auto value : original)
    compressor.AddValue(value);
  compressor.Close();
  const auto bytes = compressor.compressed_bytes_last_block();
  require(bytes.length() > 0, "Serf-XOR32 returned an empty stream");
  auto decompressor = SerfXORDecompressor32();
  verify(original, decompressor.Decompress(bytes), bound);
}

void test_qt32()
{
  const auto original = telemetry_like_signal();
  constexpr float bound = 0.025F;
  auto compressor = SerfQtCompressor32(static_cast<int>(original.size()), bound);
  for (const auto value : original)
    compressor.AddValue(value);
  compressor.Close();
  const auto bytes = compressor.compressed_bytes();
  require(bytes.length() > 0, "Serf-Qt32 returned an empty stream");
  auto decompressor = SerfQtDecompressor32();
  verify(original, decompressor.Decompress(bytes), bound);
}

}  // namespace

int main()
{
  test_xor32();
  test_qt32();
  std::cout << "Official Serf float32 round-trip tests passed\n";
  return 0;
}
