#include "QoZ/quantizer/IntegerQuantizer.hpp"
#include "QoZ/quantizer/QoIIntegerQuantizer.hpp"
#include "QoZ/utils/Config.hpp"
#include "QoZ/qoi/QoIInfo.hpp"
#include "QoZ/utils/Statistic.hpp"
#include "QoZ/api/impl/SZInterp.hpp"
#include "QoZ/compressor/SZInterpolationCompressor.hpp"
#include "QoZ/encoder/HuffmanEncoder.hpp"
#include "QoZ/lossless/Lossless_zstd.hpp"

#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message)
{
  if (!condition)
    throw std::runtime_error(message);
}

bool same_float_bits(float left, float right)
{
  return std::memcmp(&left, &right, sizeof(float)) == 0;
}

void test_linear_quantizer()
{
  QoZ::LinearQuantizer<float> quantizer(0.125, 32768);
  float positive = 1.0F;
  require(
      quantizer.quantize_and_overwrite(positive, 0.7F) == 32769,
      "linear quantizer changed a normal positive code");
  require(std::abs(positive - 0.95F) < 1.0e-6F, "linear positive replay changed");

  float negative = 0.4F;
  require(
      quantizer.quantize_and_overwrite(negative, 0.7F) == 32767,
      "linear quantizer changed a normal negative code");
  require(std::abs(negative - 0.45F) < 1.0e-6F, "linear negative replay changed");

  quantizer.clear();
  const auto maximum = std::numeric_limits<float>::max();
  require(
      quantizer.quantize(maximum, -maximum) == 0,
      "linear quantizer did not reject a non-representable scaled residual");
  float unpredictable = maximum;
  require(
      quantizer.quantize_and_overwrite(unpredictable, -maximum) == 0,
      "linear overwrite did not reject a non-representable scaled residual");
  require(same_float_bits(unpredictable, maximum), "linear unpredictable value changed");

  // Offset by one byte so every typed field is deliberately unaligned. The
  // native-endian wire bytes must remain unchanged without typed pointer IO.
  std::vector<unsigned char> storage(128, 0xa5U);
  auto* write = storage.data() + 1;
  auto* begin = write;
  quantizer.save(write);
  const auto encoded_size = static_cast<std::size_t>(write - begin);
  require(
      encoded_size == 1 + sizeof(double) + sizeof(int) + sizeof(std::size_t) +
              sizeof(float),
      "linear quantizer serialized an unexpected byte count");
  QoZ::LinearQuantizer<float> decoded;
  const auto* read = begin;
  auto remaining = encoded_size;
  decoded.load(read, remaining);
  require(decoded.get_radius() == 32768, "linear radius did not round-trip");
  require(decoded.get_eb() == 0.125, "linear error bound did not round-trip");
  require(
      same_float_bits(decoded.recover_unpred(), maximum),
      "linear unpredictable value did not round-trip");

  quantizer.clear();
  write = storage.data() + 1;
  begin = write;
  quantizer.save(write);
  require(
      static_cast<std::size_t>(write - begin) ==
          1 + sizeof(double) + sizeof(int) + sizeof(std::size_t),
      "empty linear quantizer serialized an unexpected byte count");
}

void test_variable_error_bound_quantizer()
{
  QoZ::VariableEBLinearQuantizer<float, float> quantizer(32768);
  float normal = 1.0F;
  require(
      quantizer.quantize_and_overwrite(normal, 0.7F, 0.125F) == 32769,
      "variable-EB quantizer changed a normal code");
  require(std::abs(normal - 0.95F) < 1.0e-6F, "variable-EB replay changed");

  quantizer.clear();
  const auto maximum = std::numeric_limits<float>::max();
  float unpredictable = maximum;
  require(
      quantizer.quantize_and_overwrite(unpredictable, -maximum, 0.125F) == 0,
      "variable-EB quantizer did not reject an overflowing residual");
  require(
      same_float_bits(unpredictable, maximum),
      "variable-EB unpredictable value changed");

  std::vector<unsigned char> storage(128, 0x5aU);
  auto* write = storage.data() + 1;
  auto* begin = write;
  quantizer.save(write);
  const auto encoded_size = static_cast<std::size_t>(write - begin);
  QoZ::VariableEBLinearQuantizer<float, float> decoded;
  const auto* read = begin;
  auto remaining = encoded_size;
  decoded.load(read, remaining);
  require(decoded.get_radius() == 32768, "variable-EB radius did not round-trip");
  require(
      same_float_bits(decoded.recover_unpred(), maximum),
      "variable-EB unpredictable value did not round-trip");
}

void test_log_error_bound_quantizer()
{
  QoZ::EBLogQuantizer<float> quantizer(1.0e-7, 2.0, 32, 1.0e-3F);
  const auto reference_code = quantizer.quantize(1.0e-5F);
  std::vector<unsigned char> storage(128, 0x3cU);
  auto* write = storage.data() + 1;
  auto* begin = write;
  quantizer.save(write);
  const auto encoded_size = static_cast<std::size_t>(write - begin);
  QoZ::EBLogQuantizer<float> decoded;
  const auto* read = begin;
  auto remaining = encoded_size;
  decoded.load(read, remaining);
  require(decoded.get_radius() == 32, "log-EB radius did not round-trip");
  require(
      decoded.quantize(1.0e-5F) == reference_code,
      "log-EB parameters did not round-trip");
}

void test_bounded_huffman_decode()
{
  const auto bins = std::vector<int>{0, 1, 0, 1, 1, 0, 1, 0};
  QoZ::HuffmanEncoder<int> encoder;
  encoder.preprocess_encode(bins, 0);
  auto payload = std::vector<unsigned char>(4096, 0U);
  auto* write = payload.data();
  encoder.save(write);
  encoder.encode(bins, write);
  encoder.postprocess_encode();
  payload.resize(static_cast<std::size_t>(write - payload.data()));

  const auto expect_format_error = [](auto&& action, const char* message) {
    auto rejected = false;
    try {
      action();
    }
    catch (const QoZ::CodecFormatError&) {
      rejected = true;
    }
    require(rejected, message);
  };

  const auto* valid_read = payload.data();
  auto valid_remaining = payload.size();
  QoZ::HuffmanEncoder<int> valid_decoder;
  valid_decoder.load(valid_read, valid_remaining);
  const auto tree_bytes = static_cast<std::size_t>(valid_read - payload.data());
  require(
      tree_bytes < payload.size() && valid_remaining == payload.size() - tree_bytes,
      "Huffman tree load did not debit its exact byte count");
  const auto decoded = valid_decoder.decode(
      valid_read, bins.size(), valid_remaining);
  require(decoded == bins, "bounded Huffman decode changed a valid symbol stream");
  valid_decoder.postprocess_decode();

  expect_format_error(
      [&] {
        const auto* read = payload.data();
        auto remaining = tree_bytes - 1U;
        QoZ::HuffmanEncoder<int> decoder;
        decoder.load(read, remaining);
      },
      "Huffman tree load accepted a physically truncated tree");

  expect_format_error(
      [&] {
        const auto* read = payload.data();
        auto remaining = payload.size();
        QoZ::HuffmanEncoder<int> decoder;
        decoder.load(read, remaining);
        require(remaining > sizeof(std::size_t), "Huffman fixture has no bit payload");
        --remaining;
        (void)decoder.decode(read, bins.size(), remaining);
      },
      "Huffman decode accepted an encoded length beyond physical bytes");

  // Keep the complete physical bit payload but make its declared logical
  // length too short.  This must reach the count != targetLength guard,
  // independently of the physical-remaining check above.
  auto logically_exhausted = payload;
  const auto zero_encoded_bytes = std::size_t{0};
  std::memcpy(
      logically_exhausted.data() + tree_bytes,
      &zero_encoded_bytes,
      sizeof(zero_encoded_bytes));
  expect_format_error(
      [&] {
        const auto* read = logically_exhausted.data();
        auto remaining = logically_exhausted.size();
        QoZ::HuffmanEncoder<int> decoder;
        decoder.load(read, remaining);
        (void)decoder.decode(read, bins.size(), remaining);
      },
      "Huffman decode accepted logical bit exhaustion before the target count");

  auto missing_child = payload;
  const auto node_count = QoZ::bytesToInt32_bigEndian(
      missing_child.data() + sizeof(int));
  require(node_count > 1 && node_count <= 256, "unexpected Huffman test tree width");
  const auto left_table = sizeof(int) + 2U * sizeof(int) + 1U;
  const auto right_table = left_table + static_cast<std::size_t>(node_count);
  missing_child[left_table] = 0U;
  missing_child[right_table] = 0U;
  expect_format_error(
      [&] {
        const auto* read = missing_child.data();
        auto remaining = missing_child.size();
        QoZ::HuffmanEncoder<int> decoder;
        decoder.load(read, remaining);
        (void)decoder.decode(read, bins.size(), remaining);
      },
      "Huffman decode followed an empty tree child");
}

void test_degenerate_2d_hpez_fails_closed()
{
  auto source = std::vector<float>{-0.75F, -0.125F, 0.5F, 1.0F, 0.25F};
  QoZ::Config config(1, source.size());
  config.cmprAlgo = QoZ::ALGO_INTERP;
  config.errorBoundMode = QoZ::EB_ABS;
  config.absErrorBound = 0.35;
  config.QoZ = 1;
  config.qoi = 0;
  config.verbose = false;
  config.rng = QoZ::data_range(source.data(), source.size());
  config.relErrorBound = config.absErrorBound / config.rng;
  auto tuning_data = source;
  Tuning<float, 2>(config, tuning_data.data());
  config.cmprAlgo = QoZ::ALGO_INTERP;
  config.qoi = 0;
  config.qoi_tuned = true;
  config.use_global_eb = true;
  config.absErrorBound = 0.35;

  using Compressor = QoZ::SZInterpolationCompressor<
      float,
      2,
      QoZ::LinearQuantizer<float>,
      QoZ::HuffmanEncoder<int>,
      QoZ::Lossless_zstd>;
  auto compressor = Compressor(
      QoZ::LinearQuantizer<float>(config.absErrorBound, 65536 / 2),
      QoZ::HuffmanEncoder<int>(),
      QoZ::Lossless_zstd());
  auto compressed_size = std::size_t{0};
  auto rejected = false;
  try {
    auto* compressed = compressor.compress(
        config, source.data(), compressed_size);
    delete[] compressed;
  }
  catch (const QoZ::CodecInvariantError&) {
    rejected = true;
  }
  require(
      rejected,
      "two-dimensional HPEZ did not reject a one-by-five symbol-count mismatch");
}

void test_defined_2d_hpez_round_trip()
{
  constexpr std::size_t channels = 2;
  constexpr std::size_t time = 128;
  constexpr double bound = 0.025;
  auto source = std::vector<float>(channels * time);
  auto state = std::uint32_t{0x6d2b79f5U};
  for (std::size_t channel = 0; channel < channels; ++channel) {
    for (std::size_t index = 0; index < time; ++index) {
      state = state * 1664525U + 1013904223U;
      const auto unit = static_cast<double>(state >> 8U) /
          static_cast<double>(1U << 24U);
      source[channel * time + index] = static_cast<float>(
          2.0 * unit - 1.0 + 0.125 * static_cast<double>(channel));
    }
  }
  QoZ::Config config(channels, time);
  config.cmprAlgo = QoZ::ALGO_INTERP;
  config.errorBoundMode = QoZ::EB_ABS;
  config.absErrorBound = bound;
  config.QoZ = 1;
  config.qoi = 0;
  config.verbose = false;
  config.rng = QoZ::data_range(source.data(), source.size());
  config.relErrorBound = config.absErrorBound / config.rng;
  auto tuning_data = source;
  Tuning<float, 2>(config, tuning_data.data());
  config.cmprAlgo = QoZ::ALGO_INTERP;
  config.qoi = 0;
  config.qoi_tuned = true;
  config.use_global_eb = true;
  config.absErrorBound = bound;

  using Compressor = QoZ::SZInterpolationCompressor<
      float,
      2,
      QoZ::LinearQuantizer<float>,
      QoZ::HuffmanEncoder<int>,
      QoZ::Lossless_zstd>;
  auto make_compressor = [] {
    return Compressor(
        QoZ::LinearQuantizer<float>(bound, 65536 / 2),
        QoZ::HuffmanEncoder<int>(),
        QoZ::Lossless_zstd());
  };
  auto in_place = source;
  auto compressed_size = std::size_t{0};
  auto encoder = make_compressor();
  auto* compressed = encoder.compress(
      config, in_place.data(), compressed_size);
  require(compressed != nullptr && compressed_size != 0, "defined HPEZ stream is empty");
  auto decoded = std::vector<float>(source.size());
  auto decoder = make_compressor();
  decoder.decompress(compressed, compressed_size, decoded.data());
  delete[] compressed;
  require(
      std::memcmp(
          decoded.data(), in_place.data(), decoded.size() * sizeof(float)) == 0,
      "defined HPEZ byte decode disagrees with encoder reconstruction");
}

}  // namespace

int main()
{
  try {
    test_linear_quantizer();
    test_variable_error_bound_quantizer();
    test_log_error_bound_quantizer();
    test_bounded_huffman_decode();
    test_degenerate_2d_hpez_fails_closed();
    test_defined_2d_hpez_round_trip();
    std::cout << "QPET upstream safety regression tests passed\n";
    return 0;
  }
  catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
