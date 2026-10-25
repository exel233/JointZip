#include "SynchronizedRepairCodec.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using qpet_correlation::ssrc::BitStream;
using qpet_correlation::ssrc::ValueTransform;

void require(bool condition, const std::string& message)
{
  if (!condition)
    throw std::runtime_error(message);
}

template <class Callable>
void require_rejected(Callable&& callable, const std::string& label)
{
  auto rejected = false;
  try {
    callable();
  }
  catch (const std::exception&) {
    rejected = true;
  }
  require(rejected, "SSRC accepted " + label);
}

std::string hex(const BitStream& stream)
{
  auto output = std::ostringstream();
  output << stream.bit_count << ':';
  for (const auto byte : stream.bytes)
    output << std::hex << std::setfill('0') << std::setw(2)
           << static_cast<unsigned int>(byte);
  return output.str();
}

void test_subset_golden_and_boundaries()
{
  using namespace qpet_correlation::ssrc;
  const auto positions = std::vector<std::size_t>{0, 3, 4, 9, 15};
  const auto encoded = encode_subset(17, positions);
  const auto expected = std::string("14:dc50");
  require(hex(encoded) == expected,
          "subset golden mismatch: observed " + hex(encoded));
  require(decode_subset(17, positions.size(), encoded) == positions,
          "subset golden did not round trip");

  require(encode_subset(9, {}).bit_count == 0U,
          "empty subset emitted arithmetic bits");
  const auto all = std::vector<std::size_t>{0, 1, 2, 3, 4, 5, 6, 7, 8};
  require(encode_subset(9, all).bit_count == 0U,
          "all-selected subset emitted arithmetic bits");
  require(decode_subset(9, 0, {}) == std::vector<std::size_t>{},
          "empty deterministic subset failed");
  require(decode_subset(9, 9, {}) == all,
          "all-selected deterministic subset failed");

  for (const auto time :
       std::array<std::size_t, 5>{3U, 8U, 9U, 127U, 128U}) {
    for (const auto count :
         std::array<std::size_t, 2>{std::size_t{1}, time - 1U}) {
      auto fixture = std::vector<std::size_t>();
      for (std::size_t index = 0; index < count; ++index)
        fixture.push_back((index * time) / count);
      const auto stream = encode_subset(time, fixture);
      require(decode_subset(time, count, stream) == fixture,
              "subset boundary fixture failed");
    }
  }
}

void test_subset_random_reference()
{
  using namespace qpet_correlation::ssrc;
  auto rng = std::mt19937(0x51c0deU);
  for (std::size_t trial = 0; trial < 1000; ++trial) {
    const auto time = std::size_t{1} + rng() % 200U;
    auto universe = std::vector<std::size_t>(time);
    for (std::size_t index = 0; index < time; ++index)
      universe[index] = index;
    std::shuffle(universe.begin(), universe.end(), rng);
    const auto count = static_cast<std::size_t>(rng() % (time + 1U));
    universe.resize(count);
    std::sort(universe.begin(), universe.end());
    const auto first = encode_subset(time, universe);
    const auto second = encode_subset(time, universe);
    require(first == second, "subset encoder is nondeterministic");
    require(decode_subset(time, count, first) == universe,
            "random subset differs from reference");
  }
}

void test_values_golden_roundtrip_and_rescale()
{
  using namespace qpet_correlation::ssrc;
  const auto codes = std::vector<std::int8_t>{
      1, 1, 1, -1, -1, 0, 7, 7, -7, 2, 2, 2, 2, 127, -127};
  const auto raw = encode_values(codes, ValueTransform::raw_zigzag);
  const auto xored = encode_values(codes, ValueTransform::previous_xor);
  require(hex(raw.stream) == "104:020207b6f2215d9e343dfb8adc",
          "raw value golden mismatch: observed " + hex(raw.stream));
  require(hex(xored.stream) == "104:0200018e3b2b4a1bd1d995a02e",
          "XOR value golden mismatch: observed " + hex(xored.stream));
  require(decode_values(raw.stream, codes.size(), ValueTransform::raw_zigzag).codes ==
              codes,
          "raw value golden did not round trip");
  require(decode_values(
              xored.stream, codes.size(), ValueTransform::previous_xor).codes ==
              codes,
          "XOR value golden did not round trip");

  auto repeated = std::vector<std::int8_t>(70000, 1);
  const auto large = encode_values(repeated, ValueTransform::previous_xor);
  const auto decoded = decode_values(
      large.stream, repeated.size(), ValueTransform::previous_xor);
  require(decoded.codes == repeated, "rescale fixture did not round trip");
  require(large.model_rescales > 0U &&
              decoded.model_rescales == large.model_rescales,
          "adaptive rescale was not exercised symmetrically");
}

void test_value_alphabet_and_random_reference()
{
  using namespace qpet_correlation::ssrc;
  auto alphabet = std::vector<std::int8_t>();
  for (int value = -127; value <= 127; ++value)
    alphabet.push_back(static_cast<std::int8_t>(value));
  for (const auto transform :
       {ValueTransform::raw_zigzag, ValueTransform::previous_xor}) {
    const auto encoded = encode_values(alphabet, transform);
    require(decode_values(encoded.stream, alphabet.size(), transform).codes ==
                alphabet,
            "complete repair alphabet did not round trip");
  }

  auto rng = std::mt19937(0xa11ceU);
  for (std::size_t trial = 0; trial < 300; ++trial) {
    const auto count = std::size_t{1} + rng() % 400U;
    auto codes = std::vector<std::int8_t>();
    codes.reserve(count);
    for (std::size_t index = 0; index < count; ++index)
      codes.push_back(static_cast<std::int8_t>(
          static_cast<int>(rng() % 255U) - 127));
    for (const auto transform :
         {ValueTransform::raw_zigzag, ValueTransform::previous_xor}) {
      const auto first = encode_values(codes, transform);
      const auto second = encode_values(codes, transform);
      require(first.stream == second.stream &&
                  first.model_rescales == second.model_rescales,
              "value encoder is nondeterministic");
      require(decode_values(first.stream, count, transform).codes == codes,
              "random value sequence differs from reference");
    }
  }
}

void test_strict_framing_and_transactionality()
{
  using namespace qpet_correlation::ssrc;
  const auto positions = std::vector<std::size_t>{1, 4, 8, 9, 15, 22};
  const auto subset = encode_subset(29, positions);
  auto output_positions = std::vector<std::size_t>{999U};

  auto trailing = subset;
  trailing.bytes.push_back(0U);
  require_rejected(
      [&] { decode_subset_into(29, positions.size(), trailing, output_positions); },
      "a trailing subset byte");
  require(output_positions == std::vector<std::size_t>{999U},
          "failed subset decode was not transactional");

  if (!subset.bytes.empty()) {
    auto truncated = subset;
    truncated.bytes.pop_back();
    require_rejected(
        [&] { decode_subset(29, positions.size(), truncated); },
        "a truncated subset byte field");
  }
  auto extra_bit = subset;
  ++extra_bit.bit_count;
  if (extra_bit.bytes.size() < (extra_bit.bit_count + 7U) / 8U)
    extra_bit.bytes.push_back(0U);
  require_rejected(
      [&] { decode_subset(29, positions.size(), extra_bit); },
      "a noncanonical extra subset bit");

  if (subset.bit_count > 1U) {
    auto short_bits = subset;
    --short_bits.bit_count;
    short_bits.bytes.resize(
        static_cast<std::size_t>((short_bits.bit_count + 7U) / 8U));
    const auto used = static_cast<unsigned int>(short_bits.bit_count % 8U);
    if (used != 0U)
      short_bits.bytes.back() &=
          static_cast<std::uint8_t>(0xffU << (8U - used));
    require_rejected(
        [&] { decode_subset(29, positions.size(), short_bits); },
        "a shortened subset bit field");
  }

  auto nonzero_padding = subset;
  if (nonzero_padding.bit_count % 8U != 0U) {
    nonzero_padding.bytes.back() |= 1U;
    require_rejected(
        [&] { decode_subset(29, positions.size(), nonzero_padding); },
        "nonzero subset padding");
  }

  const auto codes = std::vector<std::int8_t>{1, -2, 3, 3, 0, 7, -7};
  const auto values = encode_values(codes, ValueTransform::previous_xor).stream;
  auto output_codes = std::vector<std::int8_t>{42};
  auto value_trailing = values;
  value_trailing.bytes.push_back(0U);
  require_rejected(
      [&] {
        decode_values_into(
            value_trailing, codes.size(), ValueTransform::previous_xor,
            output_codes);
      },
      "a trailing value byte");
  require(output_codes == std::vector<std::int8_t>{42},
          "failed value decode was not transactional");
  if (values.bit_count > 1U) {
    auto short_values = values;
    --short_values.bit_count;
    short_values.bytes.resize(
        static_cast<std::size_t>((short_values.bit_count + 7U) / 8U));
    const auto used = static_cast<unsigned int>(short_values.bit_count % 8U);
    if (used != 0U)
      short_values.bytes.back() &=
          static_cast<std::uint8_t>(0xffU << (8U - used));
    require_rejected(
        [&] {
          decode_values_into(
              short_values, codes.size(), ValueTransform::previous_xor,
              output_codes);
        },
        "a shortened value bit field");
    require(output_codes == std::vector<std::int8_t>{42},
            "shortened value decode was not transactional");
  }
  auto value_extra_bit = values;
  ++value_extra_bit.bit_count;
  if (value_extra_bit.bytes.size() < (value_extra_bit.bit_count + 7U) / 8U)
    value_extra_bit.bytes.push_back(0U);
  require_rejected(
      [&] {
        decode_values(
            value_extra_bit, codes.size(), ValueTransform::previous_xor);
      },
      "a noncanonical extra value bit");
  if (values.bit_count % 8U != 0U) {
    auto value_padding = values;
    value_padding.bytes.back() |= 1U;
    require_rejected(
        [&] {
          decode_values(value_padding, codes.size(), ValueTransform::previous_xor);
        },
        "nonzero value padding");
  }
  require_rejected(
      [&] { decode_values(values, codes.size() + 1U, ValueTransform::previous_xor); },
      "a mismatched value count");
  require_rejected(
      [&] {
        encode_values(
            {std::numeric_limits<std::int8_t>::min()},
            ValueTransform::raw_zigzag);
      },
      "the illegal -128 repair code");
  require_rejected(
      [&] { decode_subset(5, 6, {}); },
      "a subset count greater than time");
  require_rejected(
      [&] { encode_subset(5, {1U, 1U}); },
      "duplicate subset positions");
  require_rejected(
      [&] { encode_subset(5, {5U}); },
      "an out-of-range subset position");
}

void test_illegal_restored_zigzag_255()
{
  using namespace qpet_correlation::ssrc;
  // Construct the canonical arithmetic stream for one raw transformed byte
  // 255. Public encoding cannot create it because -128 is outside the repair
  // grid, while a hostile payload can.
  detail::BinaryArithmeticEncoder encoder;
  detail::AdaptivePrefixModel model;
  detail::encode_byte(encoder, model, 0U, 255U);
  const auto hostile = encoder.finish();
  require_rejected(
      [&] { decode_values(hostile, 1, ValueTransform::raw_zigzag); },
      "a transformed value restoring ZigZag 255");
}

}  // namespace

int main()
{
  try {
    test_subset_golden_and_boundaries();
    test_subset_random_reference();
    test_values_golden_roundtrip_and_rescale();
    test_value_alphabet_and_random_reference();
    test_strict_framing_and_transactionality();
    test_illegal_restored_zigzag_255();
    std::cout << "SSRC arithmetic/subset/value tests passed\n";
    return 0;
  }
  catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
