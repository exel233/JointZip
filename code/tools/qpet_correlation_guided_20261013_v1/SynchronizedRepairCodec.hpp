#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qpet_correlation::ssrc {

struct BitStream {
  std::uint64_t bit_count = 0;
  std::vector<std::uint8_t> bytes;
};

inline bool operator==(const BitStream& left, const BitStream& right)
{
  return left.bit_count == right.bit_count && left.bytes == right.bytes;
}

inline bool operator!=(const BitStream& left, const BitStream& right)
{
  return !(left == right);
}

namespace detail {

inline std::size_t framed_byte_count(std::uint64_t bit_count)
{
  if (bit_count >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) - 7U)
    throw std::runtime_error("SSRC bit count exceeds size_t framing");
  return static_cast<std::size_t>((bit_count + 7U) / 8U);
}

inline void validate_bit_stream(const BitStream& stream)
{
  const auto expected = framed_byte_count(stream.bit_count);
  if (stream.bytes.size() != expected)
    throw std::runtime_error("SSRC bitstream byte count is not canonical");
  const auto used = static_cast<unsigned int>(stream.bit_count % 8U);
  if (used != 0U && !stream.bytes.empty()) {
    const auto padding_mask = static_cast<std::uint8_t>((1U << (8U - used)) - 1U);
    if ((stream.bytes.back() & padding_mask) != 0U)
      throw std::runtime_error("SSRC bitstream padding is nonzero");
  }
}

class BitWriter {
 public:
  void write(bool bit)
  {
    const auto byte_index = static_cast<std::size_t>(bit_count_ / 8U);
    const auto bit_in_byte = static_cast<unsigned int>(bit_count_ % 8U);
    if (byte_index == bytes_.size())
      bytes_.push_back(0U);
    if (bit)
      bytes_[byte_index] |= static_cast<std::uint8_t>(1U << (7U - bit_in_byte));
    if (bit_count_ == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("SSRC bit count overflow");
    ++bit_count_;
  }

  BitStream finish() &&
  {
    return {bit_count_, std::move(bytes_)};
  }

 private:
  std::uint64_t bit_count_ = 0;
  std::vector<std::uint8_t> bytes_;
};

class BitReader {
 public:
  explicit BitReader(const BitStream& stream) : stream_(stream)
  {
    validate_bit_stream(stream_);
  }

  // Arithmetic decoding conventionally primes a fixed-width code register.
  // Bits beyond the explicitly framed stream are deterministic zero fill.
  // Public decoders subsequently re-encode and require exact canonical bytes,
  // so a shortened or extended frame cannot be accepted through this fill.
  bool read_or_zero()
  {
    if (position_ >= stream_.bit_count)
      return false;
    const auto byte_index = static_cast<std::size_t>(position_ / 8U);
    const auto bit_in_byte = static_cast<unsigned int>(position_ % 8U);
    const auto bit = (stream_.bytes[byte_index] &
                      static_cast<std::uint8_t>(1U << (7U - bit_in_byte))) != 0U;
    ++position_;
    return bit;
  }

 private:
  const BitStream& stream_;
  std::uint64_t position_ = 0;
};

class BinaryArithmeticEncoder {
 public:
  void encode(bool symbol, std::uint64_t zero_frequency,
              std::uint64_t one_frequency)
  {
    validate_frequencies(zero_frequency, one_frequency);
    const auto total = zero_frequency + one_frequency;
    const auto span = static_cast<std::uint64_t>(high_) - low_ + 1U;
    const auto old_low = static_cast<std::uint64_t>(low_);
    const auto cumulative_low = symbol ? zero_frequency : 0U;
    const auto cumulative_high = symbol ? total : zero_frequency;
    const auto next_low = old_low + (span * cumulative_low) / total;
    const auto next_high = old_low + (span * cumulative_high) / total - 1U;
    if (next_low > next_high ||
        next_high > std::numeric_limits<std::uint32_t>::max())
      throw std::logic_error("SSRC arithmetic encoder interval collapsed");
    low_ = static_cast<std::uint32_t>(next_low);
    high_ = static_cast<std::uint32_t>(next_high);
    renormalize();
  }

  BitStream finish()
  {
    if (finished_)
      throw std::logic_error("SSRC arithmetic encoder was finished twice");
    finished_ = true;
    if (pending_ == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("SSRC arithmetic pending-bit overflow");
    ++pending_;
    if (low_ < kQuarter)
      emit_with_pending(false);
    else
      emit_with_pending(true);
    return std::move(writer_).finish();
  }

 private:
  static constexpr std::uint32_t kHalf = 0x80000000U;
  static constexpr std::uint32_t kQuarter = 0x40000000U;
  static constexpr std::uint32_t kThreeQuarters = 0xc0000000U;

  static void validate_frequencies(std::uint64_t zero, std::uint64_t one)
  {
    if (zero == 0U || one == 0U ||
        zero > std::numeric_limits<std::uint32_t>::max() - one)
      throw std::invalid_argument("SSRC arithmetic frequencies are invalid");
  }

  void emit_with_pending(bool bit)
  {
    writer_.write(bit);
    while (pending_ != 0U) {
      writer_.write(!bit);
      --pending_;
    }
  }

  void renormalize()
  {
    for (;;) {
      if (high_ < kHalf) {
        emit_with_pending(false);
      }
      else if (low_ >= kHalf) {
        emit_with_pending(true);
        low_ -= kHalf;
        high_ -= kHalf;
      }
      else if (low_ >= kQuarter && high_ < kThreeQuarters) {
        if (pending_ == std::numeric_limits<std::uint64_t>::max())
          throw std::overflow_error("SSRC arithmetic pending-bit overflow");
        ++pending_;
        low_ -= kQuarter;
        high_ -= kQuarter;
      }
      else {
        break;
      }
      low_ = static_cast<std::uint32_t>(low_ << 1U);
      high_ = static_cast<std::uint32_t>((high_ << 1U) | 1U);
    }
  }

  std::uint32_t low_ = 0U;
  std::uint32_t high_ = std::numeric_limits<std::uint32_t>::max();
  std::uint64_t pending_ = 0U;
  BitWriter writer_;
  bool finished_ = false;
};

class BinaryArithmeticDecoder {
 public:
  explicit BinaryArithmeticDecoder(const BitStream& stream) : reader_(stream)
  {
    for (unsigned int bit = 0; bit < 32U; ++bit)
      value_ = static_cast<std::uint32_t>(
          (value_ << 1U) | static_cast<std::uint32_t>(reader_.read_or_zero()));
  }

  bool decode(std::uint64_t zero_frequency, std::uint64_t one_frequency)
  {
    if (zero_frequency == 0U || one_frequency == 0U ||
        zero_frequency >
            std::numeric_limits<std::uint32_t>::max() - one_frequency)
      throw std::invalid_argument("SSRC arithmetic frequencies are invalid");
    if (value_ < low_ || value_ > high_)
      throw std::runtime_error("SSRC arithmetic code is outside its interval");
    const auto total = zero_frequency + one_frequency;
    const auto span = static_cast<std::uint64_t>(high_) - low_ + 1U;
    const auto relative = static_cast<std::uint64_t>(value_) - low_ + 1U;
    const auto scaled = (relative * total - 1U) / span;
    const auto symbol = scaled >= zero_frequency;
    const auto cumulative_low = symbol ? zero_frequency : 0U;
    const auto cumulative_high = symbol ? total : zero_frequency;
    const auto old_low = static_cast<std::uint64_t>(low_);
    const auto next_low = old_low + (span * cumulative_low) / total;
    const auto next_high = old_low + (span * cumulative_high) / total - 1U;
    if (next_low > next_high ||
        next_high > std::numeric_limits<std::uint32_t>::max())
      throw std::runtime_error("SSRC arithmetic decoder interval collapsed");
    low_ = static_cast<std::uint32_t>(next_low);
    high_ = static_cast<std::uint32_t>(next_high);
    renormalize();
    return symbol;
  }

 private:
  static constexpr std::uint32_t kHalf = 0x80000000U;
  static constexpr std::uint32_t kQuarter = 0x40000000U;
  static constexpr std::uint32_t kThreeQuarters = 0xc0000000U;

  void renormalize()
  {
    for (;;) {
      if (high_ < kHalf) {
        // No offset.
      }
      else if (low_ >= kHalf) {
        value_ -= kHalf;
        low_ -= kHalf;
        high_ -= kHalf;
      }
      else if (low_ >= kQuarter && high_ < kThreeQuarters) {
        value_ -= kQuarter;
        low_ -= kQuarter;
        high_ -= kQuarter;
      }
      else {
        break;
      }
      low_ = static_cast<std::uint32_t>(low_ << 1U);
      high_ = static_cast<std::uint32_t>((high_ << 1U) | 1U);
      value_ = static_cast<std::uint32_t>(
          (value_ << 1U) | static_cast<std::uint32_t>(reader_.read_or_zero()));
    }
  }

  BitReader reader_;
  std::uint32_t low_ = 0U;
  std::uint32_t high_ = std::numeric_limits<std::uint32_t>::max();
  std::uint32_t value_ = 0U;
};

inline std::uint8_t zigzag(std::int8_t value)
{
  if (value == std::numeric_limits<std::int8_t>::min())
    throw std::invalid_argument("SSRC value code -128 is outside the repair grid");
  const auto promoted = static_cast<int>(value);
  return static_cast<std::uint8_t>(
      promoted >= 0 ? 2 * promoted : 2 * (-promoted) - 1);
}

inline std::int8_t unzigzag(std::uint8_t value)
{
  if (value == std::numeric_limits<std::uint8_t>::max())
    throw std::runtime_error("SSRC transformed value restores illegal ZigZag 255");
  const auto decoded = (value & 1U) == 0U
      ? static_cast<int>(value / 2U)
      : -static_cast<int>((static_cast<unsigned int>(value) + 1U) / 2U);
  return static_cast<std::int8_t>(decoded);
}

inline unsigned int floor_log2(unsigned int value)
{
  if (value == 0U)
    throw std::invalid_argument("SSRC log2 input is zero");
  auto result = 0U;
  while (value > 1U) {
    value >>= 1U;
    ++result;
  }
  return result;
}

inline std::size_t previous_value_class(bool has_previous, std::uint8_t previous)
{
  if (!has_previous)
    return 0U;
  const auto code = unzigzag(previous);
  if (code == 0)
    return 1U;
  const auto promoted = static_cast<int>(code);
  const auto magnitude = static_cast<unsigned int>(promoted < 0 ? -promoted : promoted);
  const auto bucket = std::min(7U, floor_log2(magnitude));
  return 2U + static_cast<std::size_t>(promoted < 0 ? 8U : 0U) + bucket;
}

struct AdaptiveCounts {
  std::uint16_t zero = 1U;
  std::uint16_t one = 1U;
};

class AdaptivePrefixModel {
 public:
  static constexpr std::size_t kContextCount = 18U;
  static constexpr std::size_t kNodeCount = 256U;
  static constexpr std::uint32_t kRescaleTotal = 32768U;

  const AdaptiveCounts& at(std::size_t context, std::size_t node) const
  {
    if (context >= kContextCount || node == 0U || node >= kNodeCount)
      throw std::logic_error("SSRC adaptive model index is invalid");
    return counts_[context * kNodeCount + node];
  }

  void update(std::size_t context, std::size_t node, bool symbol)
  {
    auto& counts = counts_.at(context * kNodeCount + node);
    auto& selected = symbol ? counts.one : counts.zero;
    if (selected == std::numeric_limits<std::uint16_t>::max())
      throw std::overflow_error("SSRC adaptive frequency overflow");
    ++selected;
    if (static_cast<std::uint32_t>(counts.zero) + counts.one >= kRescaleTotal) {
      counts.zero = static_cast<std::uint16_t>((counts.zero + 1U) / 2U);
      counts.one = static_cast<std::uint16_t>((counts.one + 1U) / 2U);
      ++rescales_;
    }
  }

  std::size_t rescales() const noexcept { return rescales_; }

 private:
  std::array<AdaptiveCounts, kContextCount * kNodeCount> counts_{};
  std::size_t rescales_ = 0U;
};

inline void encode_byte(BinaryArithmeticEncoder& encoder,
                        AdaptivePrefixModel& model, std::size_t context,
                        std::uint8_t byte)
{
  auto node = std::size_t{1};
  for (unsigned int depth = 0; depth < 8U; ++depth) {
    const auto bit = (byte & static_cast<std::uint8_t>(1U << (7U - depth))) != 0U;
    const auto counts = model.at(context, node);
    encoder.encode(bit, counts.zero, counts.one);
    model.update(context, node, bit);
    node = node * 2U + static_cast<std::size_t>(bit);
  }
}

inline std::uint8_t decode_byte(BinaryArithmeticDecoder& decoder,
                                AdaptivePrefixModel& model,
                                std::size_t context)
{
  auto node = std::size_t{1};
  auto byte = std::uint8_t{0};
  for (unsigned int depth = 0; depth < 8U; ++depth) {
    const auto counts = model.at(context, node);
    const auto bit = decoder.decode(counts.zero, counts.one);
    if (bit)
      byte |= static_cast<std::uint8_t>(1U << (7U - depth));
    model.update(context, node, bit);
    node = node * 2U + static_cast<std::size_t>(bit);
  }
  return byte;
}

}  // namespace detail

inline BitStream encode_subset(
    std::size_t time, const std::vector<std::size_t>& positions)
{
  if (time == 0U || time > std::numeric_limits<std::uint32_t>::max())
    throw std::invalid_argument("SSRC subset time is outside [1,uint32_max]");
  if (positions.size() > time)
    throw std::invalid_argument("SSRC subset has too many positions");
  for (std::size_t index = 0; index < positions.size(); ++index) {
    if (positions[index] >= time ||
        (index != 0U && positions[index - 1U] >= positions[index]))
      throw std::invalid_argument("SSRC subset positions are not strictly increasing");
  }
  if (positions.empty() || positions.size() == time)
    return {};

  detail::BinaryArithmeticEncoder encoder;
  auto remaining_selected = positions.size();
  auto selected_index = std::size_t{0};
  for (std::size_t position = 0; position < time; ++position) {
    const auto remaining_positions = time - position;
    if (remaining_selected == 0U || remaining_selected == remaining_positions)
      break;
    const auto selected = selected_index < positions.size() &&
        positions[selected_index] == position;
    encoder.encode(
        selected,
        static_cast<std::uint64_t>(remaining_positions - remaining_selected),
        static_cast<std::uint64_t>(remaining_selected));
    if (selected) {
      ++selected_index;
      --remaining_selected;
    }
  }
  return encoder.finish();
}

inline std::vector<std::size_t> decode_subset(
    std::size_t time, std::size_t selected_count, const BitStream& stream)
{
  detail::validate_bit_stream(stream);
  if (time == 0U || time > std::numeric_limits<std::uint32_t>::max() ||
      selected_count > time)
    throw std::invalid_argument("SSRC subset decode dimensions are invalid");
  if (selected_count == 0U || selected_count == time) {
    if (stream.bit_count != 0U)
      throw std::runtime_error("SSRC deterministic subset has a nonempty bitstream");
    auto result = std::vector<std::size_t>();
    if (selected_count == time) {
      result.resize(time);
      for (std::size_t index = 0; index < time; ++index)
        result[index] = index;
    }
    return result;
  }

  detail::BinaryArithmeticDecoder decoder(stream);
  auto result = std::vector<std::size_t>();
  result.reserve(selected_count);
  auto remaining_selected = selected_count;
  for (std::size_t position = 0; position < time; ++position) {
    const auto remaining_positions = time - position;
    bool selected = false;
    if (remaining_selected == remaining_positions)
      selected = true;
    else if (remaining_selected != 0U)
      selected = decoder.decode(
          static_cast<std::uint64_t>(remaining_positions - remaining_selected),
          static_cast<std::uint64_t>(remaining_selected));
    if (selected) {
      result.push_back(position);
      --remaining_selected;
    }
  }
  if (result.size() != selected_count || encode_subset(time, result) != stream)
    throw std::runtime_error("SSRC subset bitstream is not canonical");
  return result;
}

inline void decode_subset_into(
    std::size_t time, std::size_t selected_count, const BitStream& stream,
    std::vector<std::size_t>& output)
{
  auto decoded = decode_subset(time, selected_count, stream);
  output.swap(decoded);
}

enum class ValueTransform : std::uint8_t {
  raw_zigzag = 0,
  previous_xor = 1,
};

struct EncodedValues {
  BitStream stream;
  std::size_t model_rescales = 0U;
};

struct DecodedValues {
  std::vector<std::int8_t> codes;
  std::size_t model_rescales = 0U;
};

inline EncodedValues encode_values(
    const std::vector<std::int8_t>& codes, ValueTransform transform)
{
  if (transform != ValueTransform::raw_zigzag &&
      transform != ValueTransform::previous_xor)
    throw std::invalid_argument("SSRC value transform is invalid");
  if (codes.empty())
    return {};

  detail::BinaryArithmeticEncoder encoder;
  detail::AdaptivePrefixModel model;
  auto previous = std::uint8_t{0};
  auto has_previous = false;
  for (const auto code : codes) {
    const auto zigzag = detail::zigzag(code);
    const auto transformed = transform == ValueTransform::raw_zigzag
        ? zigzag
        : static_cast<std::uint8_t>(zigzag ^ previous);
    const auto context = detail::previous_value_class(has_previous, previous);
    detail::encode_byte(encoder, model, context, transformed);
    previous = zigzag;
    has_previous = true;
  }
  return {encoder.finish(), model.rescales()};
}

inline DecodedValues decode_values(
    const BitStream& stream, std::size_t expected_count, ValueTransform transform)
{
  detail::validate_bit_stream(stream);
  if (transform != ValueTransform::raw_zigzag &&
      transform != ValueTransform::previous_xor)
    throw std::invalid_argument("SSRC value transform is invalid");
  if (expected_count == 0U) {
    if (stream.bit_count != 0U)
      throw std::runtime_error("SSRC empty value sequence has a nonempty bitstream");
    return {};
  }
  if (expected_count > std::numeric_limits<std::uint32_t>::max())
    throw std::invalid_argument("SSRC value count exceeds uint32 framing");

  detail::BinaryArithmeticDecoder decoder(stream);
  detail::AdaptivePrefixModel model;
  auto codes = std::vector<std::int8_t>();
  codes.reserve(expected_count);
  auto previous = std::uint8_t{0};
  auto has_previous = false;
  for (std::size_t index = 0; index < expected_count; ++index) {
    const auto context = detail::previous_value_class(has_previous, previous);
    const auto transformed = detail::decode_byte(decoder, model, context);
    const auto zigzag = transform == ValueTransform::raw_zigzag
        ? transformed
        : static_cast<std::uint8_t>(transformed ^ previous);
    codes.push_back(detail::unzigzag(zigzag));
    previous = zigzag;
    has_previous = true;
  }
  const auto canonical = encode_values(codes, transform);
  if (canonical.stream != stream)
    throw std::runtime_error("SSRC value bitstream is not canonical");
  return {std::move(codes), model.rescales()};
}

inline void decode_values_into(
    const BitStream& stream, std::size_t expected_count, ValueTransform transform,
    std::vector<std::int8_t>& output)
{
  auto decoded = decode_values(stream, expected_count, transform);
  output.swap(decoded.codes);
}

}  // namespace qpet_correlation::ssrc
