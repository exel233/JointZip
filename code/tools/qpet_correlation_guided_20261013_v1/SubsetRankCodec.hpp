#ifndef QPET_CORRELATION_SUBSET_RANK_CODEC_HPP
#define QPET_CORRELATION_SUBSET_RANK_CODEC_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gmpxx.h>

namespace qpet_correlation {
namespace subset_rank {

// This cap bounds both allocation and the number of GMP big-integer operations
// accepted from an untrusted stream.  The codec does not rely on machine-word
// binomial coefficients, so increasing the cap is a deliberate format-policy
// decision rather than an arithmetic requirement.
constexpr std::size_t kMaximumTime = 512;
constexpr std::size_t kMaximumDecodedPositions = 1U << 20U;
// A partial subset costs substantially more to unrank than an all/none
// channel.  Sum min(k,T-k) before doing the expensive GMP work so a stream
// cannot multiply central-binomial work across an unbounded channel count.
constexpr std::size_t kMaximumCombinadicWork = 4096;
constexpr std::uint8_t kPositionModeMask = 0x07U;
constexpr std::uint8_t kPositionModeNone = 0U;
constexpr std::uint8_t kPositionModeAll = 1U;
constexpr std::uint8_t kPositionModeSubsetRank = 5U;

class CodecError : public std::runtime_error {
 public:
  explicit CodecError(const std::string& message) : std::runtime_error(message) {}
};

inline void validate_time(std::size_t time)
{
  if (time < 3 || time > kMaximumTime)
    throw std::invalid_argument(
        "subset-rank time must be in [3, " +
        std::to_string(kMaximumTime) + "]");
}

inline mpz_class checked_binomial(std::size_t n, std::size_t k)
{
  if (n > kMaximumTime)
    throw std::invalid_argument("subset-rank binomial n exceeds the safety cap");
  if (k > n)
    throw std::invalid_argument("subset-rank binomial requires k <= n");
  k = std::min(k, n - k);
  mpz_class value = 1;
  for (std::size_t index = 1; index <= k; ++index) {
    value *= static_cast<unsigned long>(n - k + index);
    value /= static_cast<unsigned long>(index);
  }
  return value;
}

inline mpz_class binomial_or_zero(std::size_t n, std::size_t k)
{
  return k > n ? mpz_class(0) : checked_binomial(n, k);
}

inline std::size_t bit_width(const mpz_class& maximum_value)
{
  if (maximum_value < 0)
    throw std::invalid_argument("subset-rank bit width requires a nonnegative value");
  if (maximum_value == 0)
    return 0;
  return static_cast<std::size_t>(
      mpz_sizeinbase(maximum_value.get_mpz_t(), 2));
}

inline std::size_t count_bit_width(std::size_t time)
{
  validate_time(time);
  // A partial subset has k in [1, time - 1], hence k - 1 in [0, time - 2].
  return bit_width(mpz_class(static_cast<unsigned long>(time - 2U)));
}

inline void validate_positions(
    std::size_t time,
    const std::vector<std::size_t>& positions)
{
  if (time > kMaximumTime)
    throw std::invalid_argument("subset-rank positions exceed the safety cap");
  for (std::size_t index = 0; index < positions.size(); ++index) {
    if (positions[index] >= time)
      throw std::invalid_argument("subset-rank position is out of range");
    if (index != 0 && positions[index - 1] >= positions[index])
      throw std::invalid_argument(
          "subset-rank positions must be strictly increasing");
  }
}

inline mpz_class rank_positions(
    std::size_t time,
    const std::vector<std::size_t>& positions)
{
  if (time > kMaximumTime)
    throw std::invalid_argument("subset-rank time exceeds the safety cap");
  validate_positions(time, positions);
  mpz_class rank = 0;
  for (std::size_t index = 0; index < positions.size(); ++index)
    rank += binomial_or_zero(positions[index], index + 1U);
  const auto count = checked_binomial(time, positions.size());
  if (rank < 0 || rank >= count)
    throw std::logic_error("subset-rank encoder produced an invalid rank");
  return rank;
}

inline std::vector<std::size_t> unrank_positions(
    std::size_t time,
    std::size_t subset_size,
    const mpz_class& encoded_rank)
{
  if (time > kMaximumTime)
    throw std::invalid_argument("subset-rank time exceeds the safety cap");
  if (subset_size > time)
    throw std::invalid_argument("subset-rank subset size exceeds time");
  const auto count = checked_binomial(time, subset_size);
  if (encoded_rank < 0 || encoded_rank >= count)
    throw CodecError("subset-rank value is outside the binomial range");

  auto rank = encoded_rank;
  auto positions = std::vector<std::size_t>(subset_size, 0);
  auto upper = time;
  for (std::size_t remaining = subset_size; remaining != 0; --remaining) {
    // Find the largest x in [remaining - 1, upper - 1] for which C(x,
    // remaining) <= rank.  C(remaining - 1, remaining) is mathematically zero.
    auto low = remaining - 1U;
    auto high = upper - 1U;
    while (low < high) {
      const auto middle = low + (high - low + 1U) / 2U;
      if (binomial_or_zero(middle, remaining) <= rank)
        low = middle;
      else
        high = middle - 1U;
    }
    positions[remaining - 1U] = low;
    rank -= binomial_or_zero(low, remaining);
    upper = low;
  }
  if (rank != 0)
    throw std::logic_error("subset-rank decoder did not consume the rank");
  validate_positions(time, positions);
  return positions;
}

class MsbBitWriter {
 public:
  void write(const mpz_class& value, std::size_t width)
  {
    if (value < 0 || bit_width(value) > width)
      throw std::invalid_argument("subset-rank bit field does not fit its width");
    if (width > std::numeric_limits<std::size_t>::max() - bit_count_)
      throw std::overflow_error("subset-rank bitstream size overflows");
    for (std::size_t remaining = width; remaining != 0; --remaining) {
      const auto bit_index = remaining - 1U;
      if (bit_index > std::numeric_limits<mp_bitcnt_t>::max())
        throw std::overflow_error("subset-rank bit index overflows GMP");
      const auto bit = static_cast<unsigned>(mpz_tstbit(
          value.get_mpz_t(), static_cast<mp_bitcnt_t>(bit_index)));
      if ((bit_count_ & 7U) == 0)
        bytes_.push_back(0U);
      if (bit != 0)
        bytes_.back() |= static_cast<std::uint8_t>(
            1U << (7U - static_cast<unsigned>(bit_count_ & 7U)));
      ++bit_count_;
    }
  }

  std::size_t bit_count() const noexcept { return bit_count_; }
  const std::vector<std::uint8_t>& bytes() const noexcept { return bytes_; }

 private:
  std::vector<std::uint8_t> bytes_;
  std::size_t bit_count_ = 0;
};

class MsbBitReader {
 public:
  explicit MsbBitReader(
      const std::vector<std::uint8_t>& bytes,
      std::size_t byte_offset = 0)
      : bytes_(bytes), start_byte_offset_(byte_offset)
  {
    if (byte_offset > bytes.size())
      throw CodecError("subset-rank bitstream offset is out of range");
    if (bytes.size() > std::numeric_limits<std::size_t>::max() / 8U)
      throw CodecError("subset-rank bitstream size overflows");
    bit_position_ = byte_offset * 8U;
  }

  mpz_class read(std::size_t width)
  {
    if (width > available_bits())
      throw CodecError("subset-rank bitstream is truncated");
    mpz_class value = 0;
    for (std::size_t index = 0; index < width; ++index) {
      value <<= 1;
      const auto byte = bytes_[bit_position_ / 8U];
      value += (byte >> (7U - static_cast<unsigned>(bit_position_ & 7U))) & 1U;
      ++bit_position_;
    }
    return value;
  }

  std::size_t available_bits() const noexcept
  {
    return bytes_.size() * 8U - bit_position_;
  }

  std::size_t require_zero_padding_to_byte_boundary() const
  {
    const auto padding = (8U - (bit_position_ & 7U)) & 7U;
    if (padding > available_bits())
      throw CodecError("subset-rank bitstream padding is truncated");
    for (std::size_t index = 0; index < padding; ++index) {
      const auto position = bit_position_ + index;
      const auto byte = bytes_[position / 8U];
      if (((byte >> (7U - static_cast<unsigned>(position & 7U))) & 1U) != 0)
        throw CodecError("subset-rank bitstream has nonzero padding");
    }
    const auto end_byte = (bit_position_ + padding) / 8U;
    if (end_byte < start_byte_offset_)
      throw CodecError("subset-rank consumed-byte accounting underflows");
    return end_byte - start_byte_offset_;
  }

 private:
  const std::vector<std::uint8_t>& bytes_;
  std::size_t start_byte_offset_ = 0;
  std::size_t bit_position_ = 0;
};

inline std::vector<std::uint8_t> encode_interleaved_position_stream(
    std::size_t time,
    const std::vector<std::uint8_t>& descriptors,
    const std::vector<std::vector<std::size_t>>& positions)
{
  validate_time(time);
  if (descriptors.size() != positions.size())
    throw std::invalid_argument(
        "subset-rank descriptor and position channel counts differ");

  MsbBitWriter stream;
  const auto count_width = count_bit_width(time);
  auto encoded_positions = std::size_t{0};
  auto combinadic_work = std::size_t{0};
  auto account_positions = [&](std::size_t count) {
    if (count > kMaximumDecodedPositions - encoded_positions)
      throw std::invalid_argument(
          "subset-rank encoded position count exceeds its safety cap");
    encoded_positions += count;
  };
  for (std::size_t channel = 0; channel < descriptors.size(); ++channel) {
    const auto mode = descriptors[channel] & kPositionModeMask;
    const auto& selected = positions[channel];
    validate_positions(time, selected);
    if (mode == kPositionModeNone) {
      if (!selected.empty())
        throw std::invalid_argument("subset-rank none channel is not empty");
      continue;
    }
    if (mode == kPositionModeAll) {
      if (selected.size() != time)
        throw std::invalid_argument("subset-rank all channel is incomplete");
      for (std::size_t index = 0; index < time; ++index) {
        if (selected[index] != index)
          throw std::invalid_argument("subset-rank all channel is not canonical");
      }
      account_positions(time);
      continue;
    }
    if (mode != kPositionModeSubsetRank)
      throw std::invalid_argument("subset-rank descriptor has an unknown mode");
    if (selected.empty() || selected.size() == time)
      throw std::invalid_argument("subset-rank partial channel is not partial");
    account_positions(selected.size());
    const auto channel_work = std::min(selected.size(), time - selected.size());
    if (channel_work > kMaximumCombinadicWork - combinadic_work)
      throw std::invalid_argument(
          "subset-rank combinadic work exceeds its safety cap");
    combinadic_work += channel_work;
    stream.write(
        mpz_class(static_cast<unsigned long>(selected.size() - 1U)),
        count_width);
    const auto combinations = checked_binomial(time, selected.size());
    stream.write(rank_positions(time, selected), bit_width(combinations - 1));
  }
  return stream.bytes();
}

struct DecodedPositionPrefix {
  std::vector<std::vector<std::size_t>> positions;
  std::size_t bytes_consumed = 0;
};

inline DecodedPositionPrefix decode_interleaved_position_prefix(
    std::size_t time,
    const std::vector<std::uint8_t>& descriptors,
    const std::vector<std::uint8_t>& raw,
    std::size_t offset,
    std::size_t expected_positions)
{
  validate_time(time);
  if (expected_positions > kMaximumDecodedPositions)
    throw CodecError("subset-rank expected position count exceeds its safety cap");
  auto result = DecodedPositionPrefix{};
  result.positions.resize(descriptors.size());
  MsbBitReader stream(raw, offset);
  const auto count_width = count_bit_width(time);
  auto decoded_positions = std::size_t{0};
  auto combinadic_work = std::size_t{0};
  auto account_positions = [&](std::size_t count) {
    if (count > expected_positions - decoded_positions)
      throw CodecError(
          "subset-rank position count exceeds correction metadata");
    decoded_positions += count;
  };
  for (std::size_t channel = 0; channel < descriptors.size(); ++channel) {
    const auto mode = descriptors[channel] & kPositionModeMask;
    if (mode == kPositionModeNone)
      continue;
    if (mode == kPositionModeAll) {
      account_positions(time);
      result.positions[channel].resize(time);
      for (std::size_t index = 0; index < time; ++index)
        result.positions[channel][index] = index;
      continue;
    }
    if (mode != kPositionModeSubsetRank)
      throw CodecError("subset-rank descriptor has an unknown mode");
    const auto encoded_count = stream.read(count_width);
    if (encoded_count >
        mpz_class(static_cast<unsigned long>(time - 2U)))
      throw CodecError("subset-rank partial count is invalid");
    if (!encoded_count.fits_ulong_p())
      throw CodecError("subset-rank partial count does not fit an unsigned long");
    const auto subset_size =
        static_cast<std::size_t>(encoded_count.get_ui()) + 1U;
    account_positions(subset_size);
    const auto channel_work = std::min(subset_size, time - subset_size);
    if (channel_work > kMaximumCombinadicWork - combinadic_work)
      throw CodecError("subset-rank combinadic work exceeds its safety cap");
    combinadic_work += channel_work;
    const auto combinations = checked_binomial(time, subset_size);
    const auto rank = stream.read(bit_width(combinations - 1));
    if (rank >= combinations)
      throw CodecError("subset-rank value is outside the binomial range");
    result.positions[channel] = unrank_positions(time, subset_size, rank);
  }
  if (decoded_positions != expected_positions)
    throw CodecError(
        "subset-rank position count disagrees with correction metadata");
  result.bytes_consumed = stream.require_zero_padding_to_byte_boundary();
  return result;
}

}  // namespace subset_rank
}  // namespace qpet_correlation

#endif
