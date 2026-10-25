#ifndef QPET_CORRELATION_FEASIBLE_VALUE_TARGETS_HPP
#define QPET_CORRELATION_FEASIBLE_VALUE_TARGETS_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qpet_correlation::feasible_value {

// Correction formats 4 and 6 reserve int8 code zero as an exact-value escape.
// This out-of-range marker is diagnostics only; it is never serialized.
constexpr std::int16_t kExactEscapeCode = 256;
constexpr int kMaximumCodeMagnitude = 127;

enum class CandidateKind : std::uint8_t {
  minimum_pointwise = 0,
  packed_bit_width = 1,
  homotopy_fraction = 2,
};

struct HomotopyLevel {
  std::uint16_t numerator = 0;
  std::uint16_t denominator = 1;
};

struct TargetCandidate {
  CandidateKind kind = CandidateKind::minimum_pointwise;
  // Nonzero only for packed_bit_width candidates.
  std::uint8_t proposed_bit_width = 0;
  // Used only for homotopy_fraction candidates.
  HomotopyLevel level{};
  std::vector<float> values;
  // Per-value diagnostics: zero means keep the base value, a nonzero int8
  // value means use the channel scale, and kExactEscapeCode means an exact
  // float32 escape. The production serializer remains the source of truth.
  std::vector<std::int16_t> codes;
  std::size_t changed_values = 0;
  std::size_t quantized_values = 0;
  std::size_t exact_escapes = 0;
  std::uint16_t maximum_zigzag = 0;
};

struct TargetFamily {
  std::vector<TargetCandidate> candidates;
  std::size_t mandatory_values = 0;
  std::size_t mandatory_exact_escapes = 0;
};

namespace detail {

inline std::uint32_t float32_bits(float value)
{
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

inline bool bitwise_equal(
    const std::vector<float>& left, const std::vector<float>& right)
{
  return left.size() == right.size() &&
      (left.empty() ||
       std::memcmp(left.data(), right.data(), left.size() * sizeof(float)) == 0);
}

inline std::uint16_t zigzag_code(int code)
{
  if (code < -kMaximumCodeMagnitude || code > kMaximumCodeMagnitude)
    throw std::out_of_range("feasible-value code exceeds the supported grid");
  const auto encoded = code >= 0 ? 2 * code : 2 * (-code) - 1;
  return static_cast<std::uint16_t>(encoded);
}

// Keep this expression identical to correction formats 4 and 6. Final
// production acceptance must nevertheless replay the actual decoder bitwise,
// because compiler floating-point contraction rules are an external concern.
inline float decode_grid_value(float base, int code, float scale)
{
  return static_cast<float>(base + static_cast<float>(code) * scale);
}

inline double pointwise_error(float original, float reconstructed)
{
  return std::abs(
      static_cast<double>(original) - static_cast<double>(reconstructed));
}

inline bool code_fits_width(int code, std::uint8_t bit_width)
{
  if (code == 0)
    return true;
  if (bit_width < 1 || bit_width > 8)
    throw std::invalid_argument("packed correction bit width must be in [1,8]");
  const auto maximum = static_cast<std::uint16_t>((1U << bit_width) - 1U);
  return zigzag_code(code) <= maximum;
}

struct Endpoint {
  bool base_feasible = false;
  bool forced_exact = false;
  int minimum_code = 0;
  int nearest_code = 0;
};

inline bool better_minimum_code(
    int candidate,
    double candidate_error,
    int incumbent,
    double incumbent_error)
{
  const auto candidate_key = std::pair<int, double>{
      std::abs(candidate), candidate_error};
  const auto incumbent_key = std::pair<int, double>{
      std::abs(incumbent), incumbent_error};
  if (candidate_key != incumbent_key)
    return candidate_key < incumbent_key;
  const auto candidate_zigzag = zigzag_code(candidate);
  const auto incumbent_zigzag = zigzag_code(incumbent);
  return candidate_zigzag != incumbent_zigzag
      ? candidate_zigzag < incumbent_zigzag
      : candidate < incumbent;
}

inline bool better_nearest_code(
    int candidate,
    double candidate_error,
    int incumbent,
    double incumbent_error)
{
  const auto candidate_key = std::pair<double, int>{
      candidate_error, std::abs(candidate)};
  const auto incumbent_key = std::pair<double, int>{
      incumbent_error, std::abs(incumbent)};
  if (candidate_key != incumbent_key)
    return candidate_key < incumbent_key;
  const auto candidate_zigzag = zigzag_code(candidate);
  const auto incumbent_zigzag = zigzag_code(incumbent);
  return candidate_zigzag != incumbent_zigzag
      ? candidate_zigzag < incumbent_zigzag
      : candidate < incumbent;
}

inline Endpoint find_endpoint(
    float original,
    float base,
    float scale,
    double pointwise_tolerance)
{
  Endpoint result;
  const auto base_error = pointwise_error(original, base);
  result.base_feasible = base_error <= pointwise_tolerance;

  bool has_grid_code = false;
  auto minimum_code = 0;
  auto nearest_code = 0;
  auto minimum_error = base_error;
  auto nearest_error = base_error;
  if (!result.base_feasible) {
    minimum_error = std::numeric_limits<double>::infinity();
    nearest_error = std::numeric_limits<double>::infinity();
  }

  if (scale > 0.0F && std::isfinite(scale)) {
    for (auto code = -kMaximumCodeMagnitude;
         code <= kMaximumCodeMagnitude;
         ++code) {
      if (code == 0)
        continue;
      const auto value = decode_grid_value(base, code, scale);
      if (!std::isfinite(value))
        continue;
      const auto error = pointwise_error(original, value);
      if (error > pointwise_tolerance)
        continue;
      if (!has_grid_code ||
          better_minimum_code(code, error, minimum_code, minimum_error)) {
        minimum_code = code;
        minimum_error = error;
      }
      if (!has_grid_code ||
          better_nearest_code(code, error, nearest_code, nearest_error)) {
        nearest_code = code;
        nearest_error = error;
      }
      has_grid_code = true;
    }
  }

  if (result.base_feasible) {
    if (has_grid_code &&
        better_nearest_code(nearest_code, nearest_error, 0, base_error))
      result.nearest_code = nearest_code;
    // The minimum-magnitude pointwise-feasible action is no action.
    result.minimum_code = 0;
    return result;
  }
  if (!has_grid_code) {
    result.forced_exact = true;
    return result;
  }
  result.minimum_code = minimum_code;
  result.nearest_code = nearest_code;
  if ((result.minimum_code < 0) != (result.nearest_code < 0) ||
      std::abs(result.minimum_code) > std::abs(result.nearest_code))
    throw std::logic_error("feasible-value endpoints do not define a code homotopy");
  return result;
}

inline void record_code(TargetCandidate& candidate, std::int16_t code)
{
  candidate.codes.push_back(code);
  if (code == 0)
    return;
  ++candidate.changed_values;
  if (code == kExactEscapeCode) {
    ++candidate.exact_escapes;
    return;
  }
  ++candidate.quantized_values;
  candidate.maximum_zigzag = std::max(
      candidate.maximum_zigzag, zigzag_code(static_cast<int>(code)));
}

inline int interpolate_code(
    const Endpoint& endpoint, HomotopyLevel level)
{
  if (level.denominator == 0 || level.numerator > level.denominator)
    throw std::invalid_argument("homotopy level must lie in [0,1]");
  if (endpoint.forced_exact || endpoint.nearest_code == 0)
    return 0;
  const auto sign = endpoint.nearest_code < 0 ? -1 : 1;
  const auto minimum = std::abs(endpoint.minimum_code);
  const auto nearest = std::abs(endpoint.nearest_code);
  const auto delta = nearest - minimum;
  const auto offset =
      static_cast<std::uint64_t>(delta) * level.numerator / level.denominator;
  return sign * (minimum + static_cast<int>(offset));
}

inline int closest_width_limited_code(
    const Endpoint& endpoint, std::uint8_t bit_width)
{
  if (endpoint.forced_exact || endpoint.nearest_code == 0)
    return 0;
  const auto sign = endpoint.nearest_code < 0 ? -1 : 1;
  const auto minimum = std::abs(endpoint.minimum_code);
  const auto nearest = std::abs(endpoint.nearest_code);
  for (auto magnitude = nearest; magnitude >= minimum; --magnitude) {
    const auto code = sign * magnitude;
    if (code_fits_width(code, bit_width))
      return code;
  }
  return 0;
}

inline void append_unique(
    std::vector<TargetCandidate>& candidates, TargetCandidate candidate)
{
  const auto duplicate = std::any_of(
      candidates.begin(), candidates.end(), [&](const TargetCandidate& retained) {
        return bitwise_equal(retained.values, candidate.values);
      });
  if (!duplicate)
    candidates.push_back(std::move(candidate));
}

}  // namespace detail

class TargetBuilder {
 public:
  TargetBuilder(
      const std::vector<float>& original,
      const std::vector<float>& base,
      const std::vector<float>& channel_scales,
      std::size_t channels,
      std::size_t time,
      double pointwise_tolerance)
      : original_(original),
        base_(base),
        channel_scales_(channel_scales),
        channels_(channels),
        time_(time),
        pointwise_tolerance_(pointwise_tolerance)
  {
    if (channels_ == 0 || time_ == 0 ||
        channels_ > std::numeric_limits<std::size_t>::max() / time_ ||
        original_.size() != channels_ * time_ || base_.size() != original_.size() ||
        channel_scales_.size() != channels_)
      throw std::invalid_argument("feasible-value target dimensions are invalid");
    if (!(pointwise_tolerance_ > 0.0) || !std::isfinite(pointwise_tolerance_))
      throw std::invalid_argument("pointwise tolerance must be finite and positive");
    endpoints_.reserve(original_.size());
    for (std::size_t index = 0; index < original_.size(); ++index) {
      if (!std::isfinite(original_[index]) || !std::isfinite(base_[index]))
        throw std::invalid_argument("feasible-value inputs must be finite float32");
      const auto scale = channel_scales_[index / time_];
      if (!(scale >= 0.0F) || !std::isfinite(scale))
        throw std::invalid_argument("feasible-value channel scales are invalid");
      endpoints_.push_back(detail::find_endpoint(
          original_[index], base_[index], scale, pointwise_tolerance_));
    }
  }

  TargetCandidate minimum_candidate() const
  {
    TargetCandidate candidate;
    candidate.kind = CandidateKind::minimum_pointwise;
    candidate.values.reserve(original_.size());
    candidate.codes.reserve(original_.size());
    for (std::size_t index = 0; index < original_.size(); ++index) {
      const auto& endpoint = endpoints_[index];
      if (endpoint.forced_exact) {
        candidate.values.push_back(original_[index]);
        detail::record_code(candidate, kExactEscapeCode);
        continue;
      }
      const auto code = endpoint.minimum_code;
      const auto value = code == 0
          ? base_[index]
          : detail::decode_grid_value(
                base_[index], code, channel_scales_[index / time_]);
      require_pointwise(index, value);
      candidate.values.push_back(value);
      detail::record_code(candidate, static_cast<std::int16_t>(code));
    }
    return candidate;
  }

  TargetCandidate packed_bit_width_candidate(std::uint8_t bit_width) const
  {
    if (bit_width < 1 || bit_width > 8)
      throw std::invalid_argument("packed correction bit width must be in [1,8]");
    TargetCandidate candidate;
    candidate.kind = CandidateKind::packed_bit_width;
    candidate.proposed_bit_width = bit_width;
    candidate.values.reserve(original_.size());
    candidate.codes.reserve(original_.size());
    for (std::size_t index = 0; index < original_.size(); ++index) {
      const auto& endpoint = endpoints_[index];
      auto code = detail::closest_width_limited_code(endpoint, bit_width);
      if (endpoint.forced_exact || (!endpoint.base_feasible && code == 0)) {
        candidate.values.push_back(original_[index]);
        detail::record_code(candidate, kExactEscapeCode);
        continue;
      }
      const auto value = code == 0
          ? base_[index]
          : detail::decode_grid_value(
                base_[index], code, channel_scales_[index / time_]);
      require_pointwise(index, value);
      candidate.values.push_back(value);
      detail::record_code(candidate, static_cast<std::int16_t>(code));
    }
    return candidate;
  }

  TargetCandidate homotopy_candidate(HomotopyLevel level) const
  {
    if (level.denominator == 0 || level.numerator > level.denominator)
      throw std::invalid_argument("homotopy level must lie in [0,1]");
    TargetCandidate candidate;
    candidate.kind = CandidateKind::homotopy_fraction;
    candidate.level = level;
    candidate.values.reserve(original_.size());
    candidate.codes.reserve(original_.size());
    for (std::size_t index = 0; index < original_.size(); ++index) {
      const auto& endpoint = endpoints_[index];
      if (endpoint.forced_exact) {
        candidate.values.push_back(original_[index]);
        detail::record_code(candidate, kExactEscapeCode);
        continue;
      }
      auto code = detail::interpolate_code(endpoint, level);
      auto value = code == 0
          ? base_[index]
          : detail::decode_grid_value(
                base_[index], code, channel_scales_[index / time_]);
      // Float32 rounding can make an otherwise interval-monotone integer path
      // fail exactly at a boundary. Fall back to the already proven minimum
      // endpoint; never expand the admissible error by a hidden epsilon.
      if (detail::pointwise_error(original_[index], value) > pointwise_tolerance_) {
        code = endpoint.minimum_code;
        value = code == 0
            ? base_[index]
            : detail::decode_grid_value(
                  base_[index], code, channel_scales_[index / time_]);
      }
      require_pointwise(index, value);
      candidate.values.push_back(value);
      detail::record_code(candidate, static_cast<std::int16_t>(code));
    }
    return candidate;
  }

  TargetFamily minimum_and_bit_width_family() const
  {
    TargetFamily family;
    for (const auto& endpoint : endpoints_) {
      family.mandatory_values += !endpoint.base_feasible;
      family.mandatory_exact_escapes += endpoint.forced_exact;
    }
    detail::append_unique(family.candidates, minimum_candidate());
    // Include all actual format-6 alphabets. Code zero is an escape, which
    // makes the one-bit alphabet intentionally asymmetric {-1, escape}.
    for (std::uint8_t bit_width = 1; bit_width <= 8; ++bit_width)
      detail::append_unique(
          family.candidates, packed_bit_width_candidate(bit_width));
    return family;
  }

  TargetFamily bounded_default_family() const
  {
    auto family = minimum_and_bit_width_family();
    // These rational levels avoid platform-sensitive decimal-to-binary alpha
    // parsing. Endpoints alpha=0 and alpha=1 are already represented by the
    // minimum candidate and the largest useful packed alphabet respectively.
    constexpr std::array<HomotopyLevel, 3> levels = {{
        {1, 4}, {1, 2}, {3, 4}}};
    for (const auto level : levels)
      detail::append_unique(family.candidates, homotopy_candidate(level));
    // One minimum proposal, eight format-6 alphabet proposals, and three
    // fractional proposals. Bitwise duplicates are removed, never added.
    if (family.candidates.size() > 12)
      throw std::logic_error("bounded feasible-value family exceeded its budget");
    return family;
  }

 private:
  void require_pointwise(std::size_t index, float value) const
  {
    if (!std::isfinite(value) ||
        detail::pointwise_error(original_[index], value) > pointwise_tolerance_)
      throw std::logic_error("generated feasible-value target violates L-infinity");
  }

  const std::vector<float>& original_;
  const std::vector<float>& base_;
  const std::vector<float>& channel_scales_;
  std::size_t channels_ = 0;
  std::size_t time_ = 0;
  double pointwise_tolerance_ = 0.0;
  std::vector<detail::Endpoint> endpoints_;
};

}  // namespace qpet_correlation::feasible_value

#endif
