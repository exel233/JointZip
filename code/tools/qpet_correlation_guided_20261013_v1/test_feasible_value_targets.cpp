#include "FeasibleValueTargets.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace fv = qpet_correlation::feasible_value;

namespace {

void require(bool condition, const char* message)
{
  if (!condition)
    throw std::runtime_error(message);
}

std::uint32_t bits(float value)
{
  std::uint32_t result = 0;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

void require_pointwise(
    const std::vector<float>& original,
    const std::vector<float>& reconstructed,
    double tolerance)
{
  require(
      original.size() == reconstructed.size(),
      "pointwise test vectors have different sizes");
  for (std::size_t index = 0; index < original.size(); ++index) {
    require(
        std::abs(
            static_cast<double>(original[index]) - reconstructed[index]) <=
            tolerance,
        "generated target failed the explicit pointwise check");
  }
}

}  // namespace

int main()
{
  // Exact binary fractions make the expected integer endpoints independent of
  // decimal parsing and boundary round-off: residual=1, scale=1/8, epsilon=3/4.
  const std::vector<float> original = {1.0F, 2.0F, 3.0F, 4.0F};
  const std::vector<float> positive_base = {0.0F, 1.0F, 2.0F, 3.0F};
  const std::vector<float> scales = {0.125F};
  const fv::TargetBuilder positive(
      original, positive_base, scales, 1, original.size(), 0.75);

  const auto minimum = positive.minimum_candidate();
  require(minimum.changed_values == original.size(), "minimum change count mismatch");
  require(
      minimum.quantized_values == original.size(),
      "minimum quantized count mismatch");
  require(minimum.exact_escapes == 0, "minimum unexpectedly used exact escapes");
  require(minimum.maximum_zigzag == 4, "minimum zigzag width mismatch");  // q=+2
  for (const auto code : minimum.codes)
    require(code == 2, "minimum positive endpoint is not q=+2");
  require_pointwise(original, minimum.values, 0.75);

  // Packed zigzag is asymmetric. Two bits cannot represent +2, so mandatory
  // positive repairs become exact escapes. Three bits can represent +3.
  const auto positive_b2 = positive.packed_bit_width_candidate(2);
  require(
      positive_b2.exact_escapes == original.size(),
      "two-bit positive fixture did not use exact escapes");
  require(positive_b2.quantized_values == 0, "two-bit escape fixture used a grid code");
  const auto positive_b3 = positive.packed_bit_width_candidate(3);
  require(positive_b3.exact_escapes == 0, "three-bit fixture used exact escapes");
  require(positive_b3.maximum_zigzag <= 7, "three-bit fixture exceeded its alphabet");
  for (const auto code : positive_b3.codes)
    require(code == 3, "three-bit positive endpoint is not q=+3");

  const std::vector<float> negative_base = {2.0F, 3.0F, 4.0F, 5.0F};
  const fv::TargetBuilder negative(
      original, negative_base, scales, 1, original.size(), 0.75);
  const auto negative_b2 = negative.packed_bit_width_candidate(2);
  require(negative_b2.exact_escapes == 0, "two-bit negative fixture escaped");
  require(negative_b2.maximum_zigzag == 3, "negative zigzag asymmetry mismatch");
  for (const auto code : negative_b2.codes)
    require(code == -2, "two-bit negative endpoint is not q=-2");

  // A point can be outside epsilon while no scale-grid value is feasible. The
  // family must use an exact escape rather than relax the hard bound.
  const std::vector<float> off_grid_original = {0.5F};
  const std::vector<float> off_grid_base = {0.0F};
  const std::vector<float> coarse_scale = {1.0F};
  const fv::TargetBuilder off_grid(
      off_grid_original, off_grid_base, coarse_scale, 1, 1, 0.125);
  const auto off_grid_minimum = off_grid.minimum_candidate();
  require(off_grid_minimum.exact_escapes == 1, "off-grid value did not escape");
  require(
      off_grid_minimum.codes.front() == fv::kExactEscapeCode,
      "off-grid value has the wrong code marker");
  require(
      bits(off_grid_minimum.values.front()) == bits(off_grid_original.front()),
      "off-grid escape did not preserve original float32 bits");

  // When the base is already feasible, alpha=0 is a true no-op and alpha=1
  // moves toward the nearest grid reconstruction without leaving the interval.
  const std::vector<float> optional_original = {1.0F};
  const std::vector<float> optional_base = {0.5F};
  const fv::TargetBuilder optional(
      optional_original, optional_base, scales, 1, 1, 0.5);
  const auto alpha0 = optional.homotopy_candidate({0, 1});
  const auto alpha1 = optional.homotopy_candidate({1, 1});
  require(alpha0.codes.front() == 0, "alpha zero was not a no-op");
  require(
      bits(alpha0.values.front()) == bits(optional_base.front()),
      "alpha zero changed base float32 bits");
  require(alpha1.codes.front() == 4, "alpha one did not reach q=+4");
  require(
      bits(alpha1.values.front()) == bits(optional_original.front()),
      "alpha one did not reach the nearest endpoint");

  const auto alpha_half = positive.homotopy_candidate({1, 2});
  for (const auto code : alpha_half.codes)
    require(code == 5, "half homotopy did not use exact rational interpolation");

  const auto family = positive.minimum_and_bit_width_family();
  require(
      family.mandatory_values == original.size(),
      "family mandatory-value count mismatch");
  require(family.mandatory_exact_escapes == 0, "family forced an avoidable escape");
  require(
      !family.candidates.empty() && family.candidates.size() <= 9,
      "minimum/bit-width family exceeded its fixed budget");
  const auto bounded_family = positive.bounded_default_family();
  require(
      bounded_family.candidates.size() <= 12,
      "default family exceeded its fixed budget");
  bool saw_fraction = false;
  for (const auto& candidate : bounded_family.candidates)
    saw_fraction = saw_fraction || candidate.kind == fv::CandidateKind::homotopy_fraction;
  require(saw_fraction, "default family lost all fixed homotopy proposals");
  // Candidate construction and bitwise de-duplication are deterministic.
  const auto repeated = positive.minimum_and_bit_width_family();
  require(
      repeated.candidates.size() == family.candidates.size(),
      "repeated family size is nondeterministic");
  for (std::size_t index = 0; index < family.candidates.size(); ++index) {
    require(
        family.candidates[index].codes == repeated.candidates[index].codes,
        "repeated family codes are nondeterministic");
    require(
        family.candidates[index].values.size() ==
            repeated.candidates[index].values.size(),
        "repeated family value sizes differ");
    for (std::size_t value = 0;
         value < family.candidates[index].values.size();
         ++value) {
      require(
          bits(family.candidates[index].values[value]) ==
              bits(repeated.candidates[index].values[value]),
          "repeated family values are not bitwise deterministic");
    }
  }

  bool invalid_level_rejected = false;
  try {
    (void)positive.homotopy_candidate({2, 1});
  }
  catch (const std::invalid_argument&) {
    invalid_level_rejected = true;
  }
  require(invalid_level_rejected, "invalid homotopy level was accepted");

  // Signed zero is numerically feasible and alpha=0 preserves the base bits;
  // the production serializer/replay remains responsible for its own equality
  // convention.
  const std::vector<float> positive_zero = {0.0F};
  const std::vector<float> negative_zero = {-0.0F};
  const std::vector<float> zero_scale = {0.0F};
  const fv::TargetBuilder signed_zero(
      positive_zero, negative_zero, zero_scale, 1, 1, 1.0e-6);
  const auto signed_zero_minimum = signed_zero.minimum_candidate();
  require(
      bits(signed_zero_minimum.values.front()) == bits(negative_zero.front()),
      "signed-zero no-op changed base bits");

  // A zero scale is legal for an unchanged channel, but a mandatory changed
  // point must use an exact escape.
  const std::vector<float> zero_scale_original = {1.0F};
  const std::vector<float> zero_scale_base = {0.0F};
  const fv::TargetBuilder zero_scale_changed(
      zero_scale_original, zero_scale_base, zero_scale, 1, 1, 0.5);
  require(
      zero_scale_changed.minimum_candidate().exact_escapes == 1,
      "zero-scale mandatory repair did not escape");

  // Exercise the largest positive int8 magnitude supported by the existing
  // legacy denominator portfolio and format-6 zigzag packing.
  const std::vector<float> extreme_original = {127.0F, -127.0F};
  const std::vector<float> extreme_base = {0.0F, 0.0F};
  const std::vector<float> unit_scales = {1.0F};
  const fv::TargetBuilder extreme(
      extreme_original, extreme_base, unit_scales, 1, 2, 0.25);
  const auto extreme_b8 = extreme.packed_bit_width_candidate(8);
  require(extreme_b8.codes[0] == 127, "positive int8 endpoint mismatch");
  require(extreme_b8.codes[1] == -127, "negative int8 endpoint mismatch");
  require(extreme_b8.maximum_zigzag == 254, "eight-bit zigzag maximum mismatch");

  // At a nextafter boundary, use the exact double difference as epsilon and
  // ensure the strict (non-expanded) comparison retains the base.
  const auto adjacent = std::nextafter(1.0F, 0.0F);
  const auto adjacent_error =
      static_cast<double>(1.0F) - static_cast<double>(adjacent);
  const std::vector<float> adjacent_original = {1.0F};
  const std::vector<float> adjacent_base = {adjacent};
  const fv::TargetBuilder adjacent_builder(
      adjacent_original, adjacent_base, zero_scale, 1, 1, adjacent_error);
  require(
      bits(adjacent_builder.minimum_candidate().values.front()) == bits(adjacent),
      "nextafter boundary was not treated as feasible");

  bool nonfinite_rejected = false;
  try {
    const std::vector<float> invalid_original = {
        std::numeric_limits<float>::infinity()};
    (void)fv::TargetBuilder(
        invalid_original, zero_scale_base, zero_scale, 1, 1, 0.5);
  }
  catch (const std::invalid_argument&) {
    nonfinite_rejected = true;
  }
  require(nonfinite_rejected, "non-finite input was accepted");
  return 0;
}
