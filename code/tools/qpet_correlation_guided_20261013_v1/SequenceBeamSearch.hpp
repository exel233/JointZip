#ifndef QPET_CORRELATION_SEQUENCE_BEAM_SEARCH_HPP
#define QPET_CORRELATION_SEQUENCE_BEAM_SEARCH_HPP

#include "CorrelationQoI.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace qpet_correlation::ssbb {

// These are algorithmic budgets, not wall-clock cutoffs. Production search
// always visits every time position in ascending order. Tests may enlarge only
// the two beam strata to compare the bounded search with the finite oracle.
constexpr std::size_t kMaximumActionsPerPosition = 5;
constexpr std::size_t kProductionFeasibleBeamWidth = 48;
constexpr std::size_t kProductionInfeasibleBeamWidth = 16;
constexpr std::size_t kProductionTotalBeamWidth =
    kProductionFeasibleBeamWidth + kProductionInfeasibleBeamWidth;
constexpr std::size_t kProductionOutputLimit = 3;
constexpr std::size_t kExhaustiveOracleCombinationLimit = 2'000'000;

static_assert(kProductionTotalBeamWidth == 64, "SSBB production beam must be B=64");

struct Constraints {
  double pointwise_tolerance = 0.0;
  double acf_tolerance = -1.0;
  double minimum_correlation = -1.0;
  std::size_t max_lag = 0;
};

struct SearchBudget {
  std::size_t feasible_beam_width = kProductionFeasibleBeamWidth;
  std::size_t infeasible_beam_width = kProductionInfeasibleBeamWidth;
  std::size_t output_limit = kProductionOutputLimit;
};

inline SearchBudget production_budget()
{
  return {};
}

enum class ActionKind : std::uint8_t {
  keep = 0,
  base = 1,
  grid = 2,
  exact = 3,
};

struct Action {
  ActionKind kind = ActionKind::keep;
  float value = 0.0F;
  // Nonzero only for a decoder-exact grid action.
  std::int16_t code = 0;
  // Smallest packed zigzag width at which this grid value was discovered.
  std::uint8_t bit_width = 0;
  std::uint8_t ordinal = 0;
};

struct PositionActions {
  std::size_t index = 0;
  std::vector<Action> actions;
};

// This tuple is an explicitly non-byte proxy. In particular, it does not model
// the correction mask, Zstd context, format-5 constants, or competition among
// formats 0--6. Integration must serialize every returned candidate with the
// production serializer before making any rate decision.
struct RateProxy {
  std::size_t exact_escapes = 0;
  std::uint8_t maximum_bit_width = 0;
  std::size_t changed_values = 0;
  std::uint64_t zigzag_sum = 0;
  // Sufficient-statistics score used only for beam ordering/stratification.
  // It is not a hard feasibility decision for a partial search state.
  double constraint_violation = 0.0;
};

struct PairMoments {
  std::size_t count = 0;
  long double left_sum = 0.0L;
  long double right_sum = 0.0L;
  long double left_square_sum = 0.0L;
  long double right_square_sum = 0.0L;
  long double cross_sum = 0.0L;

  void append(long double left, long double right)
  {
    if (count == std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("SSBB pair-moment count overflow");
    ++count;
    left_sum += left;
    right_sum += right;
    left_square_sum += left * left;
    right_square_sum += right * right;
    cross_sum += left * right;
  }

  double correlation() const
  {
    if (count < 2)
      return 0.0;
    const auto n = static_cast<long double>(count);
    const auto covariance = cross_sum - left_sum * right_sum / n;
    const auto left_energy =
        std::max(0.0L, left_square_sum - left_sum * left_sum / n);
    const auto right_energy =
        std::max(0.0L, right_square_sum - right_sum * right_sum / n);
    const auto denominator = std::sqrt(left_energy * right_energy);
    if (!(denominator > qpet_correlation::kDenominatorFloor))
      return 0.0;
    const auto value = static_cast<double>(covariance / denominator);
    if (!std::isfinite(value))
      return 0.0;
    // Raw moments can exceed the mathematical range by a few ulps. This value
    // is a ranking proxy; the direct two-pass verifier remains authoritative.
    return std::clamp(value, -1.0, 1.0);
  }
};

// Copyable sufficient statistics for arbitrary point replacement. replace()
// updates Pearson in O(1) and all requested ACF lags in O(L), then mutates the
// supplied reconstruction. The reconstruction must match these statistics on
// entry; keeping it outside the class avoids duplicating every beam tensor.
class SufficientStatistics {
 public:
  static SufficientStatistics build(
      const std::vector<float>& original,
      const std::vector<float>& reconstructed,
      std::size_t max_lag)
  {
    if (original.empty() || original.size() != reconstructed.size())
      throw std::invalid_argument("SSBB statistic vectors have invalid dimensions");
    if (max_lag >= original.size())
      throw std::invalid_argument("SSBB statistic max_lag exceeds the series");
    SufficientStatistics result;
    result.count_ = original.size();
    result.acf_.resize(max_lag);
    for (std::size_t index = 0; index < original.size(); ++index) {
      if (!std::isfinite(original[index]) || !std::isfinite(reconstructed[index]))
        throw std::invalid_argument("SSBB statistics require finite float32 values");
      result.pearson_.append(original[index], reconstructed[index]);
    }
    for (std::size_t lag = 1; lag <= max_lag; ++lag) {
      for (std::size_t index = 0; index < original.size() - lag; ++index)
        result.acf_[lag - 1].append(
            reconstructed[index], reconstructed[index + lag]);
    }
    return result;
  }

  void replace(
      const std::vector<float>& original,
      std::vector<float>& reconstructed,
      std::size_t index,
      float new_value)
  {
    if (original.size() != count_ || reconstructed.size() != count_ ||
        index >= count_)
      throw std::invalid_argument("SSBB statistic replacement dimensions are invalid");
    if (!std::isfinite(new_value))
      throw std::invalid_argument("SSBB statistic replacement is non-finite");
    const auto old_value = reconstructed[index];
    if (bits(old_value) == bits(new_value))
      return;
    const auto old_ld = static_cast<long double>(old_value);
    const auto new_ld = static_cast<long double>(new_value);
    const auto delta = new_ld - old_ld;
    const auto square_delta = new_ld * new_ld - old_ld * old_ld;

    pearson_.right_sum += delta;
    pearson_.right_square_sum += square_delta;
    pearson_.cross_sum += static_cast<long double>(original[index]) * delta;
    for (std::size_t lag = 1; lag <= acf_.size(); ++lag) {
      auto& moments = acf_[lag - 1];
      if (index < count_ - lag) {
        moments.left_sum += delta;
        moments.left_square_sum += square_delta;
        moments.cross_sum +=
            delta * static_cast<long double>(reconstructed[index + lag]);
      }
      if (index >= lag) {
        moments.right_sum += delta;
        moments.right_square_sum += square_delta;
        moments.cross_sum +=
            static_cast<long double>(reconstructed[index - lag]) * delta;
      }
    }
    reconstructed[index] = new_value;
  }

  std::size_t count() const { return count_; }
  std::size_t max_lag() const { return acf_.size(); }
  double pearson_correlation() const { return pearson_.correlation(); }

  double acf_correlation(std::size_t lag) const
  {
    if (lag == 0 || lag > acf_.size())
      throw std::out_of_range("SSBB ACF statistic lag is out of range");
    return acf_[lag - 1].correlation();
  }

 private:
  static std::uint32_t bits(float value)
  {
    std::uint32_t result = 0;
    std::memcpy(&result, &value, sizeof(result));
    return result;
  }

  std::size_t count_ = 0;
  PairMoments pearson_;
  std::vector<PairMoments> acf_;
};

struct Diagnostics {
  std::size_t positions_processed = 0;
  std::size_t actions_generated = 0;
  std::size_t states_generated = 0;
  std::size_t states_deduplicated = 0;
  std::size_t statistics_feasible_states = 0;
  std::size_t statistics_infeasible_states = 0;
  std::size_t feasible_states_retained = 0;
  std::size_t infeasible_states_retained = 0;
  std::size_t states_pruned_by_beam = 0;
  std::size_t terminal_states = 0;
  std::size_t terminal_direct_feasible = 0;
  std::size_t terminal_direct_infeasible = 0;
  std::size_t outputs_returned = 0;
  std::size_t configured_feasible_width = 0;
  std::size_t configured_infeasible_width = 0;
  std::size_t configured_output_limit = 0;
};

struct Candidate {
  std::vector<float> reconstruction;
  std::vector<std::uint8_t> action_ordinals;
  RateProxy proxy;
  qpet_correlation::ConstraintReport report;
};

struct SearchResult {
  std::vector<Candidate> candidates;
  std::vector<PositionActions> positions;
  Diagnostics diagnostics;
};

struct ExhaustiveOracleResult {
  bool has_feasible = false;
  Candidate best;
  std::size_t combinations = 0;
  std::size_t direct_feasible = 0;
  std::size_t direct_infeasible = 0;
};

namespace detail {

inline std::size_t checked_add(
    std::size_t left, std::size_t right, const char* message)
{
  if (right > std::numeric_limits<std::size_t>::max() - left)
    throw std::overflow_error(message);
  return left + right;
}

inline std::size_t checked_multiply(
    std::size_t left, std::size_t right, const char* message)
{
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::overflow_error(message);
  return left * right;
}

inline void checked_increment(std::size_t& value, const char* message)
{
  value = checked_add(value, 1, message);
}

inline void checked_accumulate(
    std::size_t& total, std::size_t value, const char* message)
{
  total = checked_add(total, value, message);
}

inline std::uint32_t float_bits(float value)
{
  std::uint32_t result = 0;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

inline bool bitwise_equal(float left, float right)
{
  return float_bits(left) == float_bits(right);
}

inline bool bitwise_equal(
    const std::vector<float>& left, const std::vector<float>& right)
{
  if (left.size() != right.size())
    return false;
  checked_multiply(left.size(), sizeof(float), "SSBB bitwise byte count overflow");
  return left.empty() ||
      std::memcmp(left.data(), right.data(), left.size() * sizeof(float)) == 0;
}

inline bool bits_lexicographically_less(
    const std::vector<float>& left, const std::vector<float>& right)
{
  const auto common = std::min(left.size(), right.size());
  for (std::size_t index = 0; index < common; ++index) {
    const auto a = float_bits(left[index]);
    const auto b = float_bits(right[index]);
    if (a != b)
      return a < b;
  }
  return left.size() < right.size();
}

inline std::uint16_t zigzag(int code)
{
  if (code < -127 || code > 127 || code == 0)
    throw std::out_of_range("SSBB grid code is outside [-127,-1] U [1,127]");
  return static_cast<std::uint16_t>(code > 0 ? 2 * code : 2 * (-code) - 1);
}

inline std::uint8_t required_bit_width(std::uint16_t value)
{
  if (value == 0)
    return 0;
  std::uint8_t width = 1;
  while (width < 8 && value > static_cast<std::uint16_t>((1U << width) - 1U))
    ++width;
  return width;
}

inline float decode_grid_value(float base, int code, float scale)
{
  return static_cast<float>(base + static_cast<float>(code) * scale);
}

inline double pointwise_error(float original, float reconstructed)
{
  return std::abs(
      static_cast<double>(original) - static_cast<double>(reconstructed));
}

inline bool pointwise_feasible(
    float original, float reconstructed, double tolerance)
{
  return std::isfinite(reconstructed) &&
      pointwise_error(original, reconstructed) <= tolerance;
}

// The current correction serializer uses numeric equality to decide whether a
// point is present in its mask. Consequently a value numerically equal to base
// but bitwise different (notably signed zero) cannot be reconstructed by an
// exact escape. Such an action is intentionally excluded here.
inline bool decoder_representable(float base, float value)
{
  return base != value || bitwise_equal(base, value);
}

inline int infer_grid_code(float base, float value, float scale)
{
  if (base == value || !(scale > 0.0F) || !std::isfinite(scale))
    return 0;
  const auto ratio =
      (static_cast<double>(value) - static_cast<double>(base)) /
      static_cast<double>(scale);
  if (!std::isfinite(ratio) || ratio < -127.5 || ratio > 127.5)
    return 0;
  const auto rounded = std::llround(ratio);
  if (rounded < -127 || rounded > 127 || rounded == 0)
    return 0;
  const auto decoded = decode_grid_value(base, static_cast<int>(rounded), scale);
  return bitwise_equal(decoded, value) ? static_cast<int>(rounded) : 0;
}

inline bool source_is_constant(const std::vector<float>& original)
{
  long double mean = 0.0L;
  for (const auto value : original)
    mean += value;
  mean /= static_cast<long double>(original.size());
  long double energy = 0.0L;
  for (const auto value : original) {
    const auto centered = static_cast<long double>(value) - mean;
    energy += centered * centered;
  }
  return energy <= qpet_correlation::kDenominatorFloor;
}

inline bool numeric_equal(
    const std::vector<float>& left, const std::vector<float>& right)
{
  return left.size() == right.size() &&
      std::equal(left.begin(), left.end(), right.begin());
}

inline void validate_constraints(std::size_t time, const Constraints& constraints)
{
  if (time == 0)
    throw std::invalid_argument("SSBB requires a nonempty series");
  if (!(constraints.pointwise_tolerance > 0.0) ||
      !std::isfinite(constraints.pointwise_tolerance))
    throw std::invalid_argument("SSBB pointwise tolerance must be finite and positive");
  const auto acf_enabled = constraints.acf_tolerance != -1.0;
  const auto pearson_enabled = constraints.minimum_correlation != -1.0;
  if (!acf_enabled && !pearson_enabled)
    throw std::invalid_argument("SSBB requires ACF or Pearson to be enabled");
  if (acf_enabled &&
      (!(constraints.acf_tolerance > 0.0) ||
       !std::isfinite(constraints.acf_tolerance)))
    throw std::invalid_argument("SSBB ACF tolerance must be -1 or finite positive");
  if (pearson_enabled &&
      (!(constraints.minimum_correlation > 0.0) ||
       constraints.minimum_correlation > 1.0 ||
       !std::isfinite(constraints.minimum_correlation)))
    throw std::invalid_argument(
        "SSBB minimum correlation must be -1 or finite in (0,1]");
  if (acf_enabled &&
      (time < 3 || constraints.max_lag == 0 ||
       constraints.max_lag > time - 2))
    throw std::invalid_argument("SSBB max_lag must be in [1,time-2]");
  if (pearson_enabled && time < 2)
    throw std::invalid_argument("SSBB Pearson requires at least two values");
}

inline void validate_budget(
    const SearchBudget& budget, std::size_t time)
{
  if (budget.feasible_beam_width == 0 || budget.infeasible_beam_width == 0)
    throw std::invalid_argument("SSBB must retain both beam strata");
  if (budget.output_limit == 0 ||
      budget.output_limit > kExhaustiveOracleCombinationLimit)
    throw std::invalid_argument("SSBB output limit is invalid");
  const auto total_width = checked_add(
      budget.feasible_beam_width,
      budget.infeasible_beam_width,
      "SSBB total beam width overflow");
  if (total_width > kExhaustiveOracleCombinationLimit)
    throw std::invalid_argument("SSBB test beam exceeds the explicit safety cap");
  (void)checked_multiply(
      total_width, time, "SSBB beam tensor element budget overflow");
}

inline qpet_correlation::ConstraintReport direct_report(
    const std::vector<float>& original,
    const std::vector<float>& reconstructed,
    const Constraints& constraints)
{
  return qpet_correlation::verify_constraints(
      original,
      reconstructed,
      1,
      original.size(),
      constraints.pointwise_tolerance,
      constraints.acf_tolerance,
      constraints.minimum_correlation,
      constraints.max_lag);
}

struct ConstraintContext {
  Constraints constraints;
  std::vector<double> source_acf;
  bool constant_source = false;
};

inline ConstraintContext make_context(
    const std::vector<float>& original, const Constraints& constraints)
{
  ConstraintContext context;
  context.constraints = constraints;
  context.constant_source = source_is_constant(original);
  if (constraints.acf_tolerance >= 0.0) {
    context.source_acf.reserve(constraints.max_lag);
    for (std::size_t lag = 1; lag <= constraints.max_lag; ++lag)
      context.source_acf.push_back(
          qpet_correlation::acf(original.data(), original.size(), lag));
  }
  return context;
}

inline double statistics_violation(
    const std::vector<float>& original,
    const std::vector<float>& reconstructed,
    const SufficientStatistics& statistics,
    const ConstraintContext& context)
{
  double violation = 0.0;
  for (std::size_t index = 0; index < original.size(); ++index) {
    violation = std::max(
        violation,
        pointwise_error(original[index], reconstructed[index]) /
            context.constraints.pointwise_tolerance);
  }
  if (context.constraints.acf_tolerance >= 0.0) {
    for (std::size_t lag = 1; lag <= context.constraints.max_lag; ++lag) {
      const auto difference = std::abs(
          context.source_acf[lag - 1] - statistics.acf_correlation(lag));
      violation = std::max(
          violation, difference / context.constraints.acf_tolerance);
    }
  }
  if (context.constraints.minimum_correlation >= 0.0) {
    const auto correlation = context.constant_source
        ? (numeric_equal(original, reconstructed) ? 1.0 : 0.0)
        : statistics.pearson_correlation();
    if (context.constraints.minimum_correlation < 1.0) {
      violation = std::max(
          violation,
          std::max(0.0, 1.0 - correlation) /
              (1.0 - context.constraints.minimum_correlation));
    }
    else if (correlation < 1.0) {
      violation = std::numeric_limits<double>::infinity();
    }
  }
  return std::isnan(violation)
      ? std::numeric_limits<double>::infinity()
      : violation;
}

inline RateProxy make_rate_proxy(
    const std::vector<float>& base,
    const std::vector<float>& reconstructed,
    float scale,
    double constraint_violation)
{
  if (base.size() != reconstructed.size())
    throw std::invalid_argument("SSBB proxy vectors have different dimensions");
  RateProxy proxy;
  proxy.constraint_violation = constraint_violation;
  for (std::size_t index = 0; index < base.size(); ++index) {
    if (base[index] == reconstructed[index])
      continue;
    checked_increment(proxy.changed_values, "SSBB proxy change count overflow");
    const auto code = infer_grid_code(base[index], reconstructed[index], scale);
    if (code == 0) {
      checked_increment(proxy.exact_escapes, "SSBB proxy escape count overflow");
      continue;
    }
    const auto encoded = zigzag(code);
    proxy.maximum_bit_width = std::max(
        proxy.maximum_bit_width, required_bit_width(encoded));
    if (encoded > std::numeric_limits<std::uint64_t>::max() - proxy.zigzag_sum)
      throw std::overflow_error("SSBB proxy zigzag sum overflow");
    proxy.zigzag_sum += encoded;
  }
  return proxy;
}

inline bool rate_less(const RateProxy& left, const RateProxy& right)
{
  return std::tie(
             left.exact_escapes,
             left.maximum_bit_width,
             left.changed_values,
             left.zigzag_sum) <
      std::tie(
             right.exact_escapes,
             right.maximum_bit_width,
             right.changed_values,
             right.zigzag_sum);
}

inline bool candidate_less(const Candidate& left, const Candidate& right)
{
  if (rate_less(left.proxy, right.proxy))
    return true;
  if (rate_less(right.proxy, left.proxy))
    return false;
  if (left.proxy.constraint_violation != right.proxy.constraint_violation)
    return left.proxy.constraint_violation < right.proxy.constraint_violation;
  if (left.action_ordinals != right.action_ordinals)
    return left.action_ordinals < right.action_ordinals;
  return bits_lexicographically_less(left.reconstruction, right.reconstruction);
}

inline void append_unique_action(std::vector<Action>& actions, Action action)
{
  const auto duplicate = std::find_if(
      actions.begin(), actions.end(), [&](const Action& retained) {
        return bitwise_equal(retained.value, action.value);
      });
  if (duplicate == actions.end())
    actions.push_back(action);
}

inline Action closest_grid_action(
    float original,
    float base,
    float scale,
    double pointwise_tolerance,
    std::uint8_t bit_width,
    bool& found)
{
  if (bit_width < 1 || bit_width > 8)
    throw std::invalid_argument("SSBB packed bit width must be in [1,8]");
  found = false;
  Action best;
  auto best_error = std::numeric_limits<double>::infinity();
  const auto maximum_zigzag =
      static_cast<std::uint16_t>((1U << bit_width) - 1U);
  for (auto code = -127; code <= 127; ++code) {
    if (code == 0 || zigzag(code) > maximum_zigzag)
      continue;
    const auto value = decode_grid_value(base, code, scale);
    if (!pointwise_feasible(original, value, pointwise_tolerance) ||
        !decoder_representable(base, value))
      continue;
    const auto error = pointwise_error(original, value);
    const auto candidate_key = std::tuple<double, int, std::uint16_t, int>{
        error, std::abs(code), zigzag(code), code};
    const auto best_key = std::tuple<double, int, std::uint16_t, int>{
        best_error,
        std::abs(static_cast<int>(best.code)),
        found ? zigzag(static_cast<int>(best.code)) : std::uint16_t{0},
        static_cast<int>(best.code)};
    if (!found || candidate_key < best_key) {
      found = true;
      best.kind = ActionKind::grid;
      best.value = value;
      best.code = static_cast<std::int16_t>(code);
      best.bit_width = bit_width;
      best_error = error;
    }
  }
  return best;
}

inline std::size_t hash_values(const std::vector<float>& values)
{
  std::uint64_t hash = 1469598103934665603ULL;
  for (const auto value : values) {
    auto bits = float_bits(value);
    for (std::size_t byte = 0; byte < sizeof(bits); ++byte) {
      hash ^= static_cast<unsigned char>((bits >> (8U * byte)) & 0xffU);
      hash *= 1099511628211ULL;
    }
  }
  return static_cast<std::size_t>(hash);
}

struct State {
  std::vector<float> reconstruction;
  std::vector<std::uint8_t> decisions;
  SufficientStatistics statistics;
  RateProxy proxy;
};

inline bool state_less(const State& left, const State& right, bool feasible_stratum)
{
  if (!feasible_stratum &&
      left.proxy.constraint_violation != right.proxy.constraint_violation)
    return left.proxy.constraint_violation < right.proxy.constraint_violation;
  if (rate_less(left.proxy, right.proxy))
    return true;
  if (rate_less(right.proxy, left.proxy))
    return false;
  if (feasible_stratum &&
      left.proxy.constraint_violation != right.proxy.constraint_violation)
    return left.proxy.constraint_violation < right.proxy.constraint_violation;
  if (left.decisions != right.decisions)
    return left.decisions < right.decisions;
  return bits_lexicographically_less(left.reconstruction, right.reconstruction);
}

inline std::vector<State> deduplicate_states(
    std::vector<State> states, Diagnostics& diagnostics)
{
  std::vector<State> unique;
  unique.reserve(states.size());
  std::unordered_map<std::size_t, std::vector<std::size_t>> buckets;
  buckets.reserve(states.size());
  for (auto& state : states) {
    const auto hash = hash_values(state.reconstruction);
    auto& bucket = buckets[hash];
    auto duplicate = unique.size();
    for (const auto retained_index : bucket) {
      if (bitwise_equal(
              unique[retained_index].reconstruction, state.reconstruction)) {
        duplicate = retained_index;
        break;
      }
    }
    if (duplicate != unique.size()) {
      checked_increment(
          diagnostics.states_deduplicated,
          "SSBB deduplicated-state count overflow");
      if (state.decisions < unique[duplicate].decisions)
        unique[duplicate] = std::move(state);
      continue;
    }
    bucket.push_back(unique.size());
    unique.push_back(std::move(state));
  }
  return unique;
}

inline void retain_stratum(
    std::vector<State>& states,
    std::size_t width,
    bool feasible_stratum,
    Diagnostics& diagnostics)
{
  std::stable_sort(
      states.begin(), states.end(), [&](const State& left, const State& right) {
        return state_less(left, right, feasible_stratum);
      });
  const auto retained = std::min(width, states.size());
  checked_accumulate(
      diagnostics.states_pruned_by_beam,
      states.size() - retained,
      "SSBB beam-pruned state count overflow");
  if (feasible_stratum) {
    checked_accumulate(
        diagnostics.feasible_states_retained,
        retained,
        "SSBB feasible retained-state count overflow");
  }
  else {
    checked_accumulate(
        diagnostics.infeasible_states_retained,
        retained,
        "SSBB infeasible retained-state count overflow");
  }
  states.resize(retained);
}

inline void validate_input(
    const std::vector<float>& original,
    const std::vector<float>& base,
    const std::vector<float>& feasible_start,
    float scale,
    const Constraints& constraints,
    const SearchBudget& budget)
{
  if (original.empty() || original.size() != base.size() ||
      original.size() != feasible_start.size())
    throw std::invalid_argument("SSBB input vectors have invalid dimensions");
  validate_constraints(original.size(), constraints);
  validate_budget(budget, original.size());
  if (!(scale >= 0.0F) || !std::isfinite(scale))
    throw std::invalid_argument("SSBB scale must be finite and nonnegative");
  for (std::size_t index = 0; index < original.size(); ++index) {
    if (!std::isfinite(original[index]) || !std::isfinite(base[index]) ||
        !std::isfinite(feasible_start[index]))
      throw std::invalid_argument("SSBB input contains a non-finite float32 value");
    if (!pointwise_feasible(
            original[index],
            feasible_start[index],
            constraints.pointwise_tolerance))
      throw std::invalid_argument("SSBB feasible_start violates the strict L-infinity bound");
    if (!decoder_representable(base[index], feasible_start[index]))
      throw std::invalid_argument(
          "SSBB feasible_start contains a correction-decoder signed-zero mismatch");
  }
  if (!direct_report(original, feasible_start, constraints).all_ok())
    throw std::invalid_argument("SSBB feasible_start violates a hard constraint");
}

}  // namespace detail

inline std::vector<Action> make_position_actions(
    float original,
    float base,
    float feasible_start,
    float scale,
    double pointwise_tolerance)
{
  if (!std::isfinite(original) || !std::isfinite(base) ||
      !std::isfinite(feasible_start) || !(scale >= 0.0F) ||
      !std::isfinite(scale) || !(pointwise_tolerance > 0.0) ||
      !std::isfinite(pointwise_tolerance))
    throw std::invalid_argument("SSBB action input is invalid");
  if (!detail::pointwise_feasible(original, feasible_start, pointwise_tolerance) ||
      !detail::decoder_representable(base, feasible_start))
    throw std::invalid_argument("SSBB keep action is not decoder-feasible");

  auto leading = std::vector<Action>();
  detail::append_unique_action(
      leading, Action{ActionKind::keep, feasible_start, 0, 0, 0});
  if (detail::pointwise_feasible(original, base, pointwise_tolerance))
    detail::append_unique_action(
        leading, Action{ActionKind::base, base, 0, 0, 0});

  auto exact = std::vector<Action>();
  if (detail::decoder_representable(base, original))
    detail::append_unique_action(
        exact, Action{ActionKind::exact, original, 0, 0, 0});

  auto grids = std::vector<Action>();
  if (scale > 0.0F) {
    for (std::uint8_t width = 1; width <= 8; ++width) {
      bool found = false;
      auto candidate = detail::closest_grid_action(
          original, base, scale, pointwise_tolerance, width, found);
      if (!found)
        continue;
      const auto duplicate = std::find_if(
          grids.begin(), grids.end(), [&](const Action& retained) {
            return detail::bitwise_equal(retained.value, candidate.value);
          });
      if (duplicate == grids.end())
        grids.push_back(candidate);
      else if (candidate.bit_width < duplicate->bit_width)
        duplicate->bit_width = candidate.bit_width;
    }
  }

  // Reserve all distinct keep/base/exact actions. Remaining slots represent
  // the low-bit endpoint and the closest-to-original endpoint of the grid
  // family. This keeps K<=5 while exposing a genuine rate/distortion choice.
  auto mandatory = leading;
  for (const auto& action : exact)
    detail::append_unique_action(mandatory, action);
  if (mandatory.size() > kMaximumActionsPerPosition)
    throw std::logic_error("SSBB mandatory action family exceeded K=5");
  grids.erase(
      std::remove_if(
          grids.begin(),
          grids.end(),
          [&](const Action& grid) {
            return std::any_of(
                mandatory.begin(), mandatory.end(), [&](const Action& action) {
                  return detail::bitwise_equal(action.value, grid.value);
                });
          }),
      grids.end());

  std::stable_sort(
      grids.begin(), grids.end(), [&](const Action& left, const Action& right) {
        const auto left_error = detail::pointwise_error(original, left.value);
        const auto right_error = detail::pointwise_error(original, right.value);
        return std::tie(left.bit_width, left_error, left.code) <
            std::tie(right.bit_width, right_error, right.code);
      });
  const auto grid_slots = kMaximumActionsPerPosition - mandatory.size();
  auto selected_grids = std::vector<Action>();
  if (grid_slots != 0 && !grids.empty()) {
    selected_grids.push_back(grids.front());
    if (selected_grids.size() < grid_slots) {
      const auto closest = std::min_element(
          grids.begin(), grids.end(), [&](const Action& left, const Action& right) {
            const auto left_error = detail::pointwise_error(original, left.value);
            const auto right_error = detail::pointwise_error(original, right.value);
            return std::tie(left_error, left.bit_width, left.code) <
                std::tie(right_error, right.bit_width, right.code);
          });
      if (closest != grids.end() &&
          !detail::bitwise_equal(closest->value, selected_grids.front().value))
        selected_grids.push_back(*closest);
    }
    for (const auto& grid : grids) {
      if (selected_grids.size() == grid_slots)
        break;
      detail::append_unique_action(selected_grids, grid);
    }
  }

  auto actions = leading;
  for (const auto& grid : selected_grids)
    detail::append_unique_action(actions, grid);
  for (const auto& action : exact)
    detail::append_unique_action(actions, action);
  if (actions.empty() || actions.size() > kMaximumActionsPerPosition)
    throw std::logic_error("SSBB action family is empty or exceeded K=5");
  for (std::size_t index = 0; index < actions.size(); ++index) {
    if (index > std::numeric_limits<std::uint8_t>::max())
      throw std::overflow_error("SSBB action ordinal exceeds uint8");
    actions[index].ordinal = static_cast<std::uint8_t>(index);
  }
  return actions;
}

inline std::vector<PositionActions> make_position_plan(
    const std::vector<float>& original,
    const std::vector<float>& base,
    const std::vector<float>& feasible_start,
    float scale,
    double pointwise_tolerance)
{
  if (original.empty() || original.size() != base.size() ||
      original.size() != feasible_start.size())
    throw std::invalid_argument("SSBB action-plan vectors have invalid dimensions");
  auto positions = std::vector<PositionActions>();
  positions.reserve(original.size());
  for (std::size_t index = 0; index < original.size(); ++index) {
    positions.push_back(PositionActions{
        index,
        make_position_actions(
            original[index],
            base[index],
            feasible_start[index],
            scale,
            pointwise_tolerance)});
  }
  return positions;
}

inline double statistics_constraint_violation(
    const std::vector<float>& original,
    const std::vector<float>& reconstructed,
    const SufficientStatistics& statistics,
    const Constraints& constraints)
{
  detail::validate_constraints(original.size(), constraints);
  if (original.size() != reconstructed.size() ||
      statistics.count() != original.size() ||
      statistics.max_lag() !=
          (constraints.acf_tolerance >= 0.0 ? constraints.max_lag : 0))
    throw std::invalid_argument("SSBB statistic score dimensions are invalid");
  return detail::statistics_violation(
      original,
      reconstructed,
      statistics,
      detail::make_context(original, constraints));
}

inline SearchResult search(
    const std::vector<float>& original,
    const std::vector<float>& base,
    const std::vector<float>& feasible_start,
    float scale,
    const Constraints& constraints,
    const SearchBudget& budget = production_budget())
{
  detail::validate_input(
      original, base, feasible_start, scale, constraints, budget);
  SearchResult result;
  result.diagnostics.configured_feasible_width = budget.feasible_beam_width;
  result.diagnostics.configured_infeasible_width = budget.infeasible_beam_width;
  result.diagnostics.configured_output_limit = budget.output_limit;
  result.positions = make_position_plan(
      original, base, feasible_start, scale, constraints.pointwise_tolerance);
  for (const auto& position : result.positions)
    detail::checked_accumulate(
        result.diagnostics.actions_generated,
        position.actions.size(),
        "SSBB generated-action count overflow");

  const auto effective_lag =
      constraints.acf_tolerance >= 0.0 ? constraints.max_lag : std::size_t{0};
  const auto context = detail::make_context(original, constraints);
  detail::State root{
      feasible_start,
      {},
      SufficientStatistics::build(original, feasible_start, effective_lag),
      {}};
  root.decisions.reserve(original.size());
  root.proxy = detail::make_rate_proxy(
      base,
      root.reconstruction,
      scale,
      detail::statistics_violation(
          original, root.reconstruction, root.statistics, context));
  auto beam = std::vector<detail::State>{std::move(root)};

  for (const auto& position : result.positions) {
    detail::checked_increment(
        result.diagnostics.positions_processed,
        "SSBB processed-position count overflow");
    const auto child_capacity = detail::checked_multiply(
        beam.size(),
        position.actions.size(),
        "SSBB child-state capacity overflow");
    auto children = std::vector<detail::State>();
    children.reserve(child_capacity);
    for (const auto& parent : beam) {
      for (const auto& action : position.actions) {
        auto child = parent;
        child.statistics.replace(
            original,
            child.reconstruction,
            position.index,
            action.value);
        if (child.decisions.size() == child.decisions.max_size())
          throw std::overflow_error("SSBB decision vector size overflow");
        child.decisions.push_back(action.ordinal);
        child.proxy = detail::make_rate_proxy(
            base,
            child.reconstruction,
            scale,
            detail::statistics_violation(
                original, child.reconstruction, child.statistics, context));
        children.push_back(std::move(child));
      }
    }
    detail::checked_accumulate(
        result.diagnostics.states_generated,
        children.size(),
        "SSBB generated-state count overflow");
    auto unique = detail::deduplicate_states(
        std::move(children), result.diagnostics);
    auto feasible = std::vector<detail::State>();
    auto infeasible = std::vector<detail::State>();
    feasible.reserve(unique.size());
    infeasible.reserve(unique.size());
    for (auto& state : unique) {
      // This is a stratum/rank decision only. An infeasible partial state is
      // retained and may recover after later positions are changed.
      if (state.proxy.constraint_violation <= 1.0)
        feasible.push_back(std::move(state));
      else
        infeasible.push_back(std::move(state));
    }
    detail::checked_accumulate(
        result.diagnostics.statistics_feasible_states,
        feasible.size(),
        "SSBB statistics-feasible state count overflow");
    detail::checked_accumulate(
        result.diagnostics.statistics_infeasible_states,
        infeasible.size(),
        "SSBB statistics-infeasible state count overflow");
    detail::retain_stratum(
        feasible,
        budget.feasible_beam_width,
        true,
        result.diagnostics);
    detail::retain_stratum(
        infeasible,
        budget.infeasible_beam_width,
        false,
        result.diagnostics);
    const auto retained = detail::checked_add(
        feasible.size(), infeasible.size(), "SSBB retained beam size overflow");
    beam.clear();
    beam.reserve(retained);
    for (auto& state : feasible)
      beam.push_back(std::move(state));
    for (auto& state : infeasible)
      beam.push_back(std::move(state));
    if (beam.empty())
      throw std::logic_error("SSBB beam unexpectedly lost both strata");
  }

  result.diagnostics.terminal_states = beam.size();
  auto directly_feasible = std::vector<Candidate>();
  directly_feasible.reserve(beam.size());
  for (auto& state : beam) {
    const auto report = detail::direct_report(
        original, state.reconstruction, constraints);
    if (!report.all_ok()) {
      detail::checked_increment(
          result.diagnostics.terminal_direct_infeasible,
          "SSBB terminal-infeasible count overflow");
      continue;
    }
    detail::checked_increment(
        result.diagnostics.terminal_direct_feasible,
        "SSBB terminal-feasible count overflow");
    directly_feasible.push_back(Candidate{
        std::move(state.reconstruction),
        std::move(state.decisions),
        state.proxy,
        report});
  }
  std::stable_sort(
      directly_feasible.begin(),
      directly_feasible.end(),
      detail::candidate_less);
  if (directly_feasible.size() > budget.output_limit)
    directly_feasible.resize(budget.output_limit);
  result.candidates = std::move(directly_feasible);
  result.diagnostics.outputs_returned = result.candidates.size();

  const auto observed = detail::checked_add(
      result.diagnostics.statistics_feasible_states,
      result.diagnostics.statistics_infeasible_states,
      "SSBB observed-state accounting overflow");
  const auto retained = detail::checked_add(
      result.diagnostics.feasible_states_retained,
      result.diagnostics.infeasible_states_retained,
      "SSBB retained-state accounting overflow");
  if (result.diagnostics.states_generated !=
          detail::checked_add(
              result.diagnostics.states_deduplicated,
              observed,
              "SSBB generated-state accounting overflow") ||
      observed != detail::checked_add(
          retained,
          result.diagnostics.states_pruned_by_beam,
          "SSBB beam-state accounting overflow") ||
      result.diagnostics.terminal_states !=
          detail::checked_add(
              result.diagnostics.terminal_direct_feasible,
              result.diagnostics.terminal_direct_infeasible,
              "SSBB terminal-state accounting overflow"))
    throw std::logic_error("SSBB diagnostics do not form exact partitions");
  return result;
}

// Finite-domain oracle for tests and disclosed small examples only. It
// minimizes RateProxy over this header's finite action product; it neither
// predicts nor claims optimal compressed bytes. Calls exceeding the fixed
// 2,000,000-combination cap fail explicitly before enumeration.
inline ExhaustiveOracleResult exhaustive_proxy_oracle(
    const std::vector<float>& original,
    const std::vector<float>& base,
    const std::vector<float>& feasible_start,
    float scale,
    const Constraints& constraints)
{
  const auto test_budget = production_budget();
  detail::validate_input(
      original, base, feasible_start, scale, constraints, test_budget);
  const auto positions = make_position_plan(
      original, base, feasible_start, scale, constraints.pointwise_tolerance);
  std::size_t combinations = 1;
  for (const auto& position : positions) {
    if (position.actions.empty())
      throw std::logic_error("SSBB oracle received an empty action family");
    if (combinations >
        kExhaustiveOracleCombinationLimit / position.actions.size())
      throw std::length_error(
          "SSBB exhaustive oracle exceeds 2,000,000 combinations");
    combinations *= position.actions.size();
  }
  if (combinations > kExhaustiveOracleCombinationLimit)
    throw std::length_error(
        "SSBB exhaustive oracle exceeds 2,000,000 combinations");

  ExhaustiveOracleResult result;
  result.combinations = combinations;
  const auto effective_lag =
      constraints.acf_tolerance >= 0.0 ? constraints.max_lag : std::size_t{0};
  const auto context = detail::make_context(original, constraints);
  for (std::size_t combination = 0; combination < combinations; ++combination) {
    auto reconstruction = feasible_start;
    auto statistics = SufficientStatistics::build(
        original, reconstruction, effective_lag);
    auto decisions = std::vector<std::uint8_t>();
    decisions.reserve(positions.size());
    auto cursor = combination;
    for (const auto& position : positions) {
      const auto choice = cursor % position.actions.size();
      cursor /= position.actions.size();
      const auto& action = position.actions[choice];
      statistics.replace(
          original, reconstruction, position.index, action.value);
      decisions.push_back(action.ordinal);
    }
    const auto proxy = detail::make_rate_proxy(
        base,
        reconstruction,
        scale,
        detail::statistics_violation(
            original, reconstruction, statistics, context));
    const auto report = detail::direct_report(
        original, reconstruction, constraints);
    if (!report.all_ok()) {
      detail::checked_increment(
          result.direct_infeasible,
          "SSBB oracle infeasible count overflow");
      continue;
    }
    detail::checked_increment(
        result.direct_feasible,
        "SSBB oracle feasible count overflow");
    Candidate candidate{
        std::move(reconstruction), std::move(decisions), proxy, report};
    if (!result.has_feasible || detail::candidate_less(candidate, result.best)) {
      result.best = std::move(candidate);
      result.has_feasible = true;
    }
  }
  if (!result.has_feasible ||
      result.combinations != detail::checked_add(
          result.direct_feasible,
          result.direct_infeasible,
          "SSBB oracle accounting overflow"))
    throw std::logic_error("SSBB exhaustive oracle lost its feasible incumbent");
  return result;
}

}  // namespace qpet_correlation::ssbb

#endif  // QPET_CORRELATION_SEQUENCE_BEAM_SEARCH_HPP
