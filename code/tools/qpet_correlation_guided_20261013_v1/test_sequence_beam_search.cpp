#include "SequenceBeamSearch.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace ssbb = qpet_correlation::ssbb;

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

bool same_bits(
    const std::vector<float>& left, const std::vector<float>& right)
{
  return left.size() == right.size() &&
      (left.empty() ||
       std::memcmp(left.data(), right.data(), left.size() * sizeof(float)) == 0);
}

void require_close(double observed, double expected, double tolerance, const char* message)
{
  if (!(std::abs(observed - expected) <= tolerance))
    throw std::runtime_error(
        std::string(message) + ": observed=" + std::to_string(observed) +
        ", expected=" + std::to_string(expected));
}

void require_directly_feasible(
    const std::vector<float>& original,
    const ssbb::Constraints& constraints,
    const std::vector<ssbb::Candidate>& candidates)
{
  for (const auto& candidate : candidates) {
    require(candidate.reconstruction.size() == original.size(),
            "SSBB candidate has the wrong length");
    for (std::size_t index = 0; index < original.size(); ++index) {
      require(
          std::abs(
              static_cast<double>(original[index]) -
              static_cast<double>(candidate.reconstruction[index])) <=
              constraints.pointwise_tolerance,
          "SSBB candidate violated the strict pointwise tolerance");
    }
    const auto report = qpet_correlation::verify_constraints(
        original,
        candidate.reconstruction,
        1,
        original.size(),
        constraints.pointwise_tolerance,
        constraints.acf_tolerance,
        constraints.minimum_correlation,
        constraints.max_lag);
    require(report.all_ok(), "SSBB output failed canonical direct verification");
    require(candidate.report.all_ok(), "SSBB stored a failed terminal report");
  }
}

void require_proxy_equal(
    const ssbb::RateProxy& left, const ssbb::RateProxy& right)
{
  require(left.exact_escapes == right.exact_escapes,
          "SSBB proxy exact-escape count changed");
  require(left.maximum_bit_width == right.maximum_bit_width,
          "SSBB proxy bit width changed");
  require(left.changed_values == right.changed_values,
          "SSBB proxy changed-value count changed");
  require(left.zigzag_sum == right.zigzag_sum,
          "SSBB proxy zigzag sum changed");
  require(left.constraint_violation == right.constraint_violation,
          "SSBB proxy violation changed");
}

void require_diagnostics_equal(
    const ssbb::Diagnostics& left, const ssbb::Diagnostics& right)
{
  const auto left_values = std::tie(
      left.positions_processed,
      left.actions_generated,
      left.states_generated,
      left.states_deduplicated,
      left.statistics_feasible_states,
      left.statistics_infeasible_states,
      left.feasible_states_retained,
      left.infeasible_states_retained,
      left.states_pruned_by_beam,
      left.terminal_states,
      left.terminal_direct_feasible,
      left.terminal_direct_infeasible,
      left.outputs_returned,
      left.configured_feasible_width,
      left.configured_infeasible_width,
      left.configured_output_limit);
  const auto right_values = std::tie(
      right.positions_processed,
      right.actions_generated,
      right.states_generated,
      right.states_deduplicated,
      right.statistics_feasible_states,
      right.statistics_infeasible_states,
      right.feasible_states_retained,
      right.infeasible_states_retained,
      right.states_pruned_by_beam,
      right.terminal_states,
      right.terminal_direct_feasible,
      right.terminal_direct_infeasible,
      right.outputs_returned,
      right.configured_feasible_width,
      right.configured_infeasible_width,
      right.configured_output_limit);
  require(left_values == right_values, "SSBB diagnostics are nondeterministic");
}

void require_diagnostic_partitions(const ssbb::Diagnostics& diagnostics)
{
  require(
      diagnostics.states_generated ==
          diagnostics.states_deduplicated +
              diagnostics.statistics_feasible_states +
              diagnostics.statistics_infeasible_states,
      "SSBB generated-state diagnostics do not form an exact partition");
  require(
      diagnostics.statistics_feasible_states +
              diagnostics.statistics_infeasible_states ==
          diagnostics.feasible_states_retained +
              diagnostics.infeasible_states_retained +
              diagnostics.states_pruned_by_beam,
      "SSBB retained/pruned diagnostics do not form an exact partition");
  require(
      diagnostics.terminal_states ==
          diagnostics.terminal_direct_feasible +
              diagnostics.terminal_direct_infeasible,
      "SSBB terminal diagnostics do not form an exact partition");
}

void test_sufficient_statistics()
{
  constexpr std::size_t time = 17;
  constexpr std::size_t max_lag = 5;
  auto original = std::vector<float>(time);
  auto reconstructed = std::vector<float>(time);
  for (std::size_t index = 0; index < time; ++index) {
    original[index] = static_cast<float>(
        0.7 * std::sin(0.31 * static_cast<double>(index)) +
        0.04 * static_cast<double>(index));
    reconstructed[index] = static_cast<float>(
        original[index] + 0.03 * std::cos(0.47 * static_cast<double>(index)));
  }
  auto statistics = ssbb::SufficientStatistics::build(
      original, reconstructed, max_lag);
  auto check = [&] {
    require_close(
        statistics.pearson_correlation(),
        qpet_correlation::pearson(
            original.data(), reconstructed.data(), reconstructed.size()),
        2.0e-12,
        "SSBB incremental Pearson disagrees with direct Pearson");
    for (std::size_t lag = 1; lag <= max_lag; ++lag) {
      require_close(
          statistics.acf_correlation(lag),
          qpet_correlation::acf(reconstructed.data(), reconstructed.size(), lag),
          2.0e-12,
          "SSBB incremental ACF disagrees with direct ACF");
    }
  };
  check();
  for (const auto& [index, value] :
       std::vector<std::pair<std::size_t, float>>{
           {0, original[0]},
           {time - 1, static_cast<float>(original.back() - 0.02F)},
           {7, static_cast<float>(original[7] + 0.01F)},
           {2, reconstructed[2]},
           {11, static_cast<float>(original[11] - 0.015F)}}) {
    statistics.replace(original, reconstructed, index, value);
    check();
  }
}

void test_action_boundaries()
{
  const auto adjacent = std::nextafter(1.0F, 2.0F);
  const auto epsilon =
      static_cast<double>(adjacent) - static_cast<double>(1.0F);
  const auto adjacent_actions = ssbb::make_position_actions(
      adjacent, 1.0F, 1.0F, epsilon, epsilon);
  require(!adjacent_actions.empty() &&
              adjacent_actions.size() <= ssbb::kMaximumActionsPerPosition,
          "SSBB nextafter action family is invalid");
  for (const auto& action : adjacent_actions) {
    require(
        std::abs(static_cast<double>(adjacent) - action.value) <= epsilon,
        "SSBB nextafter action crossed the exact pointwise boundary");
  }

  // +0 and -0 are numerically equal but the current correction mask cannot
  // carry their bit-only difference. The positive-zero exact action must be
  // canonicalized away rather than reaching a later bitwise replay failure.
  const auto signed_zero_actions = ssbb::make_position_actions(
      0.0F, -0.0F, -0.0F, 1.0F, 1.0e-6);
  require(!signed_zero_actions.empty(), "SSBB signed-zero action family is empty");
  for (const auto& action : signed_zero_actions)
    require(bits(action.value) == bits(-0.0F),
            "SSBB emitted an unserializable signed-zero action");

  // No nonzero scale-grid code falls in [X-epsilon,X+epsilon]. The feasible
  // start/exact value is therefore an unavoidable format-4/6 escape proxy.
  const auto forced = ssbb::make_position_actions(
      0.5F, 0.0F, 0.5F, 1.0F, 0.1);
  require(forced.size() == 1 && bits(forced.front().value) == bits(0.5F),
          "SSBB forced-exact action was not retained uniquely");
}

struct Fixture {
  std::vector<float> original;
  std::vector<float> base;
  std::vector<float> start;
  float scale = 0.0F;
  ssbb::Constraints constraints;
};

Fixture make_oracle_fixture()
{
  Fixture fixture;
  fixture.original = {0.10F, 0.82F, 0.24F, -0.43F, 0.31F};
  fixture.base = {-0.15F, 1.02F, -0.06F, -0.18F, 0.11F};
  fixture.start = fixture.original;
  fixture.scale = 0.0625F;
  fixture.constraints.pointwise_tolerance = 0.35;
  fixture.constraints.acf_tolerance = 0.35;
  fixture.constraints.minimum_correlation = 0.90;
  fixture.constraints.max_lag = 2;
  return fixture;
}

void test_wide_beam_matches_finite_oracle()
{
  const auto fixture = make_oracle_fixture();
  const auto oracle = ssbb::exhaustive_proxy_oracle(
      fixture.original,
      fixture.base,
      fixture.start,
      fixture.scale,
      fixture.constraints);
  require(oracle.has_feasible && oracle.combinations > 1,
          "SSBB finite oracle did not enumerate a nontrivial domain");
  require(oracle.combinations <= ssbb::kExhaustiveOracleCombinationLimit,
          "SSBB finite oracle exceeded its cap");

  // Each stratum is wider than the complete finite product, so this test checks
  // implementation equivalence to the proxy oracle. It is not a claim about
  // the production B=64 beam or actual compressed bytes.
  const ssbb::SearchBudget wide{
      oracle.combinations, oracle.combinations, ssbb::kProductionOutputLimit};
  const auto searched = ssbb::search(
      fixture.original,
      fixture.base,
      fixture.start,
      fixture.scale,
      fixture.constraints,
      wide);
  require(!searched.candidates.empty(), "SSBB wide beam returned no candidate");
  require(same_bits(
              searched.candidates.front().reconstruction,
              oracle.best.reconstruction),
          "SSBB wide beam missed the finite proxy oracle optimum");
  require_proxy_equal(searched.candidates.front().proxy, oracle.best.proxy);
  require(searched.positions.size() == fixture.original.size(),
          "SSBB did not process every time position");
  for (std::size_t index = 0; index < searched.positions.size(); ++index) {
    require(searched.positions[index].index == index,
            "SSBB changed the fixed time order");
    require(!searched.positions[index].actions.empty() &&
                searched.positions[index].actions.size() <=
                    ssbb::kMaximumActionsPerPosition,
            "SSBB violated the per-position K=5 action budget");
  }
  require_directly_feasible(
      fixture.original, fixture.constraints, searched.candidates);
  require_diagnostic_partitions(searched.diagnostics);
}

void test_production_budget_and_determinism()
{
  const auto fixture = make_oracle_fixture();
  const auto first = ssbb::search(
      fixture.original,
      fixture.base,
      fixture.start,
      fixture.scale,
      fixture.constraints);
  const auto second = ssbb::search(
      fixture.original,
      fixture.base,
      fixture.start,
      fixture.scale,
      fixture.constraints);
  require(first.diagnostics.configured_feasible_width +
                  first.diagnostics.configured_infeasible_width ==
              ssbb::kProductionTotalBeamWidth,
          "SSBB production search did not use B=64");
  require(first.diagnostics.configured_output_limit ==
              ssbb::kProductionOutputLimit,
          "SSBB production search did not use R=3");
  require(first.diagnostics.positions_processed == fixture.original.size(),
          "SSBB production search skipped a time position");
  require(first.diagnostics.outputs_returned <= ssbb::kProductionOutputLimit,
          "SSBB production search exceeded R=3");
  require(first.diagnostics.statistics_feasible_states > 0 &&
              first.diagnostics.statistics_infeasible_states > 0 &&
              first.diagnostics.feasible_states_retained > 0 &&
              first.diagnostics.infeasible_states_retained > 0,
          "SSBB production fixture did not exercise both beam strata");
  require_diagnostic_partitions(first.diagnostics);
  require_diagnostics_equal(first.diagnostics, second.diagnostics);
  require(first.candidates.size() == second.candidates.size(),
          "SSBB repeated search changed its output count");
  for (std::size_t index = 0; index < first.candidates.size(); ++index) {
    require(same_bits(
                first.candidates[index].reconstruction,
                second.candidates[index].reconstruction),
            "SSBB repeated reconstruction is not bitwise deterministic");
    require(first.candidates[index].action_ordinals ==
                second.candidates[index].action_ordinals,
            "SSBB repeated action path changed");
    require_proxy_equal(first.candidates[index].proxy, second.candidates[index].proxy);
  }
  require_directly_feasible(
      fixture.original, fixture.constraints, first.candidates);
}

void test_constant_and_forced_exact_outputs()
{
  const std::vector<float> constant(5, 2.0F);
  const std::vector<float> constant_base = {2.1F, 1.9F, 2.2F, 1.8F, 2.1F};
  const auto constant_start = constant;
  ssbb::Constraints constant_constraints;
  constant_constraints.pointwise_tolerance = 0.25;
  constant_constraints.acf_tolerance = 2.0;
  constant_constraints.minimum_correlation = 1.0;
  constant_constraints.max_lag = 2;
  const ssbb::SearchBudget wide{5000, 5000, 3};
  const auto constant_result = ssbb::search(
      constant,
      constant_base,
      constant_start,
      0.1F,
      constant_constraints,
      wide);
  require(!constant_result.candidates.empty(),
          "SSBB constant-source search returned no feasible result");
  require_directly_feasible(
      constant, constant_constraints, constant_result.candidates);
  for (const auto& candidate : constant_result.candidates)
    require(std::equal(
                constant.begin(),
                constant.end(),
                candidate.reconstruction.begin()),
            "SSBB violated canonical constant-source Pearson semantics");

  const std::vector<float> original = {0.0F, 0.8F, 0.5F, -0.2F, 0.7F};
  auto base = original;
  base[2] = 0.0F;
  const auto start = original;
  ssbb::Constraints constraints;
  constraints.pointwise_tolerance = 0.1;
  constraints.acf_tolerance = 0.5;
  constraints.minimum_correlation = 0.95;
  constraints.max_lag = 2;
  const auto forced = ssbb::search(
      original, base, start, 1.0F, constraints);
  require(!forced.candidates.empty(),
          "SSBB forced-exact search returned no feasible result");
  require_directly_feasible(original, constraints, forced.candidates);
  for (const auto& candidate : forced.candidates) {
    require(bits(candidate.reconstruction[2]) == bits(original[2]),
            "SSBB forced-exact point changed");
    require(candidate.proxy.exact_escapes >= 1,
            "SSBB proxy failed to record an unavoidable exact escape");
  }
}

void test_oracle_cap_is_fail_closed()
{
  constexpr std::size_t time = 10;
  auto original = std::vector<float>(time);
  auto base = std::vector<float>(time);
  auto start = std::vector<float>(time);
  for (std::size_t index = 0; index < time; ++index) {
    original[index] = static_cast<float>(
        0.3 * std::sin(0.2 * static_cast<double>(index)) +
        0.1 * static_cast<double>(index));
    base[index] = static_cast<float>(original[index] - 1.0F);
    start[index] = static_cast<float>(original[index] - 0.2F);
  }
  ssbb::Constraints constraints;
  constraints.pointwise_tolerance = 1.05;
  constraints.acf_tolerance = 2.0;
  constraints.minimum_correlation = 0.5;
  constraints.max_lag = 2;
  const auto plan = ssbb::make_position_plan(
      original, base, start, 0.125F, constraints.pointwise_tolerance);
  std::size_t combinations = 1;
  for (const auto& position : plan) {
    require(position.actions.size() == ssbb::kMaximumActionsPerPosition,
            "SSBB oracle-cap fixture did not generate K=5");
    combinations *= position.actions.size();
  }
  require(combinations > ssbb::kExhaustiveOracleCombinationLimit,
          "SSBB oracle-cap fixture is not actually over the cap");
  bool rejected = false;
  try {
    (void)ssbb::exhaustive_proxy_oracle(
        original, base, start, 0.125F, constraints);
  }
  catch (const std::length_error& error) {
    rejected =
        std::string(error.what()).find("2,000,000") != std::string::npos;
  }
  require(rejected, "SSBB exhaustive oracle did not fail closed at its cap");
}

}  // namespace

int main()
{
  test_sufficient_statistics();
  test_action_boundaries();
  test_wide_beam_matches_finite_oracle();
  test_production_budget_and_determinism();
  test_constant_and_forced_exact_outputs();
  test_oracle_cap_is_fail_closed();
  return 0;
}
